"""Separated fixed, time/speed ball and candidate-gap risk ablations.

All calibration is frozen and predictor specific. Pointwise scene-block
quantiles are empirical ingredients, not a mission-level probability claim.
The stage4 implementations remain available with their original numerics.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import swept_sphere_clearance
from .stage3_forecasting import observation_speed
from .stage4_risk import (ecp_scene_scores, local_positions, sphere_scene_scores,
                          upper_quantile)
from .stage5_forecasting import canonical_predictor, forecaster_contract


METHODS = ('A0', 'A1', 'A2', 'A3', 'A4', 'A5')
SPEED_BIN_EDGES = (0., .25, float('inf'))
SPEED_BIN_NAMES = ('stationary_observed', 'moving_observed')
SPEED_BIN_METADATA = [0., .25, 'inf']


def speed_bin_indices(speeds, edges=SPEED_BIN_EDGES):
    values, boundaries = np.asarray(speeds, dtype=float), np.asarray(edges, dtype=float)
    if (boundaries.ndim != 1 or len(boundaries) < 2 or boundaries[0] != 0
            or not np.isposinf(boundaries[-1]) or np.any(np.diff(boundaries) <= 0)
            or not np.isfinite(boundaries[:-1]).all()):
        raise ValueError('speed bins must increase from zero to positive infinity')
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('observed speeds must be finite and nonnegative')
    return np.searchsorted(boundaries[1:-1], values, side='right')


def _score_inputs(pred, labels, mask, speeds, edges):
    pred, labels = np.asarray(pred, float), np.asarray(labels, float)
    mask, speeds = np.asarray(mask, bool), np.asarray(speeds, float)
    if (pred.ndim != 5 or pred.shape[-1] != 3 or labels.shape != pred.shape
            or mask.shape != pred.shape[:3] or speeds.shape != mask.shape
            or pred.shape[-2] < 2 or any(n < 1 for n in pred.shape[:3])):
        raise ValueError('scores require [scene,window,object,time,3] and matching mask/speeds')
    if (not np.isfinite(pred[mask]).all() or not np.isfinite(labels[mask]).all()
            or not np.isfinite(speeds[mask]).all() or np.any(speeds[mask] < 0)):
        raise ValueError('observed score slots require finite predictions, labels and speeds')
    bins = speed_bin_indices(np.where(mask, speeds, 0.), edges)
    return (np.where(mask[..., None, None], pred, 0.),
            np.where(mask[..., None, None], labels, 0.), mask, bins)


def time_ball_scores(pred, labels, mask, observed_speeds_mps, edges=SPEED_BIN_EDGES):
    """Return scores[scene,bin,time], present[scene,bin].

    A scene contributes one maximum across its actual observed windows/objects
    in each speed group. Missing groups are NaN, never zero-error samples.
    """
    pred, labels, mask, bins = _score_inputs(pred, labels, mask, observed_speeds_mps, edges)
    errors = np.linalg.norm(pred-labels, axis=-1)
    scores = np.full((len(pred), len(edges)-1, pred.shape[-2]), np.nan)
    present = np.zeros(scores.shape[:2], dtype=bool)
    for b in range(len(edges)-1):
        active = mask & (bins == b)
        present[:, b] = np.any(active, axis=(1, 2))
        block = np.max(np.where(active[..., None], errors, -np.inf), axis=(1, 2))
        scores[present[:, b], b] = block[present[:, b]]
    return scores, present


def time_ecp_scores(path, pred, labels, radii, mask, observed_speeds_mps,
                    edges=SPEED_BIN_EDGES, body_radius=0.):
    """Return candidate-related scores[scene,bin,interval], present[scene,bin].

    Within each speed group, min-object forecast and label swept gaps are taken
    separately, then [forecast_gap-label_gap]+ and max over scenario windows.
    The nearest object may differ between forecast and label. Path is [time,3].
    """
    pred, labels, mask, bins = _score_inputs(pred, labels, mask, observed_speeds_mps, edges)
    path, radii = np.asarray(path, float), np.asarray(radii, float)
    if path.shape != pred.shape[-2:] or not np.isfinite(path).all():
        raise ValueError('candidate must share the finite calibration [time,3] grid')
    if (radii.shape != mask.shape or not np.isfinite(radii[mask]).all()
            or np.any(radii[mask] <= 0) or not np.isfinite(body_radius) or body_radius < 0):
        raise ValueError('swept scores require positive observed radii and nonnegative body radius')
    total_r = np.where(mask, radii, 0.)[..., None]+body_radius
    raw = [swept_sphere_clearance(path[:-1], path[1:], p[..., :-1, :],
                                p[..., 1:, :], total_r) for p in (pred, labels)]
    scores = np.full((len(pred), len(edges)-1, len(path)-1), np.nan)
    present = np.zeros(scores.shape[:2], bool)
    for b in range(len(edges)-1):
        active = mask & (bins == b)
        windows = np.any(active, axis=2)
        present[:, b] = np.any(windows, axis=1)
        nearest = [np.min(np.where(active[..., None], g, np.inf), axis=2) for g in raw]
        difference = np.zeros_like(nearest[0])
        np.subtract(nearest[0], nearest[1], out=difference, where=windows[..., None])
        block = np.max(np.maximum(difference, 0.), axis=1)
        scores[present[:, b], b] = block[present[:, b]]
    return scores, present


def time_quantiles(scores, present, alpha):
    """Finite order statistics per bin/time; an absent bin is an explicit error."""
    scores, present = np.asarray(scores, float), np.asarray(present, bool)
    if scores.ndim != 3 or present.shape != scores.shape[:2]:
        raise ValueError('time quantiles require [scene,bin,time] and [scene,bin]')
    q = np.empty(scores.shape[1:], float)
    for b in range(scores.shape[1]):
        values = scores[present[:, b], b]
        if not len(values):
            raise ValueError(f'insufficient independent calibration scenes in speed bin {b}: n=0')
        for k in range(scores.shape[-1]):
            q[b, k] = upper_quantile(values[:, k], alpha)
        if not np.isfinite(q[b]).all():
            raise ValueError(f'insufficient independent calibration scenes in speed bin {b}: '
                             f'n={len(values)}, alpha={alpha}; finite cap forbidden')
    return q


def interval_quantiles_at_samples(interval_q, times, offsets):
    """Conservative piecewise interval radius; both neighbors at a boundary."""
    values, times, offsets = map(lambda a: np.asarray(a, float), (interval_q, times, offsets))
    if (times.ndim != 1 or len(times) < 2 or values.shape[-1] != len(times)-1
            or offsets.ndim != 1 or not np.isfinite(times).all()
            or not np.isfinite(offsets).all() or np.any(np.diff(times) <= 0)
            or np.any(offsets < times[0]-1e-9) or np.any(offsets > times[-1]+1e-9)
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError('interval quantiles and requested offsets must share a finite horizon')
    left = np.clip(np.searchsorted(times[1:], offsets, side='left'), 0, len(times)-2)
    right = np.clip(np.searchsorted(times[:-1], offsets, side='right')-1, 0, len(times)-2)
    return np.maximum(values[..., left], values[..., right])


class Stage5RiskPolicy:
    """Hook compatible with RemusDWA, with no truth argument or online update."""
    def __init__(self, method='A0', library_path=None, alpha=.1, combination='add',
                 fixed_margin_m=None, cfg=None, predictor='ca'):
        if method not in METHODS or combination not in ('add', 'replace'):
            raise ValueError('invalid stage5 risk method or margin combination')
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError('alpha must lie strictly between zero and one')
        self.method, self.alpha, self.combination = method, float(alpha), combination
        self.description = {
            'A0': 'original fixed engineering prediction envelope',
            'A1': 'independently tuned explicit fixed prediction margin',
            'A2': 'time-indexed observed-speed-group calibrated error balls',
            'A3': 'legacy all-horizon joint calibrated ball form on new independent data',
            'A4': 'legacy frozen candidate-related joint swept-gap ECP form on new data',
            'A5': 'time-indexed speed-group candidate-related swept-gap ECP adaptation',
        }[method]
        self.predictor = canonical_predictor(predictor)
        if cfg is None:
            raise ValueError('stage5 risk requires the actual simulation cfg')
        self.contract = forecaster_contract(self.predictor, cfg)
        self.noise_std_m = float(cfg.sensor_noise_std_m)
        self.fixed_margin_m = None if fixed_margin_m is None else float(fixed_margin_m)
        if method == 'A1' and (self.fixed_margin_m is None
                or not np.isfinite(self.fixed_margin_m) or self.fixed_margin_m < 0):
            raise ValueError('A1 requires an explicit finite nonnegative independently tuned margin')
        self.metadata = {'method': method, 'predictor': self.predictor,
                         'predictor_contract': self.contract, 'alpha': self.alpha,
                         'combination': 'original' if method == 'A0' else combination,
                         'fixed_margin_m': self.fixed_margin_m, 'online_update': False,
                         'safety_guarantee': 'none; empirical pointwise scenario-block calibration',
                         'speed_bin_edges_mps': SPEED_BIN_METADATA,
                         'removed_components': ['fixed_prediction_margin_m', 'stage2_age_margin_m']
                            if combination == 'replace' and method != 'A0' else [],
                         'memory_extra_uncertainty_retained': True,
                         'physical_radii_and_actual_safety_margin_unchanged': True}
        if method in ('A0', 'A1'):
            return
        if library_path is None:
            raise ValueError(f'{method} requires an independent predictor-specific calibration library')
        self._load(library_path, cfg)

    def _load(self, path, cfg):
        with np.load(Path(path), allow_pickle=False) as data:
            self.pred = np.asarray(data['pred_local'], float)
            self.labels = np.asarray(data['label_local'], float)
            self.radii = np.asarray(data['radii'], float)
            self.mask = np.asarray(data['mask'], bool)
            self.speeds = np.asarray(data['observed_speeds_mps'], float)
            self.times = np.asarray(data['times_s'], float)
            ids = [str(v) for v in data['case_ids']]
            scene_ids = [str(v) for v in data['scene_ids']]
            empty = np.asarray(data['empty_scene_mask'], bool)
            provenance = json.loads(str(data['metadata_json'].item()))
        self.pred, self.labels, self.mask, _ = _score_inputs(
            self.pred, self.labels, self.mask, self.speeds, SPEED_BIN_EDGES)
        active = np.any(self.mask, axis=(1, 2))
        if (len(ids) != len(self.pred) or len(set(ids)) != len(ids)
                or len(scene_ids) != len(ids) or len(set(scene_ids)) != len(scene_ids)
                or empty.shape != active.shape or np.any(empty == active)):
            raise ValueError('distinct scene IDs and explicit complete-empty scene flags are required')
        if (self.radii.shape != self.mask.shape or not np.isfinite(self.radii[self.mask]).all()
                or np.any(self.radii[self.mask] <= 0)):
            raise ValueError('calibration requires actual positive observed obstacle radii')
        if (self.times.shape != (self.pred.shape[-2],) or not np.isfinite(self.times).all()
                or self.times[0] != 0 or np.any(np.diff(self.times) <= 0)):
            raise ValueError('calibration requires aligned increasing horizon times from zero')
        if abs(self.times[-1]-cfg.horizon_s) > 1e-9:
            raise ValueError('calibration and planner horizons differ')
        required = {'schema': 'stage5-risk-v1', 'split': 'calibration',
                    'predictor': self.predictor, 'predictor_contract': self.contract,
                    'noise_std_m': self.noise_std_m, 'case_ids': ids,
                    'speed_bin_edges_mps': SPEED_BIN_METADATA}
        for key, value in required.items():
            if provenance.get(key) != value:
                raise ValueError(f'calibration parameter/provenance mismatch: {key}')
        sources = provenance.get('scenes', [])
        if len(sources) != len(ids):
            raise ValueError('calibration must retain one source role per independent scene')
        for i, source in enumerate(sources):
            if (source.get('split') != 'calibration' or source.get('case_id') != ids[i]
                    or source.get('scene_id') != scene_ids[i]
                    or source.get('noise_std_m') != self.noise_std_m):
                raise ValueError('calibration source role/scene/noise mismatch')
        self.radii = np.where(self.mask, self.radii, 0.)
        self.active = active
        self.ball_scores = sphere_scene_scores(self.pred, self.labels, self.mask)[active]
        if not len(self.ball_scores):
            raise ValueError('calibration has no genuinely observed obstacle scenes')
        self.ball_q = upper_quantile(self.ball_scores, self.alpha)
        if not np.isfinite(self.ball_q):
            raise ValueError('insufficient independent active calibration scenes; finite cap forbidden')
        self.time_scores, self.present = time_ball_scores(self.pred, self.labels,
                                                         self.mask, self.speeds)
        # Only stratified methods need sufficient samples in *each* group.
        # Joint methods legitimately use all active scenes as their one group.
        self.time_ball_q = (time_quantiles(self.time_scores, self.present, self.alpha)
                            if self.method in ('A2', 'A5') else None)
        self.metadata.update(calibration_case_ids=ids, library_parameters=provenance,
                             calibration_scene_ids=scene_ids,
                             active_calibration_scene_count=int(active.sum()),
                             calibration_scene_count=len(ids),
                             calibration_scene_counts_by_speed_bin=self.present.sum(axis=0).tolist(),
                             empty_scenes_excluded_from_quantiles=True,
                             calibration_horizon_s=float(self.times[-1]))

    def adjust(self, states, terrain, tracks, now, cfg, forecast_context, base_clear):
        if self.method == 'A0':
            return float(base_clear), {'method': 'A0', 'quantile_m': 0.,
                                      'base_clearance_m': float(base_clear)}
        states = np.asarray(states, float)
        if (states.ndim != 2 or states.shape[1] != 16 or len(states) < 2
                or not np.isfinite(states).all() or abs(states[0, 0]) > 1e-9
                or np.any(np.diff(states[:, 0]) <= 0)):
            raise ValueError('risk input must be finite ordered relative full REMUS states')
        offsets = states[:, 0]
        if abs(offsets[-1]-cfg.horizon_s) > 1e-9:
            raise ValueError('risk and physical candidate horizons differ')
        if (forecaster_contract(self.predictor, cfg) != self.contract
                or cfg.sensor_noise_std_m != self.noise_std_m
                or canonical_predictor(forecast_context.predictor.name) != self.predictor
                or forecaster_contract(self.predictor, forecast_context.predictor.cfg) != self.contract):
            raise ValueError('actual online predictor/config differs from calibration contract')
        record = {'method': self.method, 'alpha': self.alpha, 'combination': self.combination,
                  'predictor': self.predictor, 'base_clearance_m': float(base_clear),
                  'speed_bin_edges_mps': SPEED_BIN_METADATA, 'tracks': [],
                  'memory_extra_uncertainty_retained': True}
        if not tracks:
            return float(base_clear), dict(record, quantile_m=0., observed_set_empty=True)
        q, score_kind = None, 'fixed'
        if self.method == 'A1':
            q = np.full(len(offsets), self.fixed_margin_m)
            record['fixed_margin_m'] = self.fixed_margin_m
        elif self.method == 'A3':
            q = np.full(len(offsets), self.ball_q)
            record['scene_scores_m'] = self.ball_scores.tolist()
            score_kind = 'joint_scene_ball'
        else:
            local = local_positions(states[:, 1:4], states[0, 1:4], states[0, 6])
            sampled = np.column_stack([np.interp(self.times, offsets, local[:, i]) for i in range(3)])
            if self.method == 'A4':
                scores = ecp_scene_scores(sampled, self.pred, self.labels, self.radii,
                                          self.mask, cfg.body_radius_m)[self.active]
                q = np.full(len(offsets), upper_quantile(scores, self.alpha))
                record['scene_scores_m'] = scores.tolist()
                score_kind = 'joint_scene_candidate_swept_gap'
            else:
                point_q = self.time_ball_q
                interval_q = np.maximum(point_q[:, :-1], point_q[:, 1:])
                score_kind = 'point_time_speed_ball'
                if self.method == 'A5':
                    scores, present = time_ecp_scores(sampled, self.pred, self.labels,
                        self.radii, self.mask, self.speeds, body_radius=cfg.body_radius_m)
                    interval_q = time_quantiles(scores, present, self.alpha)
                    point_q = None
                    score_kind = 'interval_time_speed_candidate_swept_gap'
                q = interval_quantiles_at_samples(interval_q, self.times, offsets)
                record.update(offsets_s=self.times.tolist(),
                    q_by_speed_bin_m=(interval_q if point_q is None else point_q).tolist(),
                    q_interval_by_speed_bin_m=interval_q.tolist(),
                    calibration_scene_counts_by_speed_bin=self.present.sum(axis=0).tolist())
        if not np.isfinite(q).all():
            raise ValueError('nonfinite calibrated radius: finite cap forbidden')
        forecasts, applied = [], []
        logging_offsets = self.times if self.method not in ('A0', 'A1') else np.array([0., cfg.horizon_s])
        for track in tracks:
            forecast = forecast_context.forecast(track, now+offsets, now)
            b = int(speed_bin_indices(observation_speed(track)))
            selected_q = q[b] if q.ndim == 2 else q
            radius = forecast.enclosing_radii_m.copy()
            removed = {}
            if self.combination == 'replace':
                components = forecast.metadata.get('radius_components', {})
                for key in ('fixed_prediction_margin_m', 'stage2_age_margin_m'):
                    if key not in components or not np.isfinite(components[key]) or components[key] < 0:
                        raise ValueError('replacement requires explicit finite engineering radius components')
                    removed[key] = float(components[key])
                radius -= sum(removed.values())
            # Never remove physical radii or the independent memory bound.
            lower = track.radius_m+cfg.body_radius_m
            memory = float(forecast.metadata.get('radius_components', {}).get('memory_extra_uncertainty_m', 0.))
            if np.any(radius < lower+memory-1e-9):
                raise ValueError('risk replacement would shrink a physical or retained memory radius')
            forecasts.append((forecast.positions, radius+selected_q))
            applied.append(float(np.max(selected_q)))
            logged_q = (interval_quantiles_at_samples(interval_q[b], self.times, logging_offsets)
                        if q.ndim == 2 else np.full(len(logging_offsets), selected_q[0]))
            record['tracks'].append({'object_id': int(track.object_id),
                                     'observed_speed_mps': observation_speed(track),
                                     'speed_bin': b, 'removed_components': removed,
                                     'applied_q_m': logged_q.tolist()})
        from .remus_planner import _clearance_positions
        clear = float(_clearance_positions(states[:, 1:4], terrain, forecasts, cfg))
        record.update(quantile_m=max(applied), score_kind=score_kind,
                      risk_clearance_m=clear, applied_q_offsets_s=logging_offsets.tolist(),
                      physical_integration_step_s=float(np.max(np.diff(offsets))))
        return clear, record


def make_policy(method='A0', library_path=None, alpha=.1, combination='add',
                fixed_margin_m=None, cfg=None, predictor='ca'):
    return Stage5RiskPolicy(method, library_path, alpha, combination,
                            fixed_margin_m, cfg, predictor)
