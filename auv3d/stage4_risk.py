"""Frozen scene-block risk pilot, inspired by Shin/Lee/Yang's ECP (CDC 2025).

The author algorithm re-evaluates mature prediction/label pairs at candidate
positions. Here three-dimensional swept gaps and ego/yaw aligned complete
scenario blocks replace their 2D point geometry. This is an offline engineering
adaptation, not the author's adaptive algorithm or a closed-loop safety theorem.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import swept_sphere_clearance


def upper_quantile(scores, alpha):
    """Finite-sample upper order statistic; insufficient data gives infinity."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('scores must be a nonempty finite nonnegative vector')
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    rank = int(np.ceil((len(values) + 1) * (1 - alpha)))
    return float(np.sort(values)[rank - 1]) if rank <= len(values) else float('inf')


def local_positions(points, origin, yaw):
    """Translate NED points and rotate horizontal axes; depth stays positive down."""
    delta = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    return delta @ np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def sphere_scene_scores(pred, labels, mask):
    errors = np.linalg.norm(pred - labels, axis=-1)
    return np.max(np.where(mask[..., None], errors, 0.), axis=(1, 2, 3))


def ecp_scene_scores(path, pred, labels, radii, mask, body_radius):
    """[min_o g_pred - min_o g_label]_+, NOT max_o(g_pred_o-g_label_o).

    Take the maximum over intervals and recorded anchors *within* each complete
    calibration scenario. Missing object slots cannot become zero-error labels.
    Empty observed sets have no obstacle constraint and contribute zero.
    """
    path = np.asarray(path, dtype=float)
    if path.shape != pred.shape[-2:] or not np.isfinite(path).all():
        raise ValueError('candidate and calibration must share [time,3] grid')
    total_r = np.asarray(radii)[..., None] + body_radius
    gaps = []
    for targets in (pred, labels):
        gap = swept_sphere_clearance(path[:-1], path[1:], targets[..., :-1, :],
                                     targets[..., 1:, :], total_r)
        gap = np.where(mask[..., None], gap, np.inf)
        gaps.append(np.min(gap, axis=2))
    present = np.any(mask, axis=2)
    difference = np.zeros_like(gaps[0])
    np.subtract(gaps[0], gaps[1], out=difference, where=present[..., None])
    return np.max(np.maximum(difference, 0.), axis=(1, 2))


class RiskPolicy:
    """B0 unchanged margin, B1 independent ball, B2 candidate-dependent ECP.

    Both B1/B2 ADD their calibrated term to exactly the same original CA
    engineering envelope. They never reduce terrain/domain/safety constraints.
    Online calls accept no obstacle truth or scene object; the library is frozen.
    """
    def __init__(self, method='B0', library_path=None, alpha=.1):
        if method not in ('B0', 'B1', 'B2'):
            raise ValueError('risk method must be B0, B1 or B2')
        self.method, self.alpha = method, float(alpha)
        if not np.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError('alpha must lie strictly between zero and one')
        self.metadata = {'method': method, 'alpha': self.alpha,
                         'online_update': False, 'aggregation': 'maximum within complete scenario',
                         'original_engineering_envelope_retained': True,
                         'safety_guarantee': 'none; empirical frozen offline pilot',
                         'calibration_frame': 'ego origin and yaw; NED depth positive down'}
        self.description = {'B0': 'original engineering margin; no calibrated risk',
                            'B1': 'independent calibrated error ball (frozen scene blocks)',
                            'B2': 'candidate-dependent ECP swept-gap pilot (frozen scene blocks)'}[method]
        if method == 'B0':
            return
        if library_path is None:
            raise ValueError('B1/B2 require an independent calibration library')
        with np.load(Path(library_path), allow_pickle=False) as data:
            self.pred = np.asarray(data['pred_local'], dtype=float)
            self.labels = np.asarray(data['label_local'], dtype=float)
            self.radii = np.asarray(data['radii'], dtype=float)
            self.mask = np.asarray(data['mask'], dtype=bool)
            self.times = np.asarray(data['times_s'], dtype=float)
            ids = [str(v) for v in data['case_ids']]
            if 'empty_scene_mask' not in data:
                raise ValueError('library must explicitly distinguish empty scenes from missing labels')
            empty_scenes = np.asarray(data['empty_scene_mask'], dtype=bool)
            provenance = json.loads(str(data['metadata_json'].item()))
        if self.pred.ndim != 5 or self.pred.shape[-1] != 3 or self.labels.shape != self.pred.shape:
            raise ValueError('calibration arrays require [scenario,anchor,object,time,3]')
        if self.mask.shape != self.pred.shape[:3] or self.radii.shape != self.mask.shape:
            raise ValueError('calibration masks/radii must match trajectory slots')
        if len(ids) != len(self.pred) or len(ids) != len(set(ids)):
            raise ValueError('calibration must contain distinct complete scenario IDs')
        if empty_scenes.shape != (len(ids),):
            raise ValueError('empty scene flags must match complete calibration scenarios')
        observed_scenes = np.any(self.mask, axis=(1, 2))
        if np.any(observed_scenes == empty_scenes):
            raise ValueError('missing labels cannot become an empty-scene zero score')
        if self.times.shape != (self.pred.shape[-2],) or len(self.times) < 2 or self.times[0] != 0 or np.any(np.diff(self.times) <= 0) or not np.isfinite(self.times).all():
            raise ValueError('calibration requires finite aligned increasing horizon offsets from zero')
        if not np.isfinite(self.pred[self.mask]).all() or not np.isfinite(self.labels[self.mask]).all() or not np.isfinite(self.radii[self.mask]).all() or np.any(self.radii[self.mask] <= 0):
            raise ValueError('observed calibration slots must contain finite trajectories and positive radii')
        if provenance.get('split') != 'calibration':
            raise ValueError('risk library must declare calibration split')
        # Inactive padding cannot introduce NaNs into vector arithmetic.
        self.pred = np.where(self.mask[..., None, None], self.pred, 0.)
        self.labels = np.where(self.mask[..., None, None], self.labels, 0.)
        self.radii = np.where(self.mask, self.radii, 0.)
        self.ball_scores = sphere_scene_scores(self.pred, self.labels, self.mask)
        self.ball_q = upper_quantile(self.ball_scores, self.alpha)
        if not np.isfinite(self.ball_q):
            raise ValueError('insufficient independent calibration scenarios for requested alpha; no finite cap allowed')
        self.metadata.update(calibration_case_ids=ids, calibration_scene_count=len(ids),
                             active_calibration_scene_count=int(np.sum(observed_scenes)),
                             empty_calibration_scene_count=int(np.sum(empty_scenes)),
                             calibration_horizon_s=float(self.times[-1]),
                             ball_quantile_m=self.ball_q, library_parameters=provenance)

    def adjust(self, states, terrain, tracks, now, cfg, forecast_context, base_clear):
        if self.method == 'B0':
            return float(base_clear), {'method': 'B0', 'quantile_m': 0., 'base_clearance_m': float(base_clear)}
        if cfg.obstacle_predictor != 'ca':
            raise ValueError('stage4 pilot is registered for the shared CA predictor only')
        states = np.asarray(states, dtype=float)
        if states.ndim != 2 or states.shape[1] != 16 or len(states) < 2 or not np.isfinite(states).all() or np.any(np.diff(states[:, 0]) <= 0):
            raise ValueError('risk input must be finite ordered full REMUS prediction states')
        # ReferencePrediction stores relative offsets; obstacle forecasts use
        # now+offsets. Subtracting now here would break every nonzero anchor.
        offsets = states[:, 0]
        if abs(offsets[0]) > 1e-7 or abs(offsets[-1] - self.times[-1]) > 1e-7:
            raise ValueError('risk and physical candidate horizon/time anchor mismatch')
        if not tracks:
            return float(base_clear), {'method': self.method, 'quantile_m': 0.,
                                      'base_clearance_m': float(base_clear), 'observed_set_empty': True}
        path = local_positions(states[:, 1:4], states[0, 1:4], states[0, 6])
        sampled = np.column_stack([np.interp(self.times, offsets, path[:, i]) for i in range(3)])
        scores = self.ball_scores if self.method == 'B1' else ecp_scene_scores(
            sampled, self.pred, self.labels, self.radii, self.mask, cfg.body_radius_m)
        q = upper_quantile(scores, self.alpha)
        # Same physical swept check at EVERY .02s tick; the .2s library only
        # defines the empirical score and is not a replacement collision check.
        from .remus_planner import _forecast_tracks, _clearance_positions
        forecasts = _forecast_tracks(tracks, offsets, now, cfg, forecast_context)
        calibrated = [(positions, np.asarray(radius) + q) for positions, radius in forecasts]
        clear = float(_clearance_positions(states[:, 1:4], terrain, calibrated, cfg))
        return clear, {'method': self.method, 'quantile_m': float(q),
                       'base_clearance_m': float(base_clear), 'risk_clearance_m': clear,
                       'scene_score_min_m': float(np.min(scores)),
                       'scene_score_max_m': float(np.max(scores)), 'alpha': self.alpha}
