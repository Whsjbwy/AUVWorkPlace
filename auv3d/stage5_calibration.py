"""Offline calibration on new complete scenes and a fixed causal sensor carrier.

The carrier is a data-acquisition protocol, NOT a REMUS execution trajectory.
It deliberately avoids method-success-dependent sampling but differs from the
closed-loop ego/FOV distribution. That covariate shift prevents safety claims.
Only this offline module can generate future labels. Online risk loads arrays.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .sensors import IdealObjectSonar, TrackMemory, validate_track
from .stage3_forecasting import observation_speed
from .stage4_calibration import yaw_local
from .stage4_risk import upper_quantile
from .stage5_protocol import (
    PHASES, PREDICTORS, SPEED_BIN_EDGES, SPEED_BIN_NAMES,
    config_for_case, make_scene,
)


SCHEMA = 'stage5-risk-v1'
SOURCE_KIND = 'registered_causal_sensor_carrier_v1'
ANCHOR_TIMES_S = np.array([20., 40., 60.])
QUERY_TIMES_S = np.arange(121, dtype=float) * .2
PHYSICS_DT_S = .02
CARRIER_SPEED_MPS = 1.0
OBSTACLE_FAMILIES = ('static3d', 'crossing3d', 'maneuvering3d', 'multi_crossing3d')


def speed_bin(speed_mps):
    value = float(speed_mps)
    if not np.isfinite(value) or value < 0:
        raise ValueError('observed speed must be finite and nonnegative')
    return 0 if value < SPEED_BIN_EDGES[1] else 1


def _selected_rows(rows):
    selected = [deepcopy(row) for row in rows if row.get('split') == 'calibration']
    if not selected:
        raise ValueError('at least one new calibration-role complete scene is required')
    for row in selected:
        config_for_case(row)
    for key in ('case_id', 'scene_id'):
        ids = [row[key] for row in selected]
        if len(set(ids)) != len(ids):
            raise ValueError('noise replicas or duplicate complete scenes cannot enlarge calibration n')
    if len({row['phase'] for row in selected}) != 1 or len({row['noise_std_m'] for row in selected}) != 1:
        raise ValueError('each library requires one phase and one noise condition')
    return selected


def risk_calibration_rows(rows):
    """Deterministically exclude ALL known empty terrain controls, never retries.

    Registration still contains them, including recorded factory failures.
    Their absence changes no obstacle score/count because empty scenes never
    contribute a zero to the risk quantiles. This rule uses family, not outcome.
    """
    selected = []
    for row in rows:
        if row.get('split') != 'calibration':
            continue
        config_for_case(row)
        if row['family'] in OBSTACLE_FAMILIES:
            selected.append(deepcopy(row))
    if not selected:
        raise ValueError('no registered obstacle calibration scenes')
    return selected


def _carrier_pose(scene, time_s):
    start, goal = np.asarray(scene.start_eta[:3]), np.asarray(scene.goal)
    delta = goal - start
    length = float(np.linalg.norm(delta))
    if length <= 0:
        raise ValueError('fixed observation carrier requires distinct endpoints')
    position = start + min(CARRIER_SPEED_MPS * time_s / length, 1.) * delta
    yaw = float(np.arctan2(delta[1], delta[0]))
    return np.r_[position, 0., 0., yaw]


def generate_scene_samples(case, *, purpose='calibration'):
    """Acquire histories causally, then attach offline truth on the same clock.

    Same sonar/FOV/delay/noise/hard-expiry as the registered configuration.
    Truth can advance after an anchor, but only immutable history copies feed
    a predictor. Acquisition extends to 84 s even if a hypothetical navigation
    would have collided: this is a fixed carrier, not a successful mission run.
    """
    cfg = config_for_case(case)
    if purpose not in ('calibration', 'evaluation'):
        raise ValueError('source purpose must be calibration or evaluation')
    if purpose == 'calibration' and case['split'] != 'calibration':
        raise ValueError('future-label acquisition accepts calibration split only')
    scene = make_scene(case, cfg)
    obstacles = scene.obstacles
    initial = obstacles.snapshot()
    empty = len(initial.ids) == 0
    sonar = IdealObjectSonar(cfg.sensor_config(), cfg.seed)
    memory = TrackMemory(cfg.track_max_age_s)
    pending, packets, anchors, snapshots = [], [], [], []
    period_ticks = round(cfg.observation_period_s / PHYSICS_DT_S)
    delay_ticks = round(cfg.observation_delay_s / PHYSICS_DT_S)
    anchor_ticks = {round(t / PHYSICS_DT_S): float(t) for t in ANCHOR_TIMES_S}
    last_tick = round((ANCHOR_TIMES_S[-1] + QUERY_TIMES_S[-1]) / PHYSICS_DT_S)
    # Empty scenes have no labels, nevertheless record every scheduled anchor.
    for tick in range(last_tick + 1):
        now = round(tick * PHYSICS_DT_S, 10)
        if tick % 10 == 0:
            snap = obstacles.snapshot()
            snapshots.append((snap.ids.copy(), snap.positions.copy(), snap.radii.copy()))
        if tick <= max(anchor_ticks) and tick % period_ticks == 0:
            observations = sonar.observe(_carrier_pose(scene, now), now, obstacles.snapshot(), scene.terrain)
            pending.append((tick + delay_ticks, observations))
        delivered = []
        remaining = []
        for delivery_tick, observations in pending:
            if delivery_tick <= tick:
                delivered.extend(observations)
                for obs in observations:
                    packets.append(dict(object_id=obs.object_id, measurement_time_s=obs.time_s,
                        delivered_time_s=now, position=obs.position.tolist(), radius_m=obs.radius_m))
            else:
                remaining.append((delivery_tick, observations))
        pending = remaining
        memory.update(delivered, now)
        if tick in anchor_ticks:
            eta = _carrier_pose(scene, now)
            anchors.append(dict(anchor_s=now, ego_ned_m=eta[:3].copy(), ego_yaw_rad=float(eta[5]),
                                tracks=deepcopy(memory.current(now))))
        if tick < last_tick and not empty:
            obstacles.step(PHYSICS_DT_S)
    windows, unavailable = [], []
    for anchor in anchors:
        now, tracks = anchor['anchor_s'], anchor['tracks']
        if not tracks:
            unavailable.append(dict(anchor_s=now, reason='no_delivered_observed_targets'))
            continue
        start = round(now / .2)
        truth = snapshots[start:start + len(QUERY_TIMES_S)]
        labels = []
        for track in tracks:
            validate_track(track, now)
            matched_packets = []
            for time_s, position in zip(track.times, track.positions):
                matches = [p for p in packets if p['object_id'] == track.object_id
                    and p['measurement_time_s'] == time_s and p['delivered_time_s'] <= now + 1e-9
                    and np.array_equal(p['position'], position)]
                if len(matches) != 1:
                    raise ValueError('every forecast history sample must have exactly one causal delivery')
                matched_packets.append(matches[0])
            positions = []
            for ids, points, radii in truth:
                indexes = np.flatnonzero(ids == track.object_id)
                if len(indexes) != 1:
                    raise ValueError('offline truth cannot uniquely label a delivered object')
                index = indexes[0]
                if abs(radii[index] - track.radius_m) > 1e-9:
                    raise ValueError('offline target radius differs from sensed geometry')
                positions.append(points[index])
            if len(positions) != len(QUERY_TIMES_S):
                raise ValueError('offline labels must cover the entire registered 24 s horizon')
            labels.append(np.asarray(positions))
        windows.append(dict(anchor, label_world=np.asarray(labels)))
    if not empty and not windows:
        raise ValueError('nonempty scene has no causal calibration labels; cannot assign a zero score')
    metadata = dict(source_kind=SOURCE_KIND, source_purpose=purpose,
        case_id=case['case_id'], scene_id=case['scene_id'],
        phase=case['phase'], split=case['split'], noise_std_m=case['noise_std_m'],
        scenario=case['scenario'], seed=case['seed'], configuration=cfg.to_dict(),
        layout_shift_m=case['layout_shift_m'], empty_obstacle_scene=empty,
        obstacle_count=len(initial.ids),
        carrier_speed_mps=CARRIER_SPEED_MPS, carrier_pose_rule='straight start-goal line; roll/pitch0; yaw along line',
        acquisition_end_time_s=84., completed_acquisition=True,
        selected_anchor_s=[w['anchor_s'] for w in windows], unavailable_anchors=unavailable,
        delivered_observation_count=len(packets), observations=packets,
        source_is_REMUS_execution=False, future_truth_used_for_forecast=False)
    return dict(windows=windows, metadata=metadata)


def generate_evaluation_samples(case):
    """Offline held-out diagnostics; preserve the original dev/tune/test role."""
    return generate_scene_samples(case, purpose='evaluation')


def _validate_sample(sample, case, *, purpose='calibration'):
    meta = sample.get('metadata', {})
    for key in ('case_id', 'scene_id', 'phase', 'split', 'noise_std_m', 'scenario', 'seed', 'layout_shift_m'):
        if meta.get(key) != case.get(key):
            raise ValueError(f'calibration source {key} differs from registration')
    if (meta.get('source_kind') != SOURCE_KIND or meta.get('source_purpose') != purpose
            or (purpose == 'calibration' and meta.get('split') != 'calibration')
            or meta.get('completed_acquisition') is not True
            or meta.get('future_truth_used_for_forecast') is not False
            or meta.get('source_is_REMUS_execution') is not False
            or meta.get('configuration') != config_for_case(case).to_dict()):
        raise ValueError('calibration requires completed registered causal acquisition, never test/navigation truth input')
    windows = sample.get('windows')
    if not isinstance(windows, list):
        raise ValueError('calibration source windows must be a list')
    seen = set()
    for window in windows:
        anchor = window.get('anchor_s')
        if anchor not in ANCHOR_TIMES_S or anchor in seen:
            raise ValueError('only unique preregistered calibration anchors are permitted')
        seen.add(anchor)
        tracks = window['tracks']
        if not tracks or len({t.object_id for t in tracks}) != len(tracks):
            raise ValueError('nonempty calibration window requires unique object tracks')
        labels = np.asarray(window['label_world'], float)
        if labels.shape != (len(tracks), len(QUERY_TIMES_S), 3) or not np.isfinite(labels).all():
            raise ValueError('calibration labels must be finite aligned complete trajectories')
        for track in tracks:
            validate_track(track, anchor)
            for time_s, position in zip(track.times, track.positions):
                matches = [p for p in meta.get('observations', []) if p['object_id'] == track.object_id
                    and p['measurement_time_s'] == time_s and p['measurement_time_s'] <= p['delivered_time_s'] <= anchor
                    and np.array_equal(p['position'], position)]
                if len(matches) != 1:
                    raise ValueError('source history contains an undelivered or future observation')
    empty = meta.get('empty_obstacle_scene')
    expected_count = {'static3d': 1, 'crossing3d': 1, 'maneuvering3d': 3,
        'multi_crossing3d': 3, 'terrain3d': 0, 'perlin_cruise_constrained3d': 0}[case['family']]
    if (not isinstance(empty, bool) or meta.get('obstacle_count') != expected_count
            or empty != (expected_count == 0) or (empty and windows) or (not empty and not windows)):
        raise ValueError('true empty scenes must be explicit; missing labels cannot be zero scenes')


def _factory(name, cfg):
    if name not in PREDICTORS:
        raise ValueError(f'predictor must be one of {PREDICTORS}')
    from .stage5_forecasting import make_forecaster
    return make_forecaster(name, cfg)


def build_library(manifest_rows, out_npz, predictor_name='ca', *, samples=None, forecaster_factory=None):
    """One independent phase/predictor/noise artifact; no existing file overwrite.

    ``samples`` optionally reuses immutable acquired histories across predictors.
    Every source's role/ID/noise/config and delivered history is checked again.
    forecaster_factory(name,cfg) is injected for tests or causal predictor reuse.
    """
    rows = _selected_rows(manifest_rows)
    if predictor_name not in PREDICTORS:
        raise ValueError('each registered predictor needs its own independent library')
    out = Path(out_npz)
    if out.exists():
        raise FileExistsError(f'preserve existing calibration library: {out}')
    if samples is None:
        samples = [generate_scene_samples(row) for row in rows]
    if len(samples) != len(rows):
        raise ValueError('calibration source count differs from complete-scene registration')
    for sample, row in zip(samples, rows, strict=True):
        _validate_sample(sample, row)
    factory = forecaster_factory or _factory
    object_count = max(1, max((len(w['tracks']) for sample in samples for w in sample['windows']), default=0))
    shape = (len(rows), len(ANCHOR_TIMES_S), object_count, len(QUERY_TIMES_S), 3)
    pred, label = np.zeros(shape), np.zeros(shape)
    radii, speeds = np.zeros(shape[:3]), np.zeros(shape[:3])
    ids = np.full(shape[:3], -1, dtype=np.int64)
    mask = np.zeros(shape[:3], bool)
    for si, (sample, row) in enumerate(zip(samples, rows, strict=True)):
        cfg = config_for_case(row)
        forecaster = factory(predictor_name, cfg)
        if forecaster.name != predictor_name:
            raise ValueError('forecast implementation identity differs from requested library predictor')
        for w in sample['windows']:
            wi = int(np.flatnonzero(ANCHOR_TIMES_S == w['anchor_s'])[0])
            origin, yaw = w['ego_ned_m'], w['ego_yaw_rad']
            for oi, track in enumerate(w['tracks']):
                # Predictor sees only the history and requested absolute times.
                forecast = forecaster.forecast(deepcopy(track), w['anchor_s'] + QUERY_TIMES_S, w['anchor_s'])
                forecast.validate(w['anchor_s'])
                if forecast.object_id != track.object_id or not np.array_equal(forecast.absolute_times, w['anchor_s'] + QUERY_TIMES_S):
                    raise ValueError('calibration forecast changed object identity or requested clock')
                pred[si, wi, oi] = yaw_local(forecast.positions, origin, yaw)
                label[si, wi, oi] = yaw_local(w['label_world'][oi], origin, yaw)
                radii[si, wi, oi], speeds[si, wi, oi] = track.radius_m, observation_speed(track)
                ids[si, wi, oi], mask[si, wi, oi] = track.object_id, True
    active = np.any(mask, axis=(1, 2))
    empty = np.asarray([s['metadata']['empty_obstacle_scene'] for s in samples], bool)
    if np.any(active == empty):
        raise ValueError('calibration active/empty flags disagree; missing data cannot be zero')
    eligible = np.stack([np.any(mask & (speeds < .25 if bi == 0 else speeds >= .25), axis=(1, 2))
                         for bi in range(2)], axis=1)
    from .stage5_forecasting import forecaster_contract
    metadata = dict(schema=SCHEMA, role='calibration_only', split='calibration',
        phase=rows[0]['phase'], predictor=predictor_name, noise_std_m=rows[0]['noise_std_m'],
        predictor_contract=forecaster_contract(predictor_name, config_for_case(rows[0])),
        case_ids=[row['case_id'] for row in rows],
        source_kind=SOURCE_KIND, statistical_unit='complete scene_id',
        registered_scene_count=len(rows), active_scene_count=int(active.sum()),
        empty_scene_count=int(empty.sum()), observed_speed_bin_scene_counts=eligible.sum(axis=0).tolist(),
        speed_bin_edges_mps=[0., .25, 'inf'], speed_bin_names=list(SPEED_BIN_NAMES),
        speed_definition='last two already delivered measurements divided by acquisition-time difference',
        horizon_s=24., query_dt_s=.2, physics_dt_s=.02,
        fixed_anchor_times_s=ANCHOR_TIMES_S.tolist(), coordinate_frame='NED anchor-yaw, positive-down depth',
        empty_scene_quantile_rule='recorded but excluded, not a zero score',
        insufficient_bin_rule='infinite finite-sample order statistic, never capped or silently pooled',
        source_is_REMUS_execution=False, future_truth_used_online=False,
        online_ACP_implemented=False, coverage_claim='none; fixed-carrier covariate shift and correlated scene families',
        scenes=[s['metadata'] for s in samples])
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        np.savez_compressed(stream, pred_local=pred, label_local=label, radii=radii,
            mask=mask, object_ids=ids, observed_speeds_mps=speeds,
            speed_bin_index=np.where(mask, (speeds >= .25).astype(int), -1),
            eligible_scene_mask_per_speed=eligible, active_scene_mask=active,
            empty_scene_mask=empty, window_mask=np.any(mask, axis=2),
            times_s=QUERY_TIMES_S, anchor_times_s=ANCHOR_TIMES_S,
            case_ids=np.asarray([r['case_id'] for r in rows]),
            scene_ids=np.asarray([r['scene_id'] for r in rows]),
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False)))
    return metadata


build_causal_library = build_library


def build_libraries(manifest_rows, out_dir, *, predictors=PREDICTORS, forecaster_factory=None):
    """Reuse each causal acquisition across predictors; noise pools stay separate."""
    rows = [row for row in manifest_rows if row.get('split') == 'calibration']
    phases = {row.get('phase') for row in rows}
    if len(phases) != 1 or not phases <= set(PHASES):
        raise ValueError('one preregistered phase is required per calibration suite')
    requested = tuple(predictors)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(PREDICTORS):
        raise ValueError('calibration suite predictors must be unique registered names')
    results = []
    for noise in sorted({row['noise_std_m'] for row in rows}):
        group = _selected_rows([row for row in rows if row['noise_std_m'] == noise])
        acquired = [generate_scene_samples(row) for row in group]
        for predictor in requested:
            out = Path(out_dir) / f"{group[0]['phase']}_{predictor}_n{round(noise * 1000):03d}.npz"
            meta = build_library(group, out, predictor, samples=acquired,
                                 forecaster_factory=forecaster_factory)
            results.append(dict(path=str(out), predictor=predictor, noise_std_m=noise, metadata=meta))
    return results


def stratified_ball_quantiles(pred, labels, mask, speeds, alpha):
    """Scene maxima per observed-speed group and time; absent labels are unknown.

    Return [speed,time] q, valid scene counts and [scene,speed,time] scores.
    No Bonferroni/joint-time guarantee is implied by marginal per-time q.
    """
    pred, labels = np.asarray(pred, float), np.asarray(labels, float)
    mask, speeds = np.asarray(mask, bool), np.asarray(speeds, float)
    if (pred.ndim != 5 or pred.shape[-1] != 3 or labels.shape != pred.shape
            or mask.shape != pred.shape[:3] or speeds.shape != mask.shape
            or not np.isfinite(pred[mask]).all() or not np.isfinite(labels[mask]).all()
            or not np.isfinite(speeds[mask]).all() or np.any(speeds[mask] < 0)):
        raise ValueError('ball calibration requires aligned finite valid trajectories and observed speeds')
    # Validate alpha even when every group has zero usable samples.
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between zero and one')
    errors = np.linalg.norm(np.where(mask[..., None, None], pred - labels, 0.), axis=-1)
    scores = np.full((len(pred), 2, pred.shape[-2]), np.nan)
    quantiles = np.full((2, pred.shape[-2]), np.inf)
    counts = np.zeros(2, int)
    for bi in range(2):
        selected = mask & (speeds < .25 if bi == 0 else speeds >= .25)
        eligible = np.any(selected, axis=(1, 2))
        counts[bi] = int(eligible.sum())
        values = np.max(np.where(selected[..., None], errors, 0.), axis=(1, 2))
        scores[eligible, bi] = values[eligible]
        for ti in range(pred.shape[-2]):
            if counts[bi]:
                quantiles[bi, ti] = upper_quantile(scores[eligible, bi, ti], alpha)
    return dict(quantiles_m=quantiles, scene_counts=counts, scene_scores_m=scores,
                speed_bin_edges_mps=[0., .25, None], marginal_time_only=True)
