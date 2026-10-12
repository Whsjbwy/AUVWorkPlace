"""Offline, scene-separated CA calibration from completed B0 REMUS runs.

Only this offline builder replays obstacle ground truth.  The resulting frozen
library contains matched forecast/label pairs in the anchor ego yaw frame; it
does not expose scenario generators to the online risk or navigation code.
No model fitting, threshold tuning, online adaptive CP or checksum is performed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .scenarios import make_stage2_scenario
from .sensors import Track, validate_track
from .stage2_config import Stage2Config
from .stage3_forecasting import CAObstacleForecaster


ANCHOR_TIMES_S = np.arange(0., 200., 20.)
QUERY_TIMES_S = np.arange(121, dtype=float) * .2
CALIBRATION_SCENE_COUNT = 24
PHYSICS_DT_S = .02
TIME_ATOL = 1e-8
POSITION_ATOL = 1e-7


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _inside(root, path):
    root, path = root.resolve(), path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'calibration result path must remain within run_root: {path}')
    return path


def _case_directory(root, row):
    if row.get('result_dir') is not None:
        path = Path(row['result_dir'])
        return _inside(root, path if path.is_absolute() else root / path)
    case_id = row['case_id']
    # A case identifier is one directory component, never an arbitrary path.
    if Path(case_id).name != case_id or case_id in ('.', '..') or '/' in case_id or '\\' in case_id:
        raise ValueError('case_id must be a single nonempty directory component')
    return _inside(root, root / 'cases' / case_id)


def _calibration_rows(rows):
    selected = [dict(row) for row in rows
                if row.get('split') == 'calibration' and row.get('method') == 'B0']
    if len(selected) != CALIBRATION_SCENE_COUNT:
        raise ValueError(f'expected exactly {CALIBRATION_SCENE_COUNT} B0 calibration scenes; got {len(selected)}')
    case_ids = [row.get('case_id') for row in selected]
    if any(not isinstance(case_id, str) or not case_id.strip() for case_id in case_ids):
        raise ValueError('calibration scenes require nonempty string case_ids')
    if len(set(case_ids)) != len(case_ids):
        raise ValueError('calibration case_ids must be unique, not duplicate scene samples')
    # Methods/alphas of the same scene are not independent calibration scenes.
    identities = [row.get('scenario_id', row['case_id']) for row in selected]
    if len(set(identities)) != len(identities):
        raise ValueError('calibration scenario identities must be unique')
    if any(row.get('program_error', False) for row in selected):
        raise ValueError('program-error runs cannot supply a completed calibration scene')
    return selected


def _states(path):
    rows = np.genfromtxt(path, delimiter=',', names=True, dtype=float, encoding='utf-8')
    names = ('time_s', 'north_m', 'east_m', 'depth_m', 'yaw_rad')
    if rows.dtype.names is None or any(name not in rows.dtype.names for name in names):
        raise ValueError('states.csv requires time_s, NED position and yaw_rad columns')
    data = np.column_stack([np.atleast_1d(rows[name]) for name in names])
    if not len(data) or not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError('states.csv must contain finite, strictly ordered actual states')
    return data


def _snapshot_arrays(snapshot):
    ids = np.asarray(snapshot.ids)
    points, radii = np.asarray(snapshot.positions, float), np.asarray(snapshot.radii, float)
    if (ids.ndim != 1 or points.shape != (len(ids), 3) or radii.shape != (len(ids),)
            or len(set(ids.tolist())) != len(ids) or not np.isfinite(points).all()
            or not np.isfinite(radii).all() or np.any(radii <= 0)):
        raise ValueError('replayed obstacle snapshot has invalid IDs, positions or radii')
    return ids.astype(int), points, radii


def _replay(scene, until, saved_snapshots):
    """Advance the original obstacle object at exactly .02 s; offline use only."""
    total = int(round(until / PHYSICS_DT_S))
    if abs(total * PHYSICS_DT_S - until) > TIME_ATOL:
        raise ValueError('replay end must lie on the original physics clock')
    saved = {}
    for row in saved_snapshots:
        time = float(row['time_s'])
        tick = int(round(time / PHYSICS_DT_S))
        if not np.isfinite(time) or abs(tick * PHYSICS_DT_S - time) > TIME_ATOL or tick < 0:
            raise ValueError('saved obstacle snapshot is off the original physics clock')
        if tick in saved:
            raise ValueError('duplicate saved obstacle snapshot timestamp')
        saved[tick] = row
    history, max_error, checked = [], 0., 0
    for tick in range(total + 1):
        ids, points, radii = _snapshot_arrays(scene.obstacles.snapshot())
        if tick in saved:
            row = saved[tick]
            expected_ids = np.asarray(row['ids'], dtype=int)
            expected = np.asarray(row['positions'], float).reshape(-1, 3)
            expected_radii = np.asarray(row['radii'], float)
            if (not np.array_equal(ids, expected_ids) or expected.shape != points.shape
                    or expected_radii.shape != radii.shape or not np.isfinite(expected).all()
                    or not np.isfinite(expected_radii).all()
                    or not np.allclose(expected_radii, radii, atol=POSITION_ATOL, rtol=0)):
                raise ValueError('saved obstacle IDs/radii disagree with original-scene replay')
            error = float(np.max(np.abs(points - expected))) if len(points) else 0.
            if error > POSITION_ATOL:
                raise ValueError('saved obstacle motion disagrees with original-scene replay')
            max_error = max(max_error, error)
            checked += 1
        # All calibration queries lie on .2 s, hence every tenth physics tick.
        if tick % 10 == 0:
            history.append((ids.copy(), points.copy(), radii.copy()))
        if tick < total:
            scene.obstacles.step(PHYSICS_DT_S)
    if not checked:
        raise ValueError('offline truth replay needs at least one saved snapshot check')
    return history, max_error, checked


def _verify_observed(track, anchor, packets):
    """Prove each recorded track sample was delivered by its planning anchor."""
    validate_track(track, anchor)
    for time, position in zip(track.times, track.positions):
        matched = False
        for packet in packets:
            delivered = packet.get('delivered_time_s')
            if (delivered is None or not np.isfinite(delivered)
                    or delivered > anchor + TIME_ATOL
                    or abs(float(packet['measurement_time_s']) - time) > TIME_ATOL):
                continue
            if float(packet['measurement_time_s']) > delivered + TIME_ATOL:
                raise ValueError('observation delivery precedes its measurement')
            for obj in packet['objects']:
                if (int(obj['object_id']) == track.object_id
                        and np.allclose(np.asarray(obj['position'], float), position,
                                        atol=POSITION_ATOL, rtol=0)):
                    matched = True
                    break
            if matched:
                break
        if not matched:
            raise ValueError('track contains a sample not causally delivered at the planning anchor')


def yaw_local(points, ego_position, yaw):
    """NED world -> anchor yaw frame; preserves depth-positive-down and distance."""
    points, origin = np.asarray(points, float), np.asarray(ego_position, float)
    if points.shape[-1] != 3 or origin.shape != (3,) or not np.isfinite(yaw):
        raise ValueError('yaw frame requires three-dimensional positions and finite yaw')
    if not np.isfinite(points).all() or not np.isfinite(origin).all():
        raise ValueError('yaw frame positions must be finite')
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return (points - origin) @ rotation


def _load_scene(root, row):
    directory = _case_directory(root, row)
    if (directory / 'program_error.json').exists():
        raise ValueError('program-error artifacts cannot supply B0 calibration data')
    experiment_path = directory / 'risk_experiment.json'
    if experiment_path.exists():
        experiment = _read_json(experiment_path)
        if not isinstance(experiment, dict) or experiment.get('method') != 'B0':
            raise ValueError('actual risk experiment metadata must identify B0 calibration data')
        identities = [experiment]
        # Current runner files have flat case_id/split fields. Also inspect
        # explicit case/job envelopes if supplied by an archived run format.
        for envelope in ('case', 'job'):
            value = experiment.get(envelope)
            if isinstance(value, dict):
                identities.append(value)
                if isinstance(value.get('case'), dict):
                    identities.append(value['case'])
        for identity in identities:
            if 'method' in identity and identity['method'] != 'B0':
                raise ValueError('actual risk experiment metadata must identify B0 calibration data')
            if 'split' in identity and identity['split'] != 'calibration':
                raise ValueError('actual risk experiment metadata must declare calibration split')
            if 'case_id' in identity and identity['case_id'] != row['case_id']:
                raise ValueError('actual risk experiment case_id disagrees with the calibration registration')
    cfg = Stage2Config.load(directory / 'config.json')
    if (cfg.obstacle_predictor != 'ca' or abs(cfg.physics_dt - .02) > TIME_ATOL
            or abs(cfg.horizon_s - 24.) > TIME_ATOL):
        raise ValueError('calibration requires the CA baseline, .02 s physics and 24 s horizon')
    summary = _read_json(directory / 'summary.json')
    if summary.get('risk_method', 'B0') != 'B0':
        raise ValueError('actual summary must identify B0 calibration data')
    if 'family' in row or 'config_overrides' in row:
        # Production registrations contain every common physical/sensing field.
        # Normalize both through Stage2Config, so JSON lists and tuple-valued
        # reference grids are compared as actual parameters rather than bytes.
        from .stage4_protocol import config_for_case
        expected = config_for_case(row)
        if cfg.to_dict() != expected.to_dict():
            raise ValueError('actual configuration differs from the registered common B0 parameters')
    scenario = row.get('scenario', summary.get('scenario'))
    if (not isinstance(scenario, str) or summary.get('scenario') != scenario
            or summary.get('backend') != 'remus6dof' or summary.get('seed') != cfg.seed
            or ('seed' in row and row['seed'] != cfg.seed)):
        raise ValueError('manifest, summary and actual configuration disagree')
    if summary.get('status') not in ('success', 'timeout', 'obstacle_collision', 'terrain_or_domain_collision'):
        raise ValueError('calibration requires a completed navigation run, including genuine research failures')
    states = _states(directory / 'states.csv')
    plans = _read_json(directory / 'planning_diagnostics.json')
    packets = _read_json(directory / 'observations.json')
    saved_snapshots = _read_json(directory / 'obstacles.json')
    if not isinstance(plans, list) or not isinstance(packets, list) or not isinstance(saved_snapshots, list):
        raise ValueError('calibration episode logs must contain lists of records')
    selected, absent = [], []
    for anchor in ANCHOR_TIMES_S:
        matches = [plan for plan in plans if abs(float(plan['time_s']) - anchor) <= TIME_ATOL]
        if len(matches) > 1:
            raise ValueError('duplicate planning diagnostics at a fixed anchor')
        if not matches:
            absent.append({'anchor_s': float(anchor), 'reason': 'no_plan_at_fixed_anchor'})
            continue
        plan = matches[0]
        ego = states[np.isclose(states[:, 0], anchor, atol=TIME_ATOL, rtol=0)]
        if len(ego) != 1:
            raise ValueError('planning anchor must have exactly one actual REMUS state')
        if not isinstance(plan['tracks'], list):
            raise ValueError('planning tracks must be a list')
        if not plan['tracks']:
            absent.append({'anchor_s': float(anchor), 'reason': 'no_observed_targets'})
            continue
        tracks, ids = [], []
        for item in plan['tracks']:
            track = Track(int(item['object_id']), float(item['radius']),
                          list(np.asarray(item['times_s'], float)),
                          list(np.asarray(item['positions'], float)))
            if track.object_id in ids:
                raise ValueError('duplicate object ID in planning track list')
            _verify_observed(track, float(anchor), packets)
            tracks.append(track)
            ids.append(track.object_id)
        selected.append((float(anchor), ego[0, 1:4], float(ego[0, 4]), tracks))
    scene = make_stage2_scenario(scenario, cfg)
    initial_ids, _, _ = _snapshot_arrays(scene.obstacles.snapshot())
    empty = len(initial_ids) == 0
    if not selected and not empty:
        raise ValueError('nonempty obstacle scene has no usable fixed-anchor observations; cannot impute zero error')
    until = max((entry[0] + 24. for entry in selected), default=0.)
    history, max_error, checked = _replay(scene, until, saved_snapshots)
    forecaster, windows = CAObstacleForecaster(cfg), []
    for anchor, origin, yaw, tracks in selected:
        samples = history[int(round(anchor / .2)):int(round(anchor / .2)) + len(QUERY_TIMES_S)]
        if len(samples) != len(QUERY_TIMES_S):
            raise ValueError('offline obstacle truth does not cover the whole 24 s query horizon')
        forecasts, labels, radii, object_ids = [], [], [], []
        for track in tracks:
            label = []
            for ids, points, true_radii in samples:
                indexes = np.flatnonzero(ids == track.object_id)
                if len(indexes) != 1:
                    raise ValueError('an observed target has no unique offline truth label')
                index = indexes[0]
                if abs(float(true_radii[index]) - track.radius_m) > POSITION_ATOL:
                    raise ValueError('observed target radius disagrees with offline truth geometry')
                label.append(points[index])
            forecast = forecaster.forecast(track, anchor + QUERY_TIMES_S, anchor)
            forecasts.append(yaw_local(forecast.positions, origin, yaw))
            labels.append(yaw_local(label, origin, yaw))
            radii.append(track.radius_m)
            object_ids.append(track.object_id)
        windows.append({'anchor_s': anchor, 'pred': np.asarray(forecasts),
                        'label': np.asarray(labels), 'radii': radii, 'object_ids': object_ids,
                        'ego_ned_m': origin.tolist(), 'ego_yaw_rad': yaw})
    metadata = {'case_id': row['case_id'], 'scenario_id': row.get('scenario_id', row['case_id']),
                'scenario': scenario, 'seed': cfg.seed, 'source_directory': str(directory),
                'navigation_status': summary['status'], 'empty_obstacle_scene': empty,
                'selected_anchor_s': [window['anchor_s'] for window in windows],
                'unavailable_fixed_anchors': absent, 'actual_end_time_s': float(states[-1, 0]),
                'replayed_snapshots_checked': checked, 'maximum_replay_position_error_m': max_error,
                'forecast_history_only_delivered_observations': True, 'configuration': cfg.to_dict()}
    return windows, metadata


def build_library(run_root, manifest_rows, out_npz):
    """Build the frozen library from exactly 24 B0 calibration-role scenes.

    Result directories are ``run_root/cases/<case_id>`` or an explicitly supplied
    ``result_dir`` within run_root. Other manifest splits/methods are not read.
    Missing files, causal observations or replay consistency raise exceptions;
    genuine collision/timeout runs are retained. Empty obstacle scenes are
    explicitly marked and contribute a mathematically empty-set zero score.
    """
    root, out = Path(run_root), Path(out_npz)
    if out.exists():
        raise FileExistsError(f'preserve existing calibration libraries: {out}')
    rows = _calibration_rows(manifest_rows)
    loaded = [_load_scene(root, row) for row in rows]
    object_count = max(1, max((len(window['object_ids']) for windows, _ in loaded
                              for window in windows), default=0))
    shape = (len(rows), len(ANCHOR_TIMES_S), object_count, len(QUERY_TIMES_S), 3)
    pred, label = np.full(shape, np.nan), np.full(shape, np.nan)
    radii = np.full(shape[:3], np.nan)
    object_ids = np.full(shape[:3], -1, dtype=np.int64)
    mask = np.zeros(shape[:3], dtype=bool)
    window_mask = np.zeros(shape[:2], dtype=bool)
    b1_scores = np.full(len(rows), np.nan)
    for scene_index, (windows, metadata) in enumerate(loaded):
        scores = []
        for window in windows:
            window_index = int(round(window['anchor_s'] / 20.))
            count = len(window['object_ids'])
            pred[scene_index, window_index, :count] = window['pred']
            label[scene_index, window_index, :count] = window['label']
            radii[scene_index, window_index, :count] = window['radii']
            object_ids[scene_index, window_index, :count] = window['object_ids']
            mask[scene_index, window_index, :count] = True
            window_mask[scene_index, window_index] = True
            scores.append(float(np.linalg.norm(window['pred'] - window['label'], axis=-1).max()))
        if scores:
            b1_scores[scene_index] = max(scores)
        elif metadata['empty_obstacle_scene']:
            b1_scores[scene_index] = 0.
        else:
            raise ValueError('missing calibration scene must not receive a zero score')
    metadata = {'schema': 'stage4_frozen_ca_scene_library_v1', 'role': 'calibration_only',
                'split': 'calibration',
                'calibration_scene_count': len(rows), 'statistical_unit': 'complete_scene',
                'fixed_anchor_times_s': ANCHOR_TIMES_S.tolist(), 'horizon_s': 24.,
                'query_dt_s': .2, 'replay_physics_dt_s': .02,
                'coordinate_frame': 'NED anchor-yaw frame; depth positive down',
                'alignment': 'rigid anchor ego translation and yaw; engineering transport assumption',
                'online_future_truth_available': False, 'offline_simulator_truth_labels': True,
                'online_ACP_implemented': False, 'scene_aggregation': 'maximum over fixed observed windows',
                'b1_score': 'scene maximum Euclidean CA position error over windows, matched objects and times',
                'coverage_claim': 'empirical pilot; no unconditional closed-loop guarantee',
                'scenes': [details for _, details in loaded]}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as stream:
        np.savez_compressed(stream, pred_local=pred, label_local=label, radii=radii, mask=mask,
                            window_mask=window_mask, object_ids=object_ids,
                            times_s=QUERY_TIMES_S, anchor_times_s=ANCHOR_TIMES_S,
                            case_ids=np.asarray([row['case_id'] for row in rows]),
                            b1_scene_scores_m=b1_scores,
                            empty_scene_mask=np.asarray([details['empty_obstacle_scene'] for _, details in loaded]),
                            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    return metadata
