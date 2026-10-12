"""Offline predictor/noise diagnostics on actual dev/tune/test scene roles.

Run as ``python -m scripts_stage5.prediction_noise --phase pilot --split dev
--registry ... --out NEW_DIRECTORY``. Nothing from this evaluation is fitted
back into calibration, the predictor, risk policy or navigation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from auv3d.stage3_forecasting import observation_speed
from auv3d.stage4_calibration import yaw_local
from auv3d.stage4_risk import sphere_scene_scores, upper_quantile
from auv3d.stage5_calibration import (
    ANCHOR_TIMES_S, QUERY_TIMES_S, _validate_sample,
    generate_evaluation_samples, stratified_ball_quantiles,
)
from auv3d.stage5_forecasting import forecaster_contract, make_forecaster
from auv3d.stage5_protocol import (
    PHASE_ALPHA, PHASES, PREDICTORS, SPEED_BIN_NAMES,
    config_for_case, manifest,
)


HORIZONS_S = (3., 6., 9., 12., 18., 24.)


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite(value):
    return float(value) if np.isfinite(value) else None


def load_registry(path, phase):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding='utf-8'))
    if value.get('schema') != 'stage5_artifact_registry_v1' or value.get('phase') != phase:
        raise ValueError('artifact registry requires the evaluated phase and stage5 schema')
    result = {}
    for row in value.get('calibration_libraries', []):
        key = (row.get('predictor'), row.get('noise_std_m'))
        if key in result or key[0] not in PREDICTORS or key[1] not in (0., .02, .1):
            raise ValueError('duplicate or unregistered predictor/noise library')
        target = Path(row['path'])
        target = target if target.is_absolute() else path.parent / target
        if not target.is_file():
            raise FileNotFoundError(target)
        result[key] = target
    if len(result) != 9:
        raise ValueError('diagnostics require all nine independent predictor/noise libraries')
    return result


def load_calibration(path, predictor, case, alpha):
    """Validate actual fields and compute independent-calibration q only."""
    cfg = config_for_case(case)
    with np.load(path, allow_pickle=False) as data:
        pred, label = np.asarray(data['pred_local'], float), np.asarray(data['label_local'], float)
        mask, speeds = np.asarray(data['mask'], bool), np.asarray(data['observed_speeds_mps'], float)
        times = np.asarray(data['times_s'], float)
        ids = [str(i) for i in data['case_ids']]
        scene_ids = [str(i) for i in data['scene_ids']]
        empty = np.asarray(data['empty_scene_mask'], bool)
        meta = json.loads(str(data['metadata_json'].item()))
    expected = dict(schema='stage5-risk-v1', split='calibration', phase=case['phase'],
        predictor=predictor, predictor_contract=forecaster_contract(predictor, cfg),
        noise_std_m=case['noise_std_m'], case_ids=ids, speed_bin_edges_mps=[0., .25, 'inf'])
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'calibration mismatch: {key}')
    if case['case_id'] in ids or case['scene_id'] in scene_ids:
        raise ValueError('evaluated case cannot occur in independent calibration')
    if (len(ids) != len(set(ids)) or len(ids) != len(pred)
            or len(scene_ids) != len(ids) or len(set(scene_ids)) != len(ids)):
        raise ValueError('calibration requires distinct complete scene cases')
    sources = meta.get('scenes', [])
    if len(sources) != len(ids):
        raise ValueError('calibration requires one actual source record per complete scene')
    for identity, scene_id, source in zip(ids, scene_ids, sources, strict=True):
        if (source.get('case_id') != identity or source.get('scene_id') != scene_id
                or source.get('split') != 'calibration' or source.get('phase') != case['phase']
                or source.get('noise_std_m') != case['noise_std_m']):
            raise ValueError('calibration source identity/role/noise differs from independent pool')
    if times.shape != QUERY_TIMES_S.shape or not np.allclose(times, QUERY_TIMES_S, atol=1e-10, rtol=0):
        raise ValueError('calibration and evaluation time grids differ')
    ball = stratified_ball_quantiles(pred, label, mask, speeds, alpha)
    active = np.any(mask, axis=(1, 2))
    if empty.shape != active.shape or np.any(active == empty):
        raise ValueError('missing calibration labels cannot become empty-scene zeros')
    safe_pred = np.where(mask[..., None, None], pred, 0.)
    safe_label = np.where(mask[..., None, None], label, 0.)
    joint = sphere_scene_scores(safe_pred, safe_label, mask)[active]
    joint_q = upper_quantile(joint, alpha) if len(joint) else float('inf')
    return dict(path=str(Path(path).resolve()), pointwise=ball, joint_q=joint_q,
                joint_scene_count=len(joint), metadata=meta)


def forecast_source(case, sample):
    """All predictors receive copies of the same delivered history, never labels."""
    from copy import deepcopy
    _validate_sample(sample, case, purpose='evaluation')
    cfg = config_for_case(case)
    windows = sample['windows']
    count = max(1, max((len(w['tracks']) for w in windows), default=0))
    shape = (len(ANCHOR_TIMES_S), count, len(QUERY_TIMES_S), 3)
    pred = np.zeros((len(PREDICTORS), *shape))
    label, local_label = np.zeros(shape), np.zeros(shape)
    mask, speeds = np.zeros(shape[:2], bool), np.zeros(shape[:2])
    ids = np.full(shape[:2], -1, dtype=np.int64)
    history_rows, forecast_records = [], []
    for window in windows:
        ai = int(np.flatnonzero(ANCHOR_TIMES_S == window['anchor_s'])[0])
        for oi, track in enumerate(window['tracks']):
            label[ai, oi] = window['label_world'][oi]
            local_label[ai, oi] = yaw_local(label[ai, oi], window['ego_ned_m'], window['ego_yaw_rad'])
            mask[ai, oi], speeds[ai, oi], ids[ai, oi] = True, observation_speed(track), track.object_id
            history_rows.append(dict(anchor_s=window['anchor_s'], object_id=track.object_id,
                acquisition_times_s=list(track.times), positions_world_m=np.asarray(track.positions).tolist(),
                radius_m=track.radius_m, observed_speed_mps=speeds[ai, oi]))
            for pi, predictor in enumerate(PREDICTORS):
                forecast = make_forecaster(predictor, cfg).forecast(deepcopy(track),
                    window['anchor_s'] + QUERY_TIMES_S, window['anchor_s'])
                forecast.validate(window['anchor_s'])
                if forecast.object_id != track.object_id or not np.array_equal(
                        forecast.absolute_times, window['anchor_s'] + QUERY_TIMES_S):
                    raise ValueError('forecast changed object identity or absolute anchor')
                pred[pi, ai, oi] = forecast.positions
                forecast_records.append(dict(predictor=predictor, anchor_s=window['anchor_s'],
                    object_id=track.object_id, metadata=forecast.metadata))
    return dict(pred_world=pred, label_world=label, label_local=local_label, mask=mask,
                observed_speeds_mps=speeds, object_ids=ids,
                histories=history_rows, forecast_records=forecast_records)


def evaluate_case(case, forecasts, calibrations, alpha):
    mask, speeds = forecasts['mask'], forecasts['observed_speeds_mps']
    labels, predictions = forecasts['label_world'], forecasts['pred_world']
    rmse_rows, coverage_rows, joint_rows = [], [], []
    for pi, predictor in enumerate(PREDICTORS):
        cal = calibrations[predictor]
        common = dict(case_id=case['case_id'], scene_id=case['scene_id'], phase=case['phase'],
            split=case['split'], family=case['family'], predictor=predictor,
            noise_std_m=case['noise_std_m'], alpha=alpha)
        errors = np.linalg.norm(predictions[pi] - labels, axis=-1)
        for horizon in HORIZONS_S:
            ti = int(np.flatnonzero(np.isclose(QUERY_TIMES_S, horizon, atol=1e-9, rtol=0))[0])
            values = errors[..., ti][mask]
            sse = float(np.sum(values ** 2))
            count = len(values)
            rmse_rows.append(dict(common, horizon_s=horizon, position_label_count=count,
                squared_euclidean_error_sum_m2=sse,
                prediction_RMSE_m=float(np.sqrt(sse / count)) if count else None))
            for bi, name in enumerate(SPEED_BIN_NAMES):
                group = mask & (speeds < .25 if bi == 0 else speeds >= .25)
                group_errors = errors[..., ti][group]
                q = cal['pointwise']['quantiles_m'][bi, ti]
                n = int(cal['pointwise']['scene_counts'][bi])
                label_count = len(group_errors)
                finite = np.isfinite(q)
                covered = int(np.sum(group_errors <= q + 1e-9)) if finite else None
                coverage_rows.append(dict(common, horizon_s=horizon, speed_bin=bi, speed_bin_name=name,
                    calibration_scene_count=n, finite_sample_rank=int(np.ceil((n + 1) * (1 - alpha))),
                    quantile_m=_finite(q), quantile_status='finite' if finite else 'insufficient_samples',
                    position_label_count=label_count, covered_position_labels=covered,
                    empirical_position_coverage=(covered / label_count) if finite and label_count else None,
                    evaluation_scene_horizon_max_error_m=float(group_errors.max()) if label_count else None,
                    evaluation_scene_horizon_covered=bool(group_errors.max() <= q + 1e-9) if finite and label_count else None))
        all_errors = errors[mask]
        q = cal['joint_q']
        finite = np.isfinite(q)
        maximum = float(all_errors.max()) if all_errors.size else None
        joint_rows.append(dict(common, calibration_active_scene_count=cal['joint_scene_count'],
            joint_quantile_m=_finite(q), quantile_status='finite' if finite else 'insufficient_samples',
            position_label_count=int(all_errors.size),
            evaluation_scene_joint_max_error_m=maximum,
            evaluation_scene_joint_covered=bool(maximum <= q + 1e-9) if finite and maximum is not None else None,
            covered_position_labels=int(np.sum(all_errors <= q + 1e-9)) if finite and all_errors.size else None))
    return rmse_rows, coverage_rows, joint_rows


def aggregate_rmse(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['phase'], row['split'], row['predictor'], row['noise_std_m'], row['horizon_s'])].append(row)
    output = []
    for key, group in sorted(groups.items()):
        count = sum(r['position_label_count'] for r in group)
        sse = sum(r['squared_euclidean_error_sum_m2'] for r in group)
        output.append(dict(zip(('phase', 'split', 'predictor', 'noise_std_m', 'horizon_s'), key),
            evaluated_case_count=len(group), cases_with_labels=sum(r['position_label_count'] > 0 for r in group),
            position_label_count=count, squared_euclidean_error_sum_m2=sse,
            prediction_RMSE_m=float(np.sqrt(sse / count)) if count else None))
    return output


def aggregate_coverage(point_rows, joint_rows):
    groups = defaultdict(list)
    for row in point_rows:
        groups[(row['phase'], row['split'], row['predictor'], row['noise_std_m'], row['horizon_s'], row['speed_bin'])].append(row)
    point = []
    for key, group in sorted(groups.items()):
        known = [r for r in group if r['quantile_status'] == 'finite' and r['position_label_count'] > 0]
        n = sum(r['position_label_count'] for r in known)
        covered = sum(r['covered_position_labels'] for r in known)
        point.append(dict(zip(('phase', 'split', 'predictor', 'noise_std_m', 'horizon_s', 'speed_bin'), key),
            calibration_scene_count=group[0]['calibration_scene_count'],
            quantile_m=group[0]['quantile_m'], quantile_status=group[0]['quantile_status'],
            position_label_count=n, covered_position_labels=covered if n else None,
            empirical_position_coverage=covered/n if n else None,
            evaluation_scene_count_with_labels=len(known),
            evaluation_scene_horizon_covered_count=sum(r['evaluation_scene_horizon_covered'] for r in known),
            empirical_scene_horizon_coverage=sum(r['evaluation_scene_horizon_covered'] for r in known)/len(known) if known else None))
    groups = defaultdict(list)
    for row in joint_rows:
        groups[(row['phase'], row['split'], row['predictor'], row['noise_std_m'])].append(row)
    joint = []
    for key, group in sorted(groups.items()):
        known = [r for r in group if r['evaluation_scene_joint_covered'] is not None]
        joint.append(dict(zip(('phase', 'split', 'predictor', 'noise_std_m'), key),
            calibration_active_scene_count=group[0]['calibration_active_scene_count'],
            joint_quantile_m=group[0]['joint_quantile_m'], quantile_status=group[0]['quantile_status'],
            evaluation_scene_count_with_labels=len(known),
            covered_scene_count=sum(r['evaluation_scene_joint_covered'] for r in known),
            empirical_scene_joint_coverage=sum(r['evaluation_scene_joint_covered'] for r in known)/len(known) if known else None,
            position_label_count=sum(r['position_label_count'] for r in known)))
    return point, joint


def _plots(out, rmse, calibrations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True)
    for ax, noise in zip(axes, (0., .02, .1)):
        for predictor in PREDICTORS:
            rows = [r for r in rmse if r['predictor'] == predictor and r['noise_std_m'] == noise and r['prediction_RMSE_m'] is not None]
            ax.plot([r['horizon_s'] for r in rows], [r['prediction_RMSE_m'] for r in rows], marker='o', label=predictor)
        ax.set(title=f'noise std = {noise:g} m', xlabel='Prediction horizon (s)', ylabel='Euclidean position RMSE (m)')
        ax.grid(alpha=.3)
        ax.legend()
    fig.suptitle('Offline fixed-carrier evaluation; no REMUS navigation claim')
    fig.tight_layout()
    fig.savefig(out / 'PREDICTION_RMSE.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for column, noise in enumerate((0., .02, .1)):
        for bi in range(2):
            ax = axes[bi, column]
            for predictor in PREDICTORS:
                q = calibrations[(predictor, noise)]['pointwise']['quantiles_m'][bi]
                ax.plot(QUERY_TIMES_S, np.where(np.isfinite(q), q, np.nan), label=predictor)
            ax.set(title=f'{SPEED_BIN_NAMES[bi]}, noise={noise:g} m', xlabel='Prediction horizon (s)', ylabel='A2 calibrated radius (m)')
            ax.grid(alpha=.3)
            ax.legend()
    fig.suptitle('Independent calibration only; per-time radii are not joint/task coverage')
    fig.tight_layout()
    fig.savefig(out / 'A2_TIME_SPEED_QUANTILES.png', dpi=150)
    plt.close(fig)


def run_diagnostics(phase, split, registry, out, *, plot=True):
    if phase not in PHASES or split not in ('dev', 'tune', 'test'):
        raise ValueError('evaluate a registered dev/tune/test role, never calibration as unseen')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cases = manifest(phase, split=split)
    alpha = PHASE_ALPHA[phase]
    paths = load_registry(registry, phase)
    calibration = {}
    for predictor, noise in paths:
        case = next(row for row in cases if row['noise_std_m'] == noise)
        calibration[(predictor, noise)] = load_calibration(paths[(predictor, noise)], predictor, case, alpha)
    q_rows = []
    for (predictor, noise), cal in calibration.items():
        for bi, name in enumerate(SPEED_BIN_NAMES):
            n = int(cal['pointwise']['scene_counts'][bi])
            for ti, horizon in enumerate(QUERY_TIMES_S):
                q = cal['pointwise']['quantiles_m'][bi, ti]
                q_rows.append(dict(phase=phase, predictor=predictor, noise_std_m=noise, alpha=alpha,
                    speed_bin=bi, speed_bin_name=name, horizon_s=float(horizon),
                    calibration_scene_count=n, finite_sample_rank=int(np.ceil((n+1)*(1-alpha))),
                    quantile_m=_finite(q), quantile_status='finite' if np.isfinite(q) else 'insufficient_samples',
                    library_path=cal['path']))
    _json(out/'EXPERIMENT_CONFIG.json', dict(phase=phase, split=split, alpha=alpha,
        registry=str(Path(registry).resolve()), cases=cases, predictors=list(PREDICTORS),
        evaluation_kind='offline_fixed_causal_carrier', calibration_modified=False,
        future_truth_used_online=False, no_hash_checks=True,
        independent_unit='complete scene_id, not noise replicas or correlated windows'))
    rmse_rows, point_rows, joint_rows, failures = [], [], [], []
    for case in cases:
        directory = out/'cases'/case['case_id']
        directory.mkdir(parents=True)
        try:
            sample = generate_evaluation_samples(case)
            forecasts = forecast_source(case, sample)
            selected = {predictor: calibration[(predictor, case['noise_std_m'])] for predictor in PREDICTORS}
            rmse, point, joint = evaluate_case(case, forecasts, selected, alpha)
            rmse_rows.extend(rmse); point_rows.extend(point); joint_rows.extend(joint)
            with (directory/'RAW_FORECASTS.npz').open('xb') as stream:
                np.savez_compressed(stream, pred_world=forecasts['pred_world'],
                    label_world=forecasts['label_world'], label_local=forecasts['label_local'],
                    mask=forecasts['mask'], observed_speeds_mps=forecasts['observed_speeds_mps'],
                    object_ids=forecasts['object_ids'], times_s=QUERY_TIMES_S,
                    anchor_times_s=ANCHOR_TIMES_S, predictors=np.asarray(PREDICTORS))
            _json(directory/'ACQUISITION.json', sample['metadata'])
            _json(directory/'HISTORIES_AND_FORECAST_RECORDS.json', dict(
                case=case, histories=forecasts['histories'], forecasts=forecasts['forecast_records']))
            _csv(directory/'RMSE_BY_HORIZON.csv', rmse)
            _csv(directory/'A2_TIME_SPEED_COVERAGE.csv', point)
            _csv(directory/'A3_JOINT_COVERAGE.csv', joint)
            print(f"{case['case_id']}: labels={int(forecasts['mask'].sum())} windows/objects, three predictors", flush=True)
        except Exception as error:
            failure = dict(case_id=case['case_id'], scene_id=case['scene_id'],
                error_type=type(error).__name__, message=str(error))
            failures.append(failure)
            _json(directory/'PROGRAM_ERROR.json', failure)
            print(f"{case['case_id']}: ERROR {type(error).__name__}: {error}", flush=True)
    grouped_rmse = aggregate_rmse(rmse_rows)
    grouped_point, grouped_joint = aggregate_coverage(point_rows, joint_rows)
    for filename, rows in (
        ('RMSE_BY_CASE_HORIZON.csv', rmse_rows), ('RMSE_BY_HORIZON.csv', grouped_rmse),
        ('A2_CALIBRATION_TIME_SPEED_QUANTILES.csv', q_rows),
        ('A2_COVERAGE_BY_CASE.csv', point_rows), ('A2_EMPIRICAL_COVERAGE.csv', grouped_point),
        ('A3_JOINT_COVERAGE_BY_CASE.csv', joint_rows), ('A3_EMPIRICAL_JOINT_COVERAGE.csv', grouped_joint)):
        _csv(out/filename, rows)
    if plot:
        _plots(out, grouped_rmse, calibration)
    summary = dict(completed_utc=datetime.now(timezone.utc).isoformat(), phase=phase, split=split,
        requested_source_cases=len(cases), unique_scene_count=len({r['scene_id'] for r in cases}),
        completed_source_cases=len(cases)-len(failures), predictor_case_evaluations=(len(cases)-len(failures))*3,
        program_error_count=len(failures), errors=failures, calibration_modified=False,
        future_truth_used_online=False, evaluation_kind='offline fixed carrier, not REMUS navigation',
        coverage_scope='position labels and whole-scene horizon/joint scores reported separately; no task-level guarantee',
        unseen_scope='dev is development diagnostics; tune is selection diagnostics; only new test is held out',
        output_directory=str(out.resolve()))
    _json(out/'SUMMARY.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=PHASES, default='pilot')
    parser.add_argument('--split', choices=('dev', 'tune', 'test'), default='dev')
    parser.add_argument('--registry', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)
    result = run_diagnostics(args.phase, args.split, args.registry, args.out, plot=not args.no_plot)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 2 if result['program_error_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
