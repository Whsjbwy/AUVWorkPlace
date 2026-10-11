"""Offline pointwise counterexamples to saved REMUS/CA planning decisions.

Read saved full-REMUS prediction samples; never replay or command a vehicle.
Same-seed obstacle future truth is generated only here, after execution, and
checked against recorded snapshots before any forecast-error attribution.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from auv3d.remus_planner import _forecast_tracks
from auv3d.sensors import Track
from auv3d.stage2_config import Stage2Config
from auv3d.stage2_failure_analysis import _truth_at_times


REQUIRED = ('summary.json', 'config.json', 'selected_plans.json',
            'planning_diagnostics.json', 'planning.csv', 'states.csv', 'obstacles.json')
SCOPE = ('Point samples from saved original REMUS integrations, normally 0.2 s apart. '
         'A sub-margin sample proves a violation at that point. Absence of such samples '
         'does not prove continuous safety or replace the original 0.02 s swept checks. '
         'Only samples through the next decision or mission end belong to executed '
         'holding intervals; later vehicle positions are unexecuted counterfactuals.')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(path, rows, empty_columns):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or list(empty_columns)
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def tick_at(time_s, dt):
    if not np.isfinite(time_s):
        raise ValueError('saved simulation time must be finite')
    tick = round(float(time_s)/dt)
    if tick < 0 or abs(tick*dt-time_s) > 1e-8:
        raise ValueError('saved time does not coincide with an original physics tick')
    return tick


def minimum(values):
    values = np.asarray(values)
    return float(values.min()) if values.size else None


def delivered_ids(observation_packets, now):
    ids = set()
    for packet in observation_packets or ():
        delivered = packet.get('delivered_time_s')
        if delivered is not None and delivered <= now+1e-9:
            ids.update(int(row['object_id']) for row in packet['objects'])
    return ids


def pointwise_decision(plan, detail, live_clearance, end_time, cfg, truth,
                       actual_by_tick, observations):
    """Evaluate saved points, with held and counterfactual intervals separated."""
    now = float(plan['time_s'])
    rows = np.asarray(plan['states'], dtype=float)
    if (rows.ndim != 2 or rows.shape[1] != 16 or len(rows) < 2
            or not np.isfinite(rows).all() or abs(rows[0, 0]) > 1e-9
            or np.any(np.diff(rows[:, 0]) <= 0)):
        raise ValueError('saved selected plan must contain finite increasing [offset,eta6,nu6,actuator3] rows')
    if plan.get('validation_backend') != 'original_remus6dof':
        raise ValueError('counterfactual attribution requires saved original REMUS predictions')
    if not np.isfinite(live_clearance) or live_clearance < cfg.safety_margin_m:
        raise ValueError('a saved feasible plan lacks a live CA-validated safety clearance')
    if float(detail['time_s']) != now or not detail['feasible']:
        raise ValueError('selected plan and causal planning diagnostics disagree')
    ticks = np.array([tick_at(now+offset, cfg.physics_dt) for offset in rows[:, 0]], dtype=int)
    end_tick = tick_at(end_time, cfg.physics_dt)
    held = ticks <= end_tick
    if not held.any():
        raise ValueError('selected plan has no sample in its actual holding interval')
    vehicle_error = {}
    for index in np.flatnonzero(held):
        tick = int(ticks[index])
        if tick not in actual_by_tick:
            raise ValueError('a held selected-plan sample has no actual physical state')
        vehicle_error[index] = float(np.linalg.norm(rows[index, 1:4]-actual_by_tick[tick]))
    maximum_vehicle_error = max(vehicle_error.values(), default=0.)
    if maximum_vehicle_error > 1e-8:
        raise ValueError('held vehicle prediction differs from actual execution; forecast-only attribution is confounded')
    snapshots = [truth[int(tick)] for tick in ticks]
    ids = np.asarray(snapshots[0].ids, dtype=int)
    if any(not np.array_equal(snapshot.ids, ids) for snapshot in snapshots):
        raise ValueError('obstacle IDs change inside an offline decision interval')
    positions = np.asarray([snapshot.positions for snapshot in snapshots], dtype=float)
    radii = np.asarray([snapshot.radii for snapshot in snapshots], dtype=float)
    gaps = np.linalg.norm(rows[:, None, 1:4]-positions, axis=-1)-radii-cfg.body_radius_m
    tracks = [Track(int(row['object_id']), float(row['radius']), list(row['times_s']),
                    [np.asarray(position, dtype=float) for position in row['positions']])
              for row in detail['tracks']]
    if len({track.object_id for track in tracks}) != len(tracks):
        raise ValueError('duplicate tracked IDs at a saved decision')
    index_by_id = {int(object_id): i for i, object_id in enumerate(ids)}
    if any(track.object_id not in index_by_id for track in tracks):
        raise ValueError('tracked obstacle ID has no same-seed truth counterpart')
    forecasts = _forecast_tracks(tracks, rows[:, 0], now, cfg)
    tracked_ids = {track.object_id for track in tracks}
    observed_before = delivered_ids(observations, now)
    tracked_mask = np.array([int(object_id) in tracked_ids for object_id in ids], dtype=bool)
    ca_by_id, errors_by_id = {}, {}
    tracked_rows = []
    for track, (predicted, combined_radius) in zip(tracks, forecasts):
        object_index = index_by_id[track.object_id]
        ca_gap = np.linalg.norm(rows[:, 1:4]-predicted, axis=1)-combined_radius
        if np.any(ca_gap < live_clearance-1e-6):
            raise ValueError('recomputed CA point gap contradicts saved full-physics live minimum')
        ca_by_id[track.object_id] = ca_gap
        forecast_error = np.linalg.norm(predicted-positions[:, object_index], axis=1)
        errors_by_id[track.object_id] = forecast_error
        for i, tick in enumerate(ticks):
            tracked_rows.append({
                'plan_time_s': now, 'absolute_time_s': int(tick)*cfg.physics_dt,
                'offset_s': float(rows[i, 0]), 'object_id': track.object_id,
                'executed_holding_interval': bool(held[i]),
                'true_physical_point_clearance_m': float(gaps[i, object_index]),
                'ca_inflated_point_clearance_m': float(ca_gap[i]),
                'live_full_physics_ca_minimum_m': float(live_clearance),
                'obstacle_ca_position_error_m': float(forecast_error[i]),
                'observation_age_at_decision_s': now-track.times[-1],
                'track_history_size': len(track.times),
                'vehicle_sample_execution_error_m': vehicle_error.get(i),
                'vehicle_north_m': float(rows[i, 1]), 'vehicle_east_m': float(rows[i, 2]),
                'vehicle_depth_m': float(rows[i, 3]),
                'truth_obstacle_north_m': float(positions[i, object_index, 0]),
                'truth_obstacle_east_m': float(positions[i, object_index, 1]),
                'truth_obstacle_depth_m': float(positions[i, object_index, 2]),
                'ca_obstacle_north_m': float(predicted[i, 0]),
                'ca_obstacle_east_m': float(predicted[i, 1]),
                'ca_obstacle_depth_m': float(predicted[i, 2]),
            })
    hazard_rows = []
    for i, j in np.argwhere(gaps < cfg.safety_margin_m):
        object_id = int(ids[j])
        tracked = object_id in tracked_ids
        category = ('tracked_ca_forecast_false_safe' if tracked
                    else 'not_currently_tracked_delivery_history_unavailable' if observations is None
                    else 'previously_delivered_but_not_currently_tracked' if object_id in observed_before
                    else 'not_yet_delivered_to_planner_at_decision')
        hazard_rows.append({
            'plan_time_s': now, 'absolute_time_s': int(ticks[i])*cfg.physics_dt,
            'offset_s': float(rows[i, 0]), 'object_id': object_id,
            'scope': 'executed_holding_interval' if held[i] else 'unexecuted_counterfactual_future',
            'category': category, 'tracked_at_decision': tracked,
            'true_physical_point_clearance_m': float(gaps[i, j]),
            'required_safety_margin_m': cfg.safety_margin_m,
            'live_full_physics_ca_minimum_m': float(live_clearance),
            'ca_inflated_point_clearance_m': float(ca_by_id[object_id][i]) if tracked else None,
            'obstacle_ca_position_error_m': float(errors_by_id[object_id][i]) if tracked else None,
            'vehicle_sample_execution_error_m': vehicle_error.get(int(i)),
            'vehicle_north_m': float(rows[i, 1]), 'vehicle_east_m': float(rows[i, 2]),
            'vehicle_depth_m': float(rows[i, 3]),
            'truth_obstacle_north_m': float(positions[i, j, 0]),
            'truth_obstacle_east_m': float(positions[i, j, 1]),
            'truth_obstacle_depth_m': float(positions[i, j, 2]),
        })
    tracked_held = gaps[held][:, tracked_mask]
    all_held = gaps[held]
    tracked_hazards = [row for row in hazard_rows if row['tracked_at_decision']]
    executed_tracked_hazards = [row for row in tracked_hazards if row['scope'] == 'executed_holding_interval']
    executed_untracked_hazards = [row for row in hazard_rows
                                  if not row['tracked_at_decision'] and row['scope'] == 'executed_holding_interval']
    decision = {
        'plan_time_s': now, 'holding_interval_end_s': end_time,
        'last_saved_held_sample_s': int(ticks[held][-1])*cfg.physics_dt,
        'unsampled_holding_tail_s': max(0., end_time-int(ticks[held][-1])*cfg.physics_dt),
        'saved_horizon_s': float(rows[-1, 0]), 'saved_point_count': len(rows),
        'maximum_saved_point_step_s': float(np.diff(rows[:, 0]).max()),
        'held_saved_point_count': int(held.sum()), 'tracked_object_count': len(tracks),
        'truth_object_count': len(ids), 'live_full_physics_ca_minimum_m': float(live_clearance),
        'minimum_tracked_true_point_clearance_held_m': minimum(tracked_held),
        'minimum_all_true_point_clearance_held_m': minimum(all_held),
        'minimum_untracked_true_point_clearance_held_m': minimum(gaps[held][:, ~tracked_mask]),
        'minimum_tracked_true_point_clearance_full_horizon_m': minimum(gaps[:, tracked_mask]),
        'minimum_all_true_point_clearance_full_horizon_m': minimum(gaps),
        'tracked_counterexample_held_sample_count': len(executed_tracked_hazards),
        'tracked_counterexample_full_horizon_sample_count': len(tracked_hazards),
        'untracked_hazard_held_sample_count': len(executed_untracked_hazards),
        'vehicle_held_sample_execution_max_error_m': maximum_vehicle_error,
    }
    return decision, hazard_rows, tracked_rows


def analyze_case(source, out):
    source, out = Path(source).resolve(), Path(out)
    if not all((source/name).is_file() for name in REQUIRED):
        raise FileNotFoundError(f'incomplete saved REMUS run: {source}')
    cfg = Stage2Config.load(source/'config.json')
    if cfg.command_delay_s != 0.:
        raise ValueError('this held-reference offline attribution supports command_delay_s=0 only')
    saved_summary = read_json(source/'summary.json')
    if saved_summary['backend'] != 'remus6dof':
        raise ValueError('P25 kinematics cannot be attributed as original REMUS execution')
    plans = read_json(source/'selected_plans.json')
    details = read_json(source/'planning_diagnostics.json')
    snapshots = read_json(source/'obstacles.json')
    observations = read_json(source/'observations.json') if (source/'observations.json').is_file() else None
    states = np.loadtxt(source/'states.csv', delimiter=',', skiprows=1, ndmin=2)
    if states.shape[1] != 16 or not np.isfinite(states).all() or np.any(np.diff(states[:, 0]) <= 0):
        raise ValueError('actual REMUS state log must have finite increasing 16-column rows')
    actual_by_tick = {tick_at(row[0], cfg.physics_dt): row[1:4] for row in states}
    if len(actual_by_tick) != len(states):
        raise ValueError('duplicate actual physical ticks')
    with (source/'planning.csv').open(encoding='utf-8', newline='') as stream:
        planning = list(csv.DictReader(stream))
    live_clearance = {tick_at(float(row['time_s']), cfg.physics_dt): float(row['predicted_clearance_m'])
                      for row in planning}
    detail_by_tick = {tick_at(row['time_s'], cfg.physics_dt): row for row in details}
    plan_ticks = [tick_at(row['time_s'], cfg.physics_dt) for row in plans]
    if len(set(plan_ticks)) != len(plans) or np.any(np.diff(plan_ticks) <= 0):
        raise ValueError('saved planning decisions must have unique increasing physical times')
    terminal = float(states[-1, 0])
    requested = {tick_at(plan['time_s']+row[0], cfg.physics_dt)
                 for plan in plans if plan['feasible'] for row in plan['states']}
    started = time.perf_counter()
    truth, verification = _truth_at_times(cfg, saved_summary['scenario'], requested, snapshots)
    decisions, hazards, tracked_samples = [], [], []
    for i, plan in enumerate(plans):
        if not plan['feasible']:
            continue
        tick = plan_ticks[i]
        end = min(terminal, float(plans[i+1]['time_s']) if i+1 < len(plans) else terminal)
        decision, found, samples = pointwise_decision(
            plan, detail_by_tick[tick], live_clearance[tick], end, cfg, truth, actual_by_tick, observations)
        decisions.append(decision)
        hazards.extend(found)
        tracked_samples.extend(samples)
    executed = [row for row in hazards if row['scope'] == 'executed_holding_interval']
    executed_tracked = [row for row in executed if row['tracked_at_decision']]
    executed_untracked = [row for row in executed if not row['tracked_at_decision']]
    distinct = lambda rows: len({(tick_at(row['absolute_time_s'], cfg.physics_dt), row['object_id']) for row in rows})
    summary = {
        'source_directory': str(source), 'scenario': saved_summary['scenario'],
        'candidate_budget': cfg.candidate_budget, 'seed': cfg.seed,
        'navigation_status': saved_summary['status'],
        'original_avoidance_acceptance_passed': saved_summary['avoidance_acceptance_passed'],
        'offline_only': True, 'new_vehicle_rollouts': 0, 'feedback_to_planner': False,
        'future_truth_source': 'same-seed shared obstacle generator checked against recorded snapshots',
        'observation_delivery_log_available': observations is not None,
        **verification, 'decision_count': len(plans), 'feasible_decisions_analyzed': len(decisions),
        'tracked_forecast_point_samples': len(tracked_samples),
        'tracked_ca_counterexample_executed_decision_count': sum(row['tracked_counterexample_held_sample_count'] > 0 for row in decisions),
        'tracked_ca_counterexample_executed_sample_count': len(executed_tracked),
        'tracked_ca_counterexample_executed_distinct_time_id_count': distinct(executed_tracked),
        'tracked_ca_counterexample_full_horizon_decision_count': sum(row['tracked_counterexample_full_horizon_sample_count'] > 0 for row in decisions),
        'untracked_hazard_executed_sample_count': len(executed_untracked),
        'untracked_hazard_executed_distinct_time_id_count': distinct(executed_untracked),
        'untracked_hazard_executed_categories': {name: sum(row['category'] == name for row in executed_untracked)
            for name in sorted({row['category'] for row in executed_untracked})},
        'vehicle_held_sample_execution_max_error_m': max((row['vehicle_held_sample_execution_max_error_m'] for row in decisions), default=0.),
        'maximum_unsampled_holding_tail_s': max((row['unsampled_holding_tail_s'] for row in decisions), default=0.),
        'maximum_saved_point_step_s': max((row['maximum_saved_point_step_s'] for row in decisions), default=0.),
        'minimum_tracked_true_point_clearance_held_m': minimum([row['true_physical_point_clearance_m'] for row in tracked_samples if row['executed_holding_interval']]),
        'strongest_executed_tracked_counterexample': min(executed_tracked, key=lambda row: row['true_physical_point_clearance_m']) if executed_tracked else None,
        'counterexample_C_demonstrated_at_saved_executed_sample': bool(executed_tracked),
        'absence_of_sampled_counterexample_proves_safety': False,
        'sampling_and_attribution_limits': SCOPE,
        'wall_clock_s': time.perf_counter()-started,
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'summary.json', summary)
    write_csv(out/'decisions.csv', decisions, ('plan_time_s',))
    write_csv(out/'hazard_points.csv', hazards, ('plan_time_s', 'absolute_time_s', 'object_id', 'scope'))
    write_csv(out/'tracked_forecast_samples.csv', tracked_samples, ('plan_time_s', 'absolute_time_s', 'object_id'))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'results_stage2/review_20261011_budget')
    parser.add_argument('--out', type=Path, default=ROOT/'results_stage2/review_20261011_counterfactual')
    parser.add_argument('--budgets', nargs='+', type=int, choices=(9, 18, 45), default=[9, 18, 45])
    parser.add_argument('--cases', nargs='+', help='case directory stems, for example multi_speed15')
    parser.add_argument('--require-complete', action='store_true', help='require every study child to have completed')
    args = parser.parse_args(argv)
    study = args.source.resolve()
    manifest = read_json(study/'budget_study.json')
    if args.require_complete and (manifest['completed_count'] != manifest['case_count'] or not manifest.get('finished_at')):
        raise RuntimeError('the study is still running; no final all-case attribution can be claimed')
    chosen = [row for row in manifest['records'] if row['exit_code'] == 0 and row['budget'] in args.budgets
              and (not args.cases or row['case'] in args.cases)]
    if not chosen:
        raise RuntimeError('no completed matching cases are available')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    results, failures = [], []
    started = time.perf_counter()
    for row in sorted(chosen, key=lambda item: (item['case'], item['budget'])):
        name = row.get('name', f'{row["case"]}_budget{row["budget"]}')
        print(f'OFFLINE {name}', flush=True)
        try:
            result = analyze_case(study/name, out/name)
            result['case'] = row['case']
            results.append(result)
            print(f'C executed decisions={result["tracked_ca_counterexample_executed_decision_count"]}; '
                  f'untracked executed samples={result["untracked_hazard_executed_sample_count"]}', flush=True)
        except Exception as error:
            failures.append({'case': row['case'], 'budget': row['budget'],
                             'error': f'{type(error).__name__}: {error}'})
            print(f'FAILED {name}: {type(error).__name__}: {error}', flush=True)
    summary = {
        'created_at': datetime.now().astimezone().isoformat(), 'actual_argv': [sys.executable, *sys.argv],
        'source_study_manifest': str(study/'budget_study.json'),
        'source_study_completed_count_at_start': manifest['completed_count'],
        'source_study_expected_case_count': manifest['case_count'],
        'final_complete_study_required': args.require_complete,
        'analyzed_case_count': len(results), 'analysis_failure_count': len(failures), 'failures': failures,
        'counterexample_C_case_count': sum(row['counterexample_C_demonstrated_at_saved_executed_sample'] for row in results),
        'tracked_ca_counterexample_executed_decision_count': sum(row['tracked_ca_counterexample_executed_decision_count'] for row in results),
        'offline_only': True, 'new_vehicle_rollouts': 0, 'feedback_to_planner': False,
        'sampling_and_attribution_limits': SCOPE, 'wall_clock_s': time.perf_counter()-started,
        'cases': results,
    }
    write_json(out/'analysis_summary.json', summary)
    rows = [{key: row[key] for key in ('case', 'scenario', 'candidate_budget', 'navigation_status',
                                      'original_avoidance_acceptance_passed', 'feasible_decisions_analyzed',
                                      'counterexample_C_demonstrated_at_saved_executed_sample',
                                      'tracked_ca_counterexample_executed_decision_count',
                                      'tracked_ca_counterexample_executed_distinct_time_id_count',
                                      'tracked_ca_counterexample_full_horizon_decision_count',
                                      'untracked_hazard_executed_distinct_time_id_count',
                                      'minimum_tracked_true_point_clearance_held_m',
                                      'vehicle_held_sample_execution_max_error_m')} for row in results]
    write_csv(out/'case_comparison.csv', rows, ('case', 'candidate_budget'))
    text = ['# Offline selected-reference forecast audit', '', SCOPE, '',
            '| Case | Budget | Feasible decisions | Executed tracked C decisions | Full-horizon tracked C decisions | Untracked executed time/ID points |',
            '|---|---:|---:|---:|---:|---:|']
    for row in results:
        text.append(f'| {row["case"]} | {row["candidate_budget"]} | {row["feasible_decisions_analyzed"]} | '
                    f'{row["tracked_ca_counterexample_executed_decision_count"]} | '
                    f'{row["tracked_ca_counterexample_full_horizon_decision_count"]} | '
                    f'{row["untracked_hazard_executed_distinct_time_id_count"]} |')
    text += ['', 'C is established here only when a tracked obstacle has a true physical point gap below the unchanged 1 m requirement inside the actual held-reference interval, despite the live full-physics CA minimum passing that requirement. Saved vehicle points are checked against actual execution. This demonstrates failure of that forecast-and-margin decision, without claiming that forecast error is the sole system-level cause or that a different action would succeed.',
             '', 'An object absent from the decision track set is reported separately as an information/tracking gap. A previously delivered object is distinguished from one not yet delivered to the planner. Full-horizon counterexamples after the next decision are hypothetical continuations and are never counted as actual execution failures.',
             '', 'Per-case files: `decisions.csv`, `hazard_points.csv`, `tracked_forecast_samples.csv`, and `summary.json`. The strongest executed tracked counterexample retains plan time, absolute time, object ID, CA/true gap, obstacle forecast error, and matching vehicle point. Raw experiments are read only. No vehicle is reintegrated and no result feeds back into planning.', '']
    if failures:
        text += ['Analysis failures are retained in `analysis_summary.json`; those cases have no successful attribution claim.', '']
    (out/'FORECAST_DECISIONS.md').write_text('\n'.join(text), encoding='utf-8')
    print(out/'analysis_summary.json', flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
