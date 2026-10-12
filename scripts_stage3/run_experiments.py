"""Registered, causal REMUS ablations with separate navigation/safety metrics.

Every case retains the common original dynamics, scene, collision monitor and
one-metre safety threshold. Group summaries vary only random seed; different
budgets, predictors, terrain guidance or sensor settings never share a rate.
Wall-clock planning is measured during synchronous offline execution, and is
not injected as physical command latency. Parallel runs share CPU resources.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import chdir, redirect_stderr, redirect_stdout
import csv
from dataclasses import dataclass, replace
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from auv3d.stage2_config import Stage2Config


SUITES = ('smoke', 'memory', 'candidate', 'predictor', 'all')
SENSOR_VARIANTS = (
    ('fov120x30_obs1', 120., 30., 1.),
    ('fov90x6_obs1', 90., 6., 1.),
    ('fov120x30_obs2', 120., 30., 2.),
)
MEMORY_VARIANTS = (
    ('hard4', {'memory_policy': 'hard_expiry', 'track_max_age_s': 4.}),
    ('hard8', {'memory_policy': 'hard_expiry', 'track_max_age_s': 8.}),
    ('hard12', {'memory_policy': 'hard_expiry', 'track_max_age_s': 12.}),
    ('retain_uncertain', {'memory_policy': 'retain_uncertain', 'track_max_age_s': 4.}),
    ('conditional', {'memory_policy': 'conditional', 'track_max_age_s': 4.}),
)
CANDIDATE_VARIANTS = (
    ('diverse9', {'candidate_budget': 9}),
    ('diverse18', {'candidate_budget': 18}),
    ('diverse45', {'candidate_budget': 45}),
    ('ranked9', {'shortlist_policy': 'ranked'}),
    ('coverage9', {'shortlist_policy': 'coverage'}),
    ('expanded_coverage9', {'candidate_scheme': 'expanded', 'shortlist_policy': 'coverage'}),
    ('expanded_coverage45', {'candidate_scheme': 'expanded', 'shortlist_policy': 'coverage', 'candidate_budget': 45}),
    ('verified_backup9', {'emergency_policy': 'verified_backup'}),
)


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    group_id: str
    suite: str
    scenario: str
    cfg: Stage2Config

    def record(self):
        return {'case_id': self.case_id, 'group_id': self.group_id,
                'suite': self.suite, 'scenario': self.scenario,
                'scenario_seed': self.cfg.seed, 'config': self.cfg.to_dict()}


def _case(suite, scenario, variant, cfg):
    cfg.validate()
    group = f'{suite}__{scenario}__{variant}'
    return ExperimentCase(f'{group}__seed{cfg.seed}', group, suite, scenario, cfg)


def predictor_cases(seed):
    """Frozen nine-second PPLNM with labelled CV tail versus common-anchor CA."""
    cfg = replace(Stage2Config(), seed=seed, max_time_s=150.,
                  prediction_history_dt_s=3.,
                  pplnm_checkpoint='artifacts_stage1/pplnm_150/pplnm.pt',
                  pplnm_calibration='artifacts_stage1/pplnm_150/pplnm_calibration.json',
                  pplnm_extension_mode='cv_after9', pplnm_threads=2)
    return [_case('predictor', scenario, f'{predictor}_history3_cv_after9',
                  replace(cfg, obstacle_predictor=predictor))
            for scenario in ('static3d', 'crossing3d', 'multi_crossing3d', 'maneuvering3d')
            for predictor in ('ca', 'pplnm')]


def build_cases(suite='smoke', seeds=(7,)):
    if suite not in SUITES:
        raise ValueError(f'unknown suite: {suite}')
    seeds = tuple(seeds)
    if (not seeds or len(set(seeds)) != len(seeds)
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds)):
        raise ValueError('seeds must be unique nonnegative integers')
    requested = SUITES[:-1] if suite == 'all' else (suite,)
    cases = []
    for family in requested:
        for seed in seeds:
            common = replace(Stage2Config(), seed=seed)
            if family == 'smoke':
                cases.extend(_case(family, scenario, 'retained_stage2', replace(common, max_time_s=120.))
                             for scenario in ('empty3d', 'static3d', 'crossing3d'))
            elif family == 'memory':
                base = replace(common, max_time_s=150., obstacle_speed_scale=1.5)
                for sensor_name, hfov, vfov, interval in SENSOR_VARIANTS:
                    sensor_cfg = replace(base, sensor_hfov_deg=hfov, sensor_vfov_deg=vfov,
                                         observation_period_s=interval)
                    for name, changes in MEMORY_VARIANTS:
                        cases.append(_case(family, 'multi_crossing3d', f'{name}__{sensor_name}',
                                           replace(sensor_cfg, **changes)))
            elif family == 'candidate':
                for scenario in ('static3d', 'multi_crossing3d'):
                    base = replace(common, max_time_s=150.,
                                   obstacle_speed_scale=1.5 if scenario == 'multi_crossing3d' else 1.)
                    for name, changes in CANDIDATE_VARIANTS:
                        cases.append(_case(family, scenario, name, replace(base, **changes)))
            else:
                cases.extend(predictor_cases(seed))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError('registered case IDs must be unique')
    return cases


METRIC_FIELDS = (
    'case_id', 'group_id', 'suite', 'scenario', 'scenario_seed', 'program_error',
    'status', 'navigation_success', 'collision', 'timeout',
    'actual_clearance_margin_violation', 'unsafe_emergency_forecast',
    'unsafe_emergency_forecast_count', 'avoidance_acceptance_passed',
    'minimum_actual_clearance_m', 'duration_s', 'goal_distance_m', 'path_length_m',
    'prediction_RMSE_by_horizon', 'prediction_RMSE_status',
    'actual_close_interactions', 'maximum_simultaneous_close_interactions',
    'planning_P95_ms', 'planning_max_ms', 'planning_deadline_miss',
    'planning_cycle_count', 'planning_deadline_miss_fraction',
    'emergency_count', 'candidate_budget', 'candidate_scheme', 'shortlist_policy',
    'emergency_policy', 'backup_full_rollout_count', 'safe_backup_event_count',
    'valid_candidates_min', 'valid_candidates_mean', 'valid_candidates_max',
    'obstacle_predictor', 'prediction_history_dt_s', 'memory_policy', 'track_max_age_s',
    'pplnm_extension_mode', 'pplnm_threads', 'calibrated_radii_applied',
    'first_failure_grid_safe_count', 'first_failure_shortlist_omission',
    'first_failure_diagnostic_time_s', 'offline_grid_wall_clock_s',
    'lost_object_count', 'track_hard_removed_count', 'track_conditional_removed_count',
    'track_reacquired_count', 'memory_maximum_extra_uncertainty_m',
    'memory_retained_beyond_expiry_plan_count', 'maximum_observed_obstacles',
    'maximum_tracked_obstacles', 'mean_active_tracks', 'mean_extra_uncertainty_m',
    'sensor_hfov_deg', 'sensor_vfov_deg', 'observation_period_s', 'observation_delay_s',
    'terrain_guidance', 'physics_dt_s', 'control_dt_s', 'horizon_s',
    'command_delay_s', 'all_states_finite', 'execution_prediction_rmse_m',
    'worker_wall_clock_s', 'workers', 'timing_interpretation', 'summary_path', 'log_path',
)


def metric_row(case, record, workers):
    summary = record.get('summary', {})
    row = {name: None for name in METRIC_FIELDS}
    failed = bool(record.get('program_error'))
    status = summary.get('status', 'program_error' if failed else 'missing_summary')
    count = summary.get('planning_exceeds_control_period_count')
    cycles = record.get('planning_cycle_count')
    cfg = case.cfg
    row.update(case_id=case.case_id, group_id=case.group_id, suite=case.suite,
               scenario=case.scenario, scenario_seed=cfg.seed, program_error=failed,
               status=status, navigation_success=summary.get('reached_goal'),
               collision=('collision' in status) if summary else None,
               timeout=(status == 'timeout') if summary else None,
               actual_clearance_margin_violation=(not summary['actual_safety_margin_met']
                                                  if 'actual_safety_margin_met' in summary else None),
               unsafe_emergency_forecast=(summary.get('unsafe_emergency_forecast_count', 0) > 0
                                           if summary else None),
               unsafe_emergency_forecast_count=summary.get('unsafe_emergency_forecast_count'),
               avoidance_acceptance_passed=summary.get('avoidance_acceptance_passed'),
               minimum_actual_clearance_m=summary.get('minimum_model_clearance_m'),
               duration_s=summary.get('duration_s'), goal_distance_m=summary.get('goal_distance_m'),
               path_length_m=summary.get('path_length_m'),
               prediction_RMSE_by_horizon='NaN',
               prediction_RMSE_status='pending_offline_same_absolute_anchor_evaluation',
               actual_close_interactions=summary.get('actual_close_interaction_count'),
               maximum_simultaneous_close_interactions=summary.get('maximum_simultaneous_close_interactions'),
               planning_P95_ms=summary.get('planning_ms_p95'), planning_max_ms=summary.get('planning_ms_max'),
               planning_deadline_miss=count, planning_cycle_count=cycles,
               planning_deadline_miss_fraction=(count/cycles if count is not None and cycles else None),
               emergency_count=summary.get('emergency_count'),
               candidate_budget=cfg.candidate_budget, candidate_scheme=cfg.candidate_scheme,
               shortlist_policy=cfg.shortlist_policy, emergency_policy=cfg.emergency_policy,
               obstacle_predictor=cfg.obstacle_predictor, prediction_history_dt_s=cfg.prediction_history_dt_s,
               pplnm_extension_mode=cfg.pplnm_extension_mode if cfg.obstacle_predictor == 'pplnm' else 'not_applicable',
               pplnm_threads=cfg.pplnm_threads if cfg.obstacle_predictor == 'pplnm' else None,
               calibrated_radii_applied=record.get('calibrated_radii_applied'),
               first_failure_grid_safe_count=record.get('candidate_first_failure', {}).get('full_grid_safe_count'),
               first_failure_shortlist_omission=record.get('candidate_first_failure', {}).get('configured_shortlist_omitted_safe_reference'),
               first_failure_diagnostic_time_s=record.get('candidate_first_failure', {}).get('time_s'),
               offline_grid_wall_clock_s=record.get('candidate_first_failure', {}).get('offline_wall_clock_s'),
               memory_policy=cfg.memory_policy, track_max_age_s=cfg.track_max_age_s,
               sensor_hfov_deg=cfg.sensor_hfov_deg, sensor_vfov_deg=cfg.sensor_vfov_deg,
               observation_period_s=cfg.observation_period_s, observation_delay_s=cfg.observation_delay_s,
               terrain_guidance=cfg.terrain_guidance, physics_dt_s=cfg.physics_dt,
               control_dt_s=cfg.control_dt, horizon_s=cfg.horizon_s, command_delay_s=cfg.command_delay_s,
               execution_prediction_rmse_m=summary.get('prediction_position_rmse_m'),
               worker_wall_clock_s=record.get('worker_wall_clock_s'), workers=workers,
               timing_interpretation=('measured wall-clock; concurrent experiments share CPU; '
                                      'synchronous offline physical execution; latency is not physical delay'
                                      if workers > 1 else 'measured wall-clock; other system processes may share CPU; '
                                      'synchronous offline physical execution; latency is not physical delay'),
               summary_path=record.get('summary_path'), log_path=record.get('log_path'),
               mean_active_tracks=record.get('mean_active_tracks'),
               mean_extra_uncertainty_m=record.get('mean_extra_uncertainty_m'))
    for name in ('backup_full_rollout_count', 'safe_backup_event_count', 'valid_candidates_min',
                 'valid_candidates_mean', 'valid_candidates_max', 'lost_object_count',
                 'track_hard_removed_count', 'track_conditional_removed_count', 'track_reacquired_count',
                 'memory_maximum_extra_uncertainty_m', 'memory_retained_beyond_expiry_plan_count',
                 'maximum_observed_obstacles', 'maximum_tracked_obstacles', 'all_states_finite'):
        row[name] = summary.get(name)
    return row


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def _execute_case(case, output, plot=True, animate=False):
    """Each process owns one fresh scene, plant, predictor and result directory."""
    out = Path(output)
    case_out, logfile = out/'cases'/case.case_id, out/'logs'/f'{case.case_id}.log'
    record = {'case_id': case.case_id, 'program_error': False,
              'summary_path': str(case_out/'summary.json'), 'log_path': str(logfile)}
    tic = time.perf_counter()
    with (logfile.open('x', encoding='utf-8') as stream, redirect_stdout(stream),
          redirect_stderr(stream), chdir(ROOT)):
        try:
            from auv3d.stage2_simulation import run_remus_episode, save_remus_episode
            print(json.dumps(case.record(), ensure_ascii=False), flush=True)
            observer = None
            if case.suite == 'candidate':
                from scripts_stage3.candidate_diagnostics import FirstFailureCapture
                observer = FirstFailureCapture(case.cfg)
            result = run_remus_episode(case.cfg, case.scenario, on_plan=observer)
            save_remus_episode(result, case_out, animate=animate, plot=plot)
            memory = result.get('memory_diagnostics', [])
            forecasts = [forecast for plan in result.get('planning_diagnostics', [])
                         for forecast in plan.get('forecast_records', [])]
            record.update(summary=result['summary'], planning_cycle_count=len(result['planning']),
                          calibrated_radii_applied=(any(forecast['calibrated_radii_applied'] for forecast in forecasts)
                                                    if forecasts else None),
                          mean_active_tracks=(sum(m['active_track_count'] for m in memory)/len(memory)
                                              if memory else None),
                          mean_extra_uncertainty_m=(sum(m['mean_uncertainty_m'] for m in memory)/len(memory)
                                                    if memory else None))
            if observer is not None:
                record['candidate_first_failure'] = observer.evaluate(case_out)
            print(json.dumps(result['summary'], ensure_ascii=False), flush=True)
        except Exception as error:
            # A research navigation failure is a normal summary; exceptions are
            # separate program failures and are never converted to safe success.
            record.update(program_error=True, error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc())
            print(record['traceback'], flush=True)
            case_out.mkdir(parents=True, exist_ok=True)
            _write_json(case_out/'program_error.json', record)
    record['worker_wall_clock_s'] = time.perf_counter()-tic
    return record


def grouped_summary(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['group_id'], []).append(row)
    result = []
    for group, members in sorted(groups.items()):
        valid = [row for row in members if not row['program_error'] and row['status'] != 'missing_summary']
        result.append({'group_id': group, 'case_count': len(members),
                       'completed_cases': len(valid), 'program_errors': len(members)-len(valid),
                       'scenario_seeds': [row['scenario_seed'] for row in members],
                       'navigation_success_count': sum(row['navigation_success'] is True for row in valid),
                       'navigation_success_rate': (sum(row['navigation_success'] is True for row in valid)/len(valid)
                                                   if valid else None),
                       'finite_horizon_acceptance_count': sum(row['avoidance_acceptance_passed'] is True for row in valid),
                       'finite_horizon_acceptance_rate': (sum(row['avoidance_acceptance_passed'] is True for row in valid)/len(valid)
                                                          if valid else None),
                       'collision_count': sum(row['collision'] is True for row in valid),
                       'timeout_count': sum(row['timeout'] is True for row in valid),
                       'grouping_scope': 'identical registered settings; only scenario seed can vary'})
    return result


def _persist_progress(out, cases, records, workers):
    completed = {record['case_id']: record for record in records}
    rows = [metric_row(case, completed[case.case_id], workers) for case in cases if case.case_id in completed]
    with (out/'EXPERIMENT_SUMMARY.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    _write_json(out/'EXPERIMENT_RECORDS.json', records)
    _write_json(out/'GROUPED_SUMMARY.json', grouped_summary(rows))
    registration = json.loads((out/'EXPERIMENT_CONFIG.json').read_text(encoding='utf-8'))
    progress = {'registered_case_count': registration['registered_case_count'],
                'selected_case_count': len(cases), 'completed_case_count': len(rows),
                'program_error_count': sum(row['program_error'] for row in rows),
                'navigation_failures_preserved': True, 'workers': workers,
                'mixed_overall_navigation_success_rate': None,
                'mixed_overall_rate_reason': 'different configurations are reported separately',
                'prediction_RMSE_status': 'pending offline same-anchor evaluation; NaN CSV cells are not measured zeros'}
    _write_json(out/'study_progress.json', progress)
    return progress


def _validate_registration(cases):
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError('registered case IDs must be unique')
    baseline = Stage2Config()
    groups = {}
    for case in cases:
        case.cfg.validate()
        for name in ('safety_margin_m', 'body_radius_m', 'goal_tolerance_m'):
            if getattr(case.cfg, name) != getattr(baseline, name):
                raise ValueError(f'the common stage3 study must retain original {name}')
        settings = case.cfg.to_dict()
        settings.pop('seed')
        signature = (case.suite, case.scenario, settings)
        if case.group_id in groups and groups[case.group_id] != signature:
            raise ValueError(f'group {case.group_id} mixes configurations other than scenario seed')
        groups[case.group_id] = signature


def run_study(output, suite='smoke', workers=1, limit=None, seeds=(7,), plot=True, animate=False,
              cases=None):
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError('workers must be a positive integer')
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError('limit must be a positive integer')
    registered = list(cases) if cases is not None else build_cases(suite, seeds)
    _validate_registration(registered)
    selected = registered[:limit] if limit is not None else registered
    if not selected:
        raise ValueError('study must contain at least one registered case')
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; existing experiments are preserved')
    out.mkdir(parents=True, exist_ok=True)
    for dirname in ('cases', 'logs', 'registered_configs'):
        (out/dirname).mkdir()
    physics_steps = sorted({case.cfg.physics_dt for case in registered})
    registration = {
        'registered_at': datetime.now().astimezone().isoformat(), 'suite': suite,
        'invocation': sys.argv, 'working_directory': str(ROOT),
        'invocation_working_directory': str(Path.cwd()), 'python': sys.executable,
        'workers': workers, 'requested_limit': limit,
        'requested_seeds': list(seeds), 'seeds': sorted({case.cfg.seed for case in registered}),
        'registered_case_count': len(registered), 'selected_case_count': len(selected),
        'selected_case_ids': [case.case_id for case in selected],
        'common_physics_dt_s': physics_steps[0] if len(physics_steps) == 1 else None,
        'physics_dt_values_s': physics_steps, 'common_safety_margin_m': 1.,
        'original_REMUS_parameters_modified': False, 'scene_geometry_modified_between_methods': False,
        'time_semantics': 'synchronous offline simulation; measured planning time is not physical delay',
        'timing_resource_scope': 'parallel cases share CPU' if workers > 1 else 'single experiment worker; no dedicated machine isolation',
        'prediction_RMSE_status': 'pending independent offline same-anchor evaluation',
        'predictor_pair_status': 'CA versus frozen PPLNM9 + explicitly uncalibrated cv_after9; common 3 s history',
        'predictor_circle_comparison': 'same heuristic enclosing circle; P25 calibrated balls exposed but not applied',
        'candidate_diagnostic_rule': 'candidate suite only: capture first feasible=False call, full production grid offline after execution, excludes planningP95',
        'registered_cases': [case.record() for case in registered],
    }
    # All configurations exist before the first simulation starts.
    _write_json(out/'EXPERIMENT_CONFIG.json', registration)
    for case in registered:
        _write_json(out/'registered_configs'/f'{case.case_id}.json', case.record())
    records = []
    _persist_progress(out, selected, records, workers)
    started = time.perf_counter()
    if workers == 1:
        for case in selected:
            print(f'START {case.case_id}', flush=True)
            record = _execute_case(case, out, plot, animate)
            records.append(record)
            _persist_progress(out, selected, records, workers)
            print(f'DONE {case.case_id}: program_error={record["program_error"]}, '
                  f'status={record.get("summary", {}).get("status")}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_execute_case, case, out, plot, animate): case for case in selected}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = {'case_id': case.case_id, 'program_error': True,
                              'error_type': type(error).__name__, 'error': str(error),
                              'traceback': traceback.format_exc()}
                    case_out = out/'cases'/case.case_id
                    case_out.mkdir(parents=True, exist_ok=True)
                    _write_json(case_out/'program_error.json', record)
                records.append(record)
                _persist_progress(out, selected, records, workers)
                print(f'DONE {case.case_id}: program_error={record["program_error"]}, '
                      f'status={record.get("summary", {}).get("status")}', flush=True)
    final = _persist_progress(out, selected, records, workers)
    final.update(completed_at=datetime.now().astimezone().isoformat(),
                 wall_clock_s=time.perf_counter()-started,
                 all_selected_cases_completed=len(records) == len(selected))
    _write_json(out/'study_summary.json', final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, default='smoke')
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seeds', nargs='+', type=int, default=[7])
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--animate', action='store_true')
    args = parser.parse_args(argv)
    os.environ['PYTHONUTF8'] = '1'
    try:
        result = run_study(args.output, args.suite, args.workers, args.limit, args.seeds,
                           plot=not args.no_plot, animate=args.animate)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if result['program_error_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
