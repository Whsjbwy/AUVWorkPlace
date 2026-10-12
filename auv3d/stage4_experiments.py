"""Paired REMUS risk experiments with separate execution and safety outcomes.

The experiment driver never supplies obstacle truth to a risk policy. Raw
episodes, including navigation failures and program errors, are preserved in
fresh output directories. CPU contention is recorded alongside wall latency.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import chdir, redirect_stderr, redirect_stdout
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('B0', 'B1', 'B2')
SPLITS = ('train', 'tune', 'calibration', 'test')
METRIC_FIELDS = (
    'case_id', 'split', 'family', 'scenario', 'scenario_seed', 'risk_method',
    'risk_alpha', 'program_error', 'status', 'navigation_success', 'collision',
    'timeout', 'actual_clearance_margin_violation', 'unsafe_emergency_forecast',
    'unsafe_emergency_forecast_count', 'avoidance_acceptance_passed',
    'minimum_actual_clearance_m', 'duration_s', 'goal_distance_m', 'path_length_m',
    'no_feasible_candidate_count', 'no_feasible_candidate_scope', 'emergency_count', 'valid_candidates_min',
    'valid_candidates_mean', 'valid_candidates_max', 'candidate_false_accept_count',
    'candidate_false_reject_count', 'candidate_misclassification_scope',
    'calibration_coverage', 'calibration_coverage_count', 'calibration_label_count',
    'calibration_coverage_scope', 'prediction_RMSE_by_horizon',
    'prediction_RMSE_status', 'actual_close_interactions',
    'maximum_simultaneous_close_interactions', 'maximum_observed_obstacles',
    'maximum_tracked_obstacles', 'lost_object_count', 'planning_P95_ms',
    'planning_max_ms', 'planning_deadline_miss', 'planning_cycle_count',
    'planning_deadline_miss_fraction', 'execution_prediction_rmse_m',
    'candidate_budget', 'candidate_scheme', 'shortlist_policy', 'emergency_policy',
    'obstacle_predictor', 'terrain_guidance', 'physics_dt_s', 'control_dt_s',
    'horizon_s', 'command_delay_s', 'all_states_finite', 'workers',
    'worker_wall_clock_s', 'timing_interpretation', 'summary_path', 'log_path',
)


def _json_value(value):
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError('experiment records cannot contain nonfinite values')
    return value


def write_json(path, value, *, exclusive=True):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(_json_value(value), stream, indent=2, ensure_ascii=False,
                  allow_nan=False)


def write_csv(path, rows, fields=METRIC_FIELDS):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def _validate_case(case):
    if not isinstance(case, dict):
        raise ValueError('registered cases must be dictionaries')
    for field in ('case_id', 'split', 'family', 'scenario', 'seed'):
        if field not in case:
            raise ValueError(f'registered case is missing {field}')
    if (not isinstance(case['case_id'], str)
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', case['case_id'])
            or case['case_id'] in ('.', '..')):
        raise ValueError('case_id must be a safe single path component')
    if case['split'] not in SPLITS:
        raise ValueError('unknown data split')
    if (not isinstance(case['seed'], int) or isinstance(case['seed'], bool)
            or case['seed'] < 0):
        raise ValueError('scenario seed must be a nonnegative integer')


@dataclass(frozen=True)
class ExperimentJob:
    case: dict
    method: str = 'B0'
    alpha: float | None = None
    calibration_path: str | None = None

    @property
    def job_id(self):
        if self.method == 'B0':
            return self.case['case_id']
        level = '' if self.alpha is None else f'__alpha{self.alpha:g}'
        return f"{self.case['case_id']}__{self.method}{level}"

    def validate(self):
        _validate_case(self.case)
        if self.method not in METHODS:
            raise ValueError('risk method must be B0, B1 or B2')
        if self.method == 'B0':
            if self.alpha is not None or self.calibration_path is not None:
                raise ValueError('B0 has no calibrated alpha or artifact')
        elif (self.alpha is None or isinstance(self.alpha, bool)
              or not np.isfinite(self.alpha) or not 0 < self.alpha < 1):
            raise ValueError('calibrated risk methods require alpha in (0, 1)')
        elif self.calibration_path is None:
            raise ValueError('calibrated risk methods require a calibration artifact')


def select_cases(cases, *, splits=None, case_ids=None, limit=None):
    cases = list(cases)
    for case in cases:
        _validate_case(case)
    if len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('registered case IDs must be unique')
    splits = tuple(splits or SPLITS)
    if not splits or any(split not in SPLITS for split in splits):
        raise ValueError('select valid nonempty data splits')
    case_ids = set(case_ids or ())
    unknown = case_ids - {case['case_id'] for case in cases}
    if unknown:
        raise ValueError(f'unknown case IDs: {sorted(unknown)}')
    selected = [case for case in cases if case['split'] in splits
                and (not case_ids or case['case_id'] in case_ids)]
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError('limit must be a positive integer')
        selected = selected[:limit]
    if not selected:
        raise ValueError('no cases match the requested selection')
    return selected


def episode_metrics(job, record, workers):
    """Summarize measured outcomes without converting arrival to safe success."""
    from .stage4_protocol import config_for_case
    cfg = config_for_case(job.case)
    summary = record.get('summary', {})
    error = bool(record.get('program_error', False))
    status = summary.get('status', 'program_error' if error else 'missing_summary')
    cycles = record.get('planning_cycle_count')
    misses = summary.get('planning_exceeds_control_period_count')
    row = dict.fromkeys(METRIC_FIELDS)
    row.update(
        case_id=job.case['case_id'], split=job.case['split'], family=job.case['family'],
        scenario=job.case['scenario'], scenario_seed=cfg.seed,
        risk_method=job.method, risk_alpha=job.alpha, program_error=error, status=status,
        navigation_success=summary.get('reached_goal'),
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
        no_feasible_candidate_count=record.get('no_feasible_candidate_count'),
        no_feasible_candidate_scope='configured full-REMUS-verified shortlist only; not proof that the complete reference grid is infeasible',
        candidate_false_accept_count=record.get('candidate_false_accept_count'),
        candidate_false_reject_count=record.get('candidate_false_reject_count'),
        candidate_misclassification_scope=record.get('candidate_misclassification_scope',
            'not evaluated; no absent counterfactual labels are imputed'),
        calibration_coverage=record.get('calibration_coverage'),
        calibration_coverage_count=record.get('calibration_coverage_count'),
        calibration_label_count=record.get('calibration_label_count'),
        calibration_coverage_scope=record.get('calibration_coverage_scope',
            'not evaluated; coverage is separate from navigation and collision'),
        prediction_RMSE_by_horizon=record.get('prediction_RMSE_by_horizon'),
        prediction_RMSE_status=record.get('prediction_RMSE_status',
            'not evaluated; execution prediction error is a different metric'),
        actual_close_interactions=summary.get('actual_close_interaction_count'),
        planning_P95_ms=summary.get('planning_ms_p95'),
        planning_max_ms=summary.get('planning_ms_max'), planning_deadline_miss=misses,
        planning_cycle_count=cycles,
        planning_deadline_miss_fraction=(misses/cycles if misses is not None and cycles else None),
        execution_prediction_rmse_m=summary.get('prediction_position_rmse_m'),
        candidate_budget=cfg.candidate_budget, candidate_scheme=cfg.candidate_scheme,
        shortlist_policy=cfg.shortlist_policy, emergency_policy=cfg.emergency_policy,
        obstacle_predictor=cfg.obstacle_predictor, terrain_guidance=cfg.terrain_guidance,
        physics_dt_s=cfg.physics_dt, control_dt_s=cfg.control_dt, horizon_s=cfg.horizon_s,
        command_delay_s=cfg.command_delay_s, workers=workers,
        worker_wall_clock_s=record.get('worker_wall_clock_s'),
        timing_interpretation=('wall-clock latency with shared CPU workers; synchronous offline '
                               'physics pauses during planning; latency is not execution delay'
                               if workers > 1 else 'wall-clock latency; synchronous offline '
                               'physics pauses during planning; latency is not execution delay'),
        summary_path=record.get('summary_path'), log_path=record.get('log_path'))
    for name in ('emergency_count', 'valid_candidates_min', 'valid_candidates_mean',
                 'valid_candidates_max', 'maximum_simultaneous_close_interactions',
                 'maximum_observed_obstacles', 'maximum_tracked_obstacles',
                 'lost_object_count', 'all_states_finite'):
        row[name] = summary.get(name)
    return row


def _make_risk_policy(job):
    if job.method == 'B0':
        return None
    # Kept lazy so baseline data collection is independent of calibration code.
    from .stage4_risk import RiskPolicy
    return RiskPolicy(job.method, library_path=job.calibration_path, alpha=job.alpha)


def _execute_job(job, output, plot=True, animate=False):
    from .stage4_protocol import config_for_case
    cfg = config_for_case(job.case)
    out = Path(output)
    case_out = out/'cases'/job.job_id
    logfile = out/'logs'/f'{job.job_id}.log'
    record = {'job_id': job.job_id, 'program_error': False,
              'summary_path': str(case_out/'summary.json'), 'log_path': str(logfile)}
    tic = time.perf_counter()
    with (logfile.open('x', encoding='utf-8') as stream,
          redirect_stdout(stream), redirect_stderr(stream), chdir(ROOT)):
        try:
            from .stage2_simulation import run_remus_episode, save_remus_episode
            print(json.dumps({'case': job.case, 'method': job.method,
                              'alpha': job.alpha, 'config': cfg.to_dict()}, ensure_ascii=False),
                  flush=True)
            policy = _make_risk_policy(job)
            kwargs = {} if policy is None else {'risk_policy': policy}
            result = run_remus_episode(cfg, job.case['scenario'], **kwargs)
            save_remus_episode(result, case_out, animate=animate, plot=plot)
            details = result.get('planning_diagnostics', [])
            record.update(summary=result['summary'], planning_cycle_count=len(result['planning']),
                          no_feasible_candidate_count=sum(not row['feasible'] for row in details))
            # Optional post-execution diagnostics are a separate channel; they
            # cannot feed future truth back into an already executed decision.
            for key, value in result.get('offline_risk_metrics', {}).items():
                record[key] = value
            if 'risk_diagnostics' in result:
                write_json(case_out/'risk_diagnostics.json', result['risk_diagnostics'])
            write_json(case_out/'risk_experiment.json', {
                'case_id': job.case['case_id'], 'split': job.case['split'],
                'method': job.method, 'alpha': job.alpha,
                'calibration_library': job.calibration_path,
                'risk_metadata': result['summary'].get('risk_metadata', {}),
                'online_obstacle_truth_supplied': False})
            print(json.dumps(result['summary'], ensure_ascii=False), flush=True)
        except Exception as error:
            record.update(program_error=True, error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc())
            print(record['traceback'], flush=True)
            case_out.mkdir(parents=True, exist_ok=True)
            write_json(case_out/'program_error.json', record)
    record['worker_wall_clock_s'] = time.perf_counter()-tic
    return record


def load_baseline_record(case, baseline_root):
    """Reuse completed B0 evidence read-only; check actual config fields."""
    from .stage4_protocol import config_for_case
    from .stage2_config import Stage2Config
    root = Path(baseline_root).resolve()
    source = root/'cases'/case['case_id']
    expected = config_for_case(case).to_dict()
    required = ('summary.json', 'config.json', 'planning.csv', 'states.csv',
                'planning_diagnostics.json', 'observations.json', 'obstacles.json')
    if (source/'program_error.json').is_file():
        raise ValueError(f'B0 reference has a recorded program error; resolve it before paired reuse: {source}')
    if not all((source/name).is_file() for name in required):
        raise FileNotFoundError(f'completed raw B0 evidence is required: {source}')
    cfg = Stage2Config.load(source/'config.json').to_dict()
    if cfg != expected:
        changed = sorted(key for key in set(cfg) | set(expected)
                         if cfg.get(key) != expected.get(key))
        raise ValueError(f'paired B0 configuration differs in fields: {changed}')
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if (summary.get('backend') != 'remus6dof'
            or summary.get('risk_method', 'B0') != 'B0'
            or summary.get('seed') != case['seed']
            or summary.get('scenario') != case['scenario']):
        raise ValueError('paired reference must be the matching actual REMUS B0 episode')
    details = json.loads((source/'planning_diagnostics.json').read_text(encoding='utf-8'))
    return {'job_id': case['case_id'], 'program_error': False, 'summary': summary,
            'summary_path': str(source/'summary.json'),
            'log_path': str(root/'logs'/f"{case['case_id']}.log"),
            'planning_cycle_count': len(details),
            'no_feasible_candidate_count': sum(not row['feasible'] for row in details),
            'worker_wall_clock_s': summary.get('wall_clock_s'),
            'reused_read_only_B0': True}


def grouped_results(rows):
    """Keep risk levels, splits and scenario families separate."""
    groups = {}
    for row in rows:
        key = (row['split'], row['family'], row['risk_method'], row['risk_alpha'])
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda pair: str(pair[0])):
        measured = [row for row in group if not row['program_error']
                    and row['navigation_success'] is not None]
        item = dict(zip(('split', 'family', 'risk_method', 'risk_alpha'), key))
        item.update(registered_runs=len(group), measured_runs=len(measured),
                    program_error_count=sum(row['program_error'] for row in group))
        for field in ('navigation_success', 'collision', 'timeout',
                      'actual_clearance_margin_violation', 'unsafe_emergency_forecast',
                      'avoidance_acceptance_passed'):
            values = [row[field] for row in measured if row[field] is not None]
            item[field+'_rate'] = float(np.mean(values)) if values else None
            item[field+'_count'] = sum(bool(value) for value in values)
        for field in ('minimum_actual_clearance_m', 'duration_s', 'path_length_m',
                      'planning_P95_ms', 'emergency_count', 'no_feasible_candidate_count'):
            values = [row[field] for row in measured if row[field] is not None]
            item[field+'_mean'] = float(np.mean(values)) if values else None
        item['denominator_note'] = ('Program errors are reported separately; navigation failures '
                                    'remain in measured rates. This group is not pooled with '
                                    'other families, splits or risk levels.')
        result.append(item)
    return result


def run_study(output, *, cases=None, splits=None, case_ids=None, limit=None,
              methods=('B0',), alphas=(0.1,), calibration_path=None,
              workers=1, plot=True, animate=False, baseline_root=None):
    from .stage4_protocol import manifest, protocol_record, config_for_case
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError('workers must be a positive integer')
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'existing experiments are preserved; choose a new output: {out}')
    registered = list(manifest() if cases is None else cases)
    selected = select_cases(registered, splits=splits, case_ids=case_ids, limit=limit)
    # Complete independent calibration trajectories first; scenarios and
    # parameters are unchanged by execution ordering.
    selected.sort(key=lambda row: ('calibration', 'train', 'tune', 'test').index(row['split']))
    methods, alphas = tuple(methods), tuple(alphas)
    if not methods or len(set(methods)) != len(methods) or any(method not in METHODS for method in methods):
        raise ValueError('methods must be unique B0, B1 or B2 names')
    if any(method != 'B0' for method in methods):
        if not alphas or len(set(alphas)) != len(alphas):
            raise ValueError('calibrated methods require unique alpha levels')
        if calibration_path is None or not Path(calibration_path).is_file():
            raise FileNotFoundError('calibrated methods require an existing calibration artifact')
        calibration_path = str(Path(calibration_path).resolve())
    jobs = [ExperimentJob(case, method, alpha, calibration_path if method != 'B0' else None)
            for case in selected for method in methods
            for alpha in ((None,) if method == 'B0' else alphas)]
    for job in jobs:
        job.validate()
    baseline_records = ([load_baseline_record(case, baseline_root) for case in selected]
                        if baseline_root is not None else [])
    if baseline_records and 'B0' in methods:
        raise ValueError('read-only B0 reuse cannot also request new B0 execution')
    out.mkdir(parents=True, exist_ok=True)
    for folder in ('cases', 'logs', 'registered_configs'):
        (out/folder).mkdir()
    for case in registered:
        write_json(out/'registered_configs'/f"{case['case_id']}.json", config_for_case(case).to_dict())
    write_json(out/'EXPERIMENT_CONFIG.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol': protocol_record(),
        'registered_case_count': len(registered), 'selected_case_count': len(selected),
        'selected_job_count': len(jobs), 'registered_cases': registered,
        'jobs': [{'case_id': job.case['case_id'], 'method': job.method, 'alpha': job.alpha,
                  'calibration_path': job.calibration_path} for job in jobs],
        'workers': workers, 'plot': plot, 'animate': animate,
        'baseline_reference_root': str(Path(baseline_root).resolve()) if baseline_root else None,
        'baseline_reuse_case_count': len(baseline_records),
        'common_physics_dt_s': .02, 'common_safety_margin_m': 1.,
        'common_goal_tolerance_m': 5.,
        'timing_note': 'Planning wall time is measured; physical time pauses during synchronous planning.'})
    records_by_id = {}
    if workers == 1:
        for job in jobs:
            record = _execute_job(job, out, plot, animate)
            records_by_id[job.job_id] = record
            print(f"{job.job_id}: {record.get('summary', {}).get('status', 'PROGRAM_ERROR')}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_execute_job, job, out, plot, animate): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                # Worker process crashes remain program errors, never research failures.
                try:
                    record = future.result()
                except Exception as error:
                    record = {'job_id': job.job_id, 'program_error': True,
                              'error_type': type(error).__name__, 'error': str(error),
                              'traceback': traceback.format_exc()}
                records_by_id[job.job_id] = record
                print(f"{job.job_id}: {record.get('summary', {}).get('status', 'PROGRAM_ERROR')}", flush=True)
    records = [records_by_id[job.job_id] for job in jobs]
    rows = [episode_metrics(job, record, workers) for job, record in zip(jobs, records)]
    if baseline_records:
        registration = json.loads((Path(baseline_root)/'EXPERIMENT_CONFIG.json').read_text(encoding='utf-8'))
        baseline_workers = registration['workers']
        rows += [episode_metrics(ExperimentJob(case), record, baseline_workers)
                 for case, record in zip(selected, baseline_records)]
        write_json(out/'REUSED_B0_RECORDS.json', baseline_records)
    write_json(out/'EXPERIMENT_RECORDS.json', records)
    write_csv(out/'EXPERIMENT_SUMMARY.csv', rows)
    groups = grouped_results(rows)
    write_json(out/'GROUPED_SUMMARY.json', groups)
    curve_fields = list(dict.fromkeys(field for row in groups for field in row))
    write_csv(out/'SAFETY_EFFICIENCY_CURVES.csv', groups, curve_fields)
    outcome = {'registered_case_count': len(registered), 'selected_case_count': len(selected),
               'executed_job_count': len(records),
               'read_only_reused_B0_count': len(baseline_records),
               'program_error_count': sum(record['program_error'] for record in records),
               'mixed_overall_success_rate': None,
               'output_directory': str(out),
               'outcome_note': 'Read split/family/method/alpha groups; program completion is not safe navigation.'}
    write_json(out/'STUDY_SUMMARY.json', outcome)
    return outcome
