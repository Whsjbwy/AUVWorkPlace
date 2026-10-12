"""Registered, paired Stage 5 experiments without outcome imputation.

The driver passes causal forecasts and an explicitly registered scene to the
original REMUS executor. Calibration artifacts and fixed-margin tuning records
are separate for each predictor and observation-noise level.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import chdir, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
import traceback

from .stage4_experiments import write_json, write_csv

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('A0', 'A1', 'A2', 'A3', 'A4', 'A5')
PREDICTORS = ('ca', 'cv', 'filtered_cv')
SPLITS = ('dev', 'tune', 'calibration', 'test')
PHASES = ('pilot', 'formal')
COMBINATIONS = ('add', 'replace')
DELAY_MODES = ('none', 'fixed', 'measured')
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS')
METRIC_FIELDS = (
    'case_id', 'scene_id', 'phase', 'split', 'family', 'scenario', 'scenario_seed',
    'method', 'predictor', 'noise_std_m', 'risk_alpha', 'risk_combination',
    'fixed_margin_m', 'calibration_path', 'tuning_path', 'planning_delay_mode',
    'planning_delay_s', 'program_error', 'status', 'navigation_success', 'safe_arrival', 'collision',
    'timeout', 'actual_clearance_margin_violation', 'minimum_actual_clearance_m',
    'unsafe_emergency_forecast', 'unsafe_emergency_forecast_count',
    'avoidance_acceptance_passed', 'duration_s', 'path_length_m', 'goal_distance_m',
    'no_feasible_candidate_count', 'no_feasible_candidate_fraction', 'emergency_count', 'planning_cycle_count',
    'verified_candidate_count', 'feasible_candidate_count', 'feasible_fraction',
    'feasible_fraction_scope', 'actual_close_interactions', 'maximum_observed_obstacles',
    'maximum_tracked_obstacles', 'lost_object_count', 'planning_P95_ms',
    'planning_max_ms', 'planning_deadline_miss', 'planning_deadline_miss_fraction',
    'physical_planning_delay_mean_s', 'physical_planning_delay_max_s',
    'candidate_false_accept_count', 'candidate_false_reject_count',
    'evaluated_selected_reference_anchor_count', 'candidate_misclassification_scope',
    'calibration_coverage', 'calibration_coverage_count', 'calibration_label_count',
    'calibration_coverage_scope', 'calibration_coverage_by_horizon', 'prediction_RMSE_by_horizon', 'prediction_RMSE_scope',
    'execution_prediction_rmse_m', 'all_states_finite', 'candidate_budget',
    'physics_dt_s', 'control_dt_s', 'horizon_s', 'transport_delay_s', 'workers',
    'worker_wall_clock_s', 'timing_interpretation', 'summary_path', 'log_path',
)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _positive_level(value, name, *, zero=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (not zero and value == 0)):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


def configure_threads():
    """Call before importing numerical runtimes in a new CLI process."""
    for variable in THREAD_VARIABLES:
        os.environ[variable] = '1'


def select_cases(cases, *, splits=None, case_ids=None, noise_std=None, limit=None):
    cases = list(cases)
    for case in cases:
        for key in ('case_id', 'scene_id', 'phase', 'split', 'family', 'scenario', 'seed'):
            if key not in case:
                raise ValueError(f'case is missing {key}')
        if (not isinstance(case['case_id'], str)
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', case['case_id'])
                or case['case_id'] in ('.', '..')):
            raise ValueError('case_id must be a safe single path component')
        if case['split'] not in SPLITS or case['phase'] not in PHASES:
            raise ValueError('unregistered split or phase')
    if len({row['case_id'] for row in cases}) != len(cases):
        raise ValueError('case IDs must be unique')
    splits = tuple(SPLITS if splits is None else splits)
    if not splits or any(value not in SPLITS for value in splits):
        raise ValueError('select valid nonempty splits')
    ids = set(case_ids or ())
    if ids - {row['case_id'] for row in cases}:
        raise ValueError('unknown case IDs requested')
    if noise_std is not None:
        _positive_level(noise_std, 'noise_std', zero=True)
    chosen = [row for row in cases if row['split'] in splits
              and (not ids or row['case_id'] in ids)
              and (noise_std is None or math.isclose(
                  row['config_overrides']['sensor_noise_std_m'], noise_std,
                  abs_tol=1e-12, rel_tol=0))]
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('limit must be a positive integer')
        chosen = chosen[:limit]
    if not chosen:
        raise ValueError('no cases match the requested selection')
    return chosen


def artifact_path(paths, predictor, noise, *, phase=None):
    """Resolve an explicit predictor/noise key; no fallback to another pool."""
    paths = paths or {}
    key = (predictor, float(noise))
    if key not in paths:
        raise FileNotFoundError(f'no independent calibration for predictor={predictor}, noise={noise:g}')
    path = Path(paths[key]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'calibration artifact is missing: {path}')
    if phase is not None:
        import numpy as np
        with np.load(path,allow_pickle=False) as library:
            metadata = json.loads(str(library['metadata_json'].item()))
        if (metadata.get('schema') != 'stage5-risk-v1'
                or metadata.get('split') != 'calibration'
                or metadata.get('role') != 'calibration_only'
                or metadata.get('phase') != phase
                or metadata.get('predictor') != predictor
                or metadata.get('noise_std_m') != noise):
            raise ValueError('calibration pool differs in actual phase, split, predictor or noise parameters')
    return str(path)


def load_artifact_registry(path):
    """JSON entries are actual fields rather than file identity digests."""
    registry = _read(path)
    result = {}
    for row in registry.get('calibration_libraries', []):
        predictor, noise = row['predictor'], row['noise_std_m']
        if predictor not in PREDICTORS:
            raise ValueError('unknown registry predictor')
        _positive_level(noise, 'registry noise', zero=True)
        key = (predictor, float(noise))
        if key in result:
            raise ValueError('duplicate predictor/noise calibration entry')
        target = Path(row['path'])
        if not target.is_absolute():
            target = Path(path).resolve().parent/target
        result[key] = str(target.resolve())
    return result


def frozen_margin(path, predictor, noise, combination, *, phase):
    record = _read(path)
    if (record.get('schema') != 'stage5_fixed_margin_tuning_v1'
            or record.get('source_split') != 'tune'
            or record.get('test_data_used') is not False
            or record.get('source_phase') != phase):
        raise ValueError('A1 requires a frozen tune-only fixed-margin record')
    matches = [row for row in record['selections']
               if row['predictor'] == predictor
               and math.isclose(row['noise_std_m'], noise, abs_tol=1e-12, rel_tol=0)
               and row['risk_combination'] == combination]
    if len(matches) != 1:
        raise ValueError('A1 tuning must independently match predictor, noise and combination')
    from .stage5_protocol import manifest
    if record.get('source_phase') not in PHASES:
        raise ValueError('A1 tuning requires its actual registered source phase')
    expected = {row['case_id'] for row in manifest(record['source_phase'],split='tune',noise_std=noise)}
    actual = matches[0].get('source_case_ids',[])
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('A1 frozen selection must use all registered independent tune scenes')
    value = matches[0]['fixed_margin_m']
    _positive_level(value, 'fixed margin', zero=True)
    return float(value)


@dataclass(frozen=True)
class ExperimentJob:
    case: dict
    method: str = 'A0'
    predictor: str = 'ca'
    alpha: float | None = None
    combination: str = 'add'
    library_path: str | None = None
    fixed_margin_m: float | None = None
    tuning_path: str | None = None
    planning_delay_mode: str = 'none'
    planning_delay_s: float = 0.

    @property
    def noise(self):
        return float(self.case['config_overrides']['sensor_noise_std_m'])

    @property
    def job_id(self):
        terms = [self.case['case_id'], self.method, self.predictor]
        if self.method != 'A0':
            terms.append(self.combination)
        if self.alpha is not None:
            terms.append(f'alpha{self.alpha:g}')
        if self.fixed_margin_m is not None:
            terms.append(f'margin{self.fixed_margin_m:g}')
        terms.append('delay'+self.planning_delay_mode+(f'{self.planning_delay_s:g}'
                     if self.planning_delay_mode == 'fixed' else ''))
        return '__'.join(terms)

    def validate(self):
        select_cases([self.case])
        if self.method not in METHODS or self.predictor not in PREDICTORS:
            raise ValueError('unknown method or predictor')
        if self.combination not in COMBINATIONS or self.planning_delay_mode not in DELAY_MODES:
            raise ValueError('unknown risk combination or delay mode')
        _positive_level(self.planning_delay_s, 'planning delay', zero=True)
        if self.planning_delay_mode != 'fixed' and self.planning_delay_s != 0:
            raise ValueError('planning_delay_s is used only in fixed mode')
        if self.method in ('A0', 'A1'):
            if self.alpha is not None or self.library_path is not None:
                raise ValueError('A0/A1 have no calibrated alpha or calibration artifact')
        elif (self.alpha is None or isinstance(self.alpha, bool)
              or not math.isfinite(self.alpha) or not 0 < self.alpha < 1
              or not self.library_path):
            raise ValueError('A2-A5 require alpha in (0,1) and their independent calibration')
        if self.method == 'A1':
            _positive_level(self.fixed_margin_m, 'A1 fixed margin', zero=True)
            if self.case['split'] != 'tune' and not self.tuning_path:
                raise ValueError('A1 outside tuning requires its frozen independent tune record')
            if self.tuning_path and frozen_margin(self.tuning_path,self.predictor,self.noise,
                    self.combination,phase=self.case['phase']) != self.fixed_margin_m:
                raise ValueError('A1 fixed margin disagrees with its actual frozen selection')
        elif self.fixed_margin_m is not None or self.tuning_path is not None:
            raise ValueError('only A1 uses a tuned fixed margin')


def episode_metrics(job, record, workers=1, *, cfg=None):
    if cfg is None:
        from .stage5_protocol import config_for_case
        cfg = config_for_case(job.case)
    summary = record.get('summary', {})
    status = summary.get('status', 'program_error' if record.get('program_error') else 'missing_summary')
    cycles = record.get('planning_cycle_count')
    verified, feasible = record.get('verified_candidate_count'), record.get('feasible_candidate_count')
    misses = summary.get('planning_exceeds_control_period_count')
    row = dict.fromkeys(METRIC_FIELDS)
    row.update(case_id=job.case['case_id'], scene_id=job.case['scene_id'], phase=job.case['phase'],
        split=job.case['split'], family=job.case['family'], scenario=job.case['scenario'],
        scenario_seed=cfg.seed, method=job.method, predictor=job.predictor, noise_std_m=job.noise,
        risk_alpha=job.alpha, risk_combination=('engineering_baseline' if job.method == 'A0' else job.combination),
        fixed_margin_m=job.fixed_margin_m, calibration_path=job.library_path, tuning_path=job.tuning_path,
        planning_delay_mode=job.planning_delay_mode, planning_delay_s=job.planning_delay_s,
        program_error=bool(record.get('program_error', False)), status=status,
        navigation_success=summary.get('reached_goal'),
        collision=('collision' in status) if summary else None, timeout=(status == 'timeout') if summary else None,
        actual_clearance_margin_violation=(not summary['actual_safety_margin_met']
            if 'actual_safety_margin_met' in summary else None),
        minimum_actual_clearance_m=summary.get('minimum_model_clearance_m'),
        unsafe_emergency_forecast=(summary['unsafe_emergency_forecast_count'] > 0
            if 'unsafe_emergency_forecast_count' in summary else None),
        unsafe_emergency_forecast_count=summary.get('unsafe_emergency_forecast_count'),
        avoidance_acceptance_passed=summary.get('avoidance_acceptance_passed'),
        duration_s=summary.get('duration_s'), path_length_m=summary.get('path_length_m'),
        goal_distance_m=summary.get('goal_distance_m'),
        no_feasible_candidate_count=record.get('no_feasible_candidate_count'),
        no_feasible_candidate_fraction=(record['no_feasible_candidate_count']/cycles
            if cycles and record.get('no_feasible_candidate_count') is not None else None),
        emergency_count=summary.get('emergency_count'), planning_cycle_count=cycles,
        verified_candidate_count=verified, feasible_candidate_count=feasible,
        feasible_fraction=feasible/verified if verified else None,
        feasible_fraction_scope='Sum of feasible references / sum of fully verified shortlist references; not entire-grid feasibility.',
        actual_close_interactions=summary.get('actual_close_interaction_count'),
        maximum_observed_obstacles=summary.get('maximum_observed_obstacles'),
        maximum_tracked_obstacles=summary.get('maximum_tracked_obstacles'),
        lost_object_count=summary.get('lost_object_count'),
        planning_P95_ms=summary.get('planning_ms_p95'), planning_max_ms=summary.get('planning_ms_max'),
        planning_deadline_miss=misses, planning_deadline_miss_fraction=misses/cycles if cycles and misses is not None else None,
        physical_planning_delay_mean_s=record.get('physical_planning_delay_mean_s',summary.get('planning_computation_delay_mean_s')),
        physical_planning_delay_max_s=record.get('physical_planning_delay_max_s',summary.get('planning_computation_delay_max_s')),
        candidate_misclassification_scope=record.get('candidate_misclassification_scope',
            'not evaluated; absent counterfactual labels are not imputed'),
        calibration_coverage_scope=record.get('calibration_coverage_scope',
            'not evaluated; arrival and lack of collision are not calibration coverage'),
        prediction_RMSE_scope=record.get('prediction_RMSE_scope', 'not evaluated'),
        execution_prediction_rmse_m=summary.get('prediction_position_rmse_m'),
        all_states_finite=summary.get('all_states_finite'), candidate_budget=cfg.candidate_budget,
        physics_dt_s=cfg.physics_dt, control_dt_s=cfg.control_dt, horizon_s=cfg.horizon_s,
        transport_delay_s=cfg.command_delay_s, workers=workers,
        worker_wall_clock_s=record.get('worker_wall_clock_s'),
        timing_interpretation=('single-worker thread limits recorded; physical planning delay mode='+job.planning_delay_mode
            if workers == 1 else 'CPU contention across workers; unsuitable for paired latency advantage claims; physical planning delay mode='+job.planning_delay_mode),
        summary_path=record.get('summary_path'), log_path=record.get('log_path'))
    for key in ('candidate_false_accept_count', 'candidate_false_reject_count',
                'evaluated_selected_reference_anchor_count', 'calibration_coverage',
                'calibration_coverage_count', 'calibration_label_count', 'calibration_coverage_by_horizon',
                'prediction_RMSE_by_horizon'):
        row[key] = record.get(key)
    actual_fields = [row['navigation_success'],row['collision'],row['actual_clearance_margin_violation']]
    row['safe_arrival'] = (bool(actual_fields[0] and not actual_fields[1] and not actual_fields[2])
                          if all(value is not None for value in actual_fields) else None)
    return row


def _policy_and_forecaster(job, cfg):
    from .stage5_forecasting import make_forecaster
    from .stage5_risk import make_policy
    forecaster = make_forecaster(job.predictor, cfg)
    policy = make_policy(job.method, library_path=job.library_path,
        alpha=job.alpha if job.alpha is not None else .1,
        combination=job.combination, fixed_margin_m=job.fixed_margin_m,
        cfg=cfg, predictor=job.predictor)
    return policy, forecaster


def _execute_job(job, output, plot=True, animate=False):
    from .stage5_protocol import config_for_case, make_scene
    cfg = config_for_case(job.case)
    output = Path(output)
    directory, logfile = output/'cases'/job.job_id, output/'logs'/(job.job_id+'.log')
    record = dict(job_id=job.job_id, program_error=False, summary_path=str(directory/'summary.json'), log_path=str(logfile))
    started = time.perf_counter()
    with logfile.open('x', encoding='utf-8') as stream, redirect_stdout(stream), redirect_stderr(stream), chdir(ROOT):
        try:
            from .stage2_simulation import run_remus_episode, save_remus_episode
            print(json.dumps(dict(job=job.__dict__, config=cfg.to_dict()), ensure_ascii=False), flush=True)
            policy, forecaster = _policy_and_forecaster(job, cfg)
            result = run_remus_episode(cfg, job.case['scenario'], risk_policy=policy,
                forecaster=forecaster, scene=make_scene(job.case, cfg),
                planning_delay_mode=job.planning_delay_mode, planning_delay_s=job.planning_delay_s)
            save_remus_episode(result, directory, plot=plot, animate=animate)
            details = result['planning_diagnostics']
            delays = [row.get('planning_computation_delay_s', 0.) for row in details]
            record.update(summary=result['summary'], planning_cycle_count=len(details),
                no_feasible_candidate_count=sum(not row['feasible'] for row in details),
                verified_candidate_count=sum(row['verified_count'] for row in details),
                feasible_candidate_count=sum(row['valid_count'] for row in details),
                physical_planning_delay_mean_s=sum(delays)/len(delays) if delays else None,
                physical_planning_delay_max_s=max(delays) if delays else None)
            write_json(directory/'stage5_experiment.json', dict(job=job.__dict__,
                actual_predictor=job.predictor, thread_environment={key:os.environ.get(key) for key in THREAD_VARIABLES},
                cfg_predictor_field_scope='Legacy cfg field is retained; actual predictor is explicitly injected.',
                online_future_truth_supplied=False))
            print(json.dumps(result['summary'], ensure_ascii=False), flush=True)
        except Exception as error:
            record.update(program_error=True, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            print(record['traceback'], flush=True)
            directory.mkdir(parents=True, exist_ok=True)
            write_json(directory/'program_error.json', record)
    record['worker_wall_clock_s'] = time.perf_counter()-started
    return record


GROUP_FIELDS = ('phase', 'split', 'family', 'method', 'predictor', 'noise_std_m',
                'risk_alpha', 'risk_combination', 'fixed_margin_m', 'planning_delay_mode', 'planning_delay_s')


def grouped_results(rows):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in GROUP_FIELDS), []).append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda pair:str(pair[0])):
        item = dict(zip(GROUP_FIELDS, key))
        item.update(registered_runs=len(group), program_error_count=sum(row['program_error'] for row in group))
        for field in ('navigation_success', 'safe_arrival', 'collision', 'timeout', 'actual_clearance_margin_violation',
                      'unsafe_emergency_forecast', 'avoidance_acceptance_passed'):
            values = [row[field] for row in group if row[field] is not None]
            item[field+'_count'] = sum(bool(value) for value in values)
            item[field+'_denominator'] = len(values)
            item[field+'_rate'] = sum(bool(value) for value in values)/len(values) if values else None
        item['denominator_scope'] = 'All research failures retained; missing/program-error outcomes remain unknown and separately counted. No pooling across method, predictor, noise, risk combination, alpha, phase, family or delay.'
        result.append(item)
    return result


def run_study(output, *, cases=None, phase='pilot', splits=None, case_ids=None, noise_std=None,
              limit=None, methods=('A0',), predictors=('ca',), alphas=None, combinations=('add',),
              calibration_paths=None, tuned_margin_path=None, fixed_margins=None,
              planning_delay_mode='none', planning_delay_s=0., workers=1, plot=True, animate=False):
    from .stage5_protocol import manifest, config_for_case, PHASE_ALPHA, protocol_record
    if phase not in PHASES:
        raise ValueError('phase must be pilot or formal')
    if isinstance(workers,bool) or not isinstance(workers,int) or workers < 1:
        raise ValueError('workers must be a positive integer')
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('existing experiments are preserved; choose a fresh output directory')
    registered = list(manifest(phase=phase) if cases is None else cases)
    chosen = select_cases(registered,splits=splits,case_ids=case_ids,noise_std=noise_std,limit=limit)
    alphas = (PHASE_ALPHA[phase],) if alphas is None else tuple(alphas)
    for values, allowed, label in ((methods,METHODS,'methods'),(predictors,PREDICTORS,'predictors'),
                                   (combinations,COMBINATIONS,'combinations')):
        if not values or len(set(values)) != len(values) or any(value not in allowed for value in values):
            raise ValueError(f'{label} must be nonempty unique registered values')
    if not alphas or len(set(alphas)) != len(alphas) or any(isinstance(a,bool) or not 0 < a < 1 for a in alphas):
        raise ValueError('alphas must be unique and within (0,1)')
    if fixed_margins is not None:
        if tuple(methods) != ('A1',) or any(row['split'] != 'tune' for row in chosen):
            raise ValueError('fixed-margin scanning is restricted to A1 independent tune data')
        if tuned_margin_path is not None or not fixed_margins or len(set(fixed_margins)) != len(fixed_margins):
            raise ValueError('tune scan requires unique margins without a frozen-margin input')
        for value in fixed_margins:
            _positive_level(value,'fixed margin',zero=True)
    jobs = []
    for case in chosen:
        for predictor in predictors:
            for method in methods:
                noise = config_for_case(case).sensor_noise_std_m
                for combination in (('add',) if method == 'A0' else combinations):
                    library = artifact_path(calibration_paths,predictor,noise,phase=case['phase']) if method in METHODS[2:] else None
                    margins = (None,)
                    if method == 'A1':
                        if fixed_margins is not None:
                            margins = tuple(fixed_margins)
                        elif tuned_margin_path is None:
                            raise ValueError('A1 requires an independent frozen tune record')
                        else:
                            margins = (frozen_margin(tuned_margin_path,predictor,noise,combination,phase=case['phase']),)
                    for margin in margins:
                        for alpha in ((None,) if method in ('A0','A1') else alphas):
                            job = ExperimentJob(case,method,predictor,alpha,combination,library,margin,
                                str(Path(tuned_margin_path).resolve()) if method == 'A1' and tuned_margin_path else None,
                                planning_delay_mode,planning_delay_s)
                            job.validate()
                            jobs.append(job)
    out.mkdir(parents=True,exist_ok=True)
    for folder in ('cases','logs','registered_configs'):
        (out/folder).mkdir()
    for case in chosen:
        write_json(out/'registered_configs'/(case['case_id']+'.json'),config_for_case(case).to_dict())
    write_json(out/'EXPERIMENT_CONFIG.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        phase=phase,protocol=protocol_record(phase),registered_cases=registered,selected_case_ids=[case['case_id'] for case in chosen],
        jobs=[dict(job.__dict__,job_id=job.job_id) for job in jobs],workers=workers,plot=plot,animate=animate,
        thread_environment={key:os.environ.get(key) for key in THREAD_VARIABLES},
        timing_scope='Serial worker timing is preferred; other OS load remains uncontrolled. Physical planning delay is an explicit independent factor.',
        scene_count_scope='Noise variants share scene_id and are not independent geometry replicates.'))
    records = {}
    if workers == 1:
        for job in jobs:
            records[job.job_id] = _execute_job(job,out,plot,animate)
            print(job.job_id+': '+records[job.job_id].get('summary',{}).get('status','PROGRAM_ERROR'),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(_execute_job,job,out,plot,animate):job for job in jobs}
            for future in as_completed(pending):
                job = pending[future]
                try:
                    records[job.job_id] = future.result()
                except Exception as error:
                    records[job.job_id] = dict(job_id=job.job_id,program_error=True,error=str(error),
                        error_type=type(error).__name__,traceback=traceback.format_exc())
                print(job.job_id+': '+records[job.job_id].get('summary',{}).get('status','PROGRAM_ERROR'),flush=True)
    ordered = [records[job.job_id] for job in jobs]
    rows = [episode_metrics(job,record,workers) for job,record in zip(jobs,ordered)]
    write_json(out/'EXPERIMENT_RECORDS.json',ordered)
    write_json(out/'EXPERIMENT_METRICS.json',rows)
    csv_rows = [dict(row,prediction_RMSE_by_horizon=json.dumps(row['prediction_RMSE_by_horizon']),
        calibration_coverage_by_horizon=json.dumps(row['calibration_coverage_by_horizon'])) for row in rows]
    write_csv(out/'EXPERIMENT_SUMMARY.csv',csv_rows,METRIC_FIELDS)
    write_json(out/'GROUPED_SUMMARY.json',grouped_results(rows))
    outcome = dict(registered_case_count=len(registered),selected_case_count=len(chosen),
        selected_geometry_count=len({case['scene_id'] for case in chosen}),executed_job_count=len(jobs),
        program_error_count=sum(record['program_error'] for record in ordered),mixed_overall_success_rate=None,
        output_directory=str(out),outcome_scope='Read distinct paired groups; program completion is not safe arrival.')
    write_json(out/'STUDY_SUMMARY.json',outcome)
    return outcome
