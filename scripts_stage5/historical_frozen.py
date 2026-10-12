"""Secondary old-artifact sensitivity, separate from the matched A0--A5 study.

Exactly two predeclared add-subset scenes, original stage4 B1/B2 numerics and
alpha .1. This post-primary-registration sensitivity is not a new main study
or a pure calibration-pool ablation: carrier and empty-scene protocols differ.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import chdir, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_LIBRARY = ROOT/'artifacts_stage4'/'ca_scene_calibration_v1.npz'
SCENE_IDS = ('s5_formal_static3d_g14_s520014',
             's5_formal_crossing3d_g14_s521014')
LABELS = {'B1': 'H4_B1_original', 'B2': 'H4_B2_original'}
FIELDS = ('case_id', 'scene_id', 'phase', 'split', 'family', 'scenario_seed',
          'historical_method', 'source_risk_method', 'predictor', 'noise_std_m',
          'alpha', 'combination', 'program_error', 'status', 'navigation_success',
          'safe_arrival', 'collision', 'actual_clearance_margin_violation',
          'minimum_actual_clearance_m', 'timeout', 'duration_s', 'path_length_m',
          'goal_distance_m', 'unsafe_emergency_forecast_count',
          'avoidance_acceptance_passed', 'no_feasible_candidate_count',
          'emergency_count', 'planning_P95_ms', 'planning_deadline_miss',
          'all_states_finite', 'workers', 'timing_scope', 'summary_path', 'log_path')
SCOPE = ('Secondary post-primary-registration historical numerical-protocol sensitivity; '
         'not pooled with A0-A5 and not a pure calibration-pool ablation. '
         'Old 24-scene library includes 8 empty scenes and actual B0 histories; '
         'new matched library uses active scenes and a fixed observation carrier.')


def configure_threads():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'


def registered_jobs():
    from auv3d.stage5_protocol import manifest, config_for_case
    rows = manifest('formal', split='test', noise_std=.02)
    selected = [next(row for row in rows if row['scene_id'] == scene_id) for scene_id in SCENE_IDS]
    for case in selected:
        config_for_case(case)
    return [dict(case=case, source_risk_method=method,
                 historical_method=LABELS[method], job_id=case['case_id']+'__'+LABELS[method])
            for case in selected for method in LABELS]


def make_original_policy(method):
    if method not in LABELS:
        raise ValueError('historical sensitivity only supports original B1/B2')
    from auv3d.stage4_risk import RiskPolicy
    # Path and ordinary actual-parameter/shape checks; no file summaries.
    return RiskPolicy(method, ORIGINAL_LIBRARY, alpha=.1)


def metrics(job, record, workers):
    summary = record.get('summary', {})
    status = summary.get('status', 'program_error' if record.get('program_error') else 'missing_summary')
    arrived = summary.get('reached_goal')
    collision = ('collision' in status) if summary else None
    violation = not summary['actual_safety_margin_met'] if 'actual_safety_margin_met' in summary else None
    safe = bool(arrived and not collision and not violation) if all(v is not None for v in (arrived, collision, violation)) else None
    case = job['case']
    return dict(case_id=case['case_id'], scene_id=case['scene_id'], phase=case['phase'],
        split=case['split'], family=case['family'], scenario_seed=case['seed'],
        historical_method=job['historical_method'], source_risk_method=job['source_risk_method'],
        predictor='ca', noise_std_m=.02, alpha=.1, combination='original_add',
        program_error=bool(record.get('program_error')), status=status,
        navigation_success=arrived, safe_arrival=safe, collision=collision,
        actual_clearance_margin_violation=violation,
        minimum_actual_clearance_m=summary.get('minimum_model_clearance_m'),
        timeout=(status == 'timeout') if summary else None, duration_s=summary.get('duration_s'),
        path_length_m=summary.get('path_length_m'), goal_distance_m=summary.get('goal_distance_m'),
        unsafe_emergency_forecast_count=summary.get('unsafe_emergency_forecast_count'),
        avoidance_acceptance_passed=summary.get('avoidance_acceptance_passed'),
        no_feasible_candidate_count=record.get('no_feasible_candidate_count'),
        emergency_count=summary.get('emergency_count'), planning_P95_ms=summary.get('planning_ms_p95'),
        planning_deadline_miss=summary.get('planning_exceeds_control_period_count'),
        all_states_finite=summary.get('all_states_finite'), workers=workers,
        timing_scope=('Worker wall times are contextual only; this run is not a matched isolated timing experiment.'),
        summary_path=record.get('summary_path'), log_path=record.get('log_path'))


def _execute(job, out, plot, animate):
    configure_threads()
    from auv3d.stage4_experiments import write_json
    from auv3d.stage5_protocol import config_for_case, make_scene
    from auv3d.stage5_forecasting import make_forecaster
    from auv3d.stage2_simulation import run_remus_episode, save_remus_episode
    out = Path(out)
    directory = out/'cases'/job['job_id']
    log = out/'logs'/(job['job_id']+'.log')
    record = dict(job_id=job['job_id'], program_error=False,
                  summary_path=str(directory/'summary.json'), log_path=str(log))
    started = time.perf_counter()
    with log.open('x', encoding='utf8') as stream, redirect_stdout(stream), redirect_stderr(stream), chdir(ROOT):
        try:
            cfg = config_for_case(job['case'])
            policy = make_original_policy(job['source_risk_method'])
            print(json.dumps(dict(job=job, actual_config=cfg.to_dict(), scope=SCOPE),ensure_ascii=False),flush=True)
            result = run_remus_episode(cfg, job['case']['scenario'], risk_policy=policy,
                forecaster=make_forecaster('ca',cfg), scene=make_scene(job['case'],cfg))
            save_remus_episode(result,directory,plot=plot,animate=animate)
            write_json(directory/'historical_experiment.json',dict(job=job,
                actual_config=cfg.to_dict(), original_library_path=str(ORIGINAL_LIBRARY),
                alpha=.1, combination='original_add', actual_predictor='ca',
                scope=SCOPE, online_future_truth_supplied=False,
                original_library_case_ids=policy.metadata['calibration_case_ids'],
                formal_primary_parameters_modified=False))
            record.update(summary=result['summary'],
                no_feasible_candidate_count=sum(not p['feasible'] for p in result['planning_diagnostics']))
            print(json.dumps(result['summary'],ensure_ascii=False),flush=True)
        except Exception as exc:
            record.update(program_error=True,error_type=type(exc).__name__,error=str(exc),
                          traceback=traceback.format_exc())
            print(record['traceback'],flush=True)
            directory.mkdir(parents=True,exist_ok=True)
            write_json(directory/'program_error.json',record)
    record['worker_wall_s_not_benchmark'] = time.perf_counter()-started
    return record


def run(output, *, workers=1, plot=True, animate=False):
    configure_threads()
    if isinstance(workers,bool) or workers not in (1,2):
        raise ValueError('historical sensitivity supports one or two workers; coordinate global CPU budget first')
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('existing historical sensitivity is preserved; choose a fresh output')
    if not ORIGINAL_LIBRARY.is_file():
        raise FileNotFoundError(f'original stage4 calibration library is missing: {ORIGINAL_LIBRARY}')
    jobs = registered_jobs()
    # Fail at setup if the preserved library cannot satisfy its original API.
    policies = {method:make_original_policy(method) for method in LABELS}
    from auv3d.stage4_experiments import write_json, write_csv
    out.mkdir(parents=True,exist_ok=True)
    (out/'cases').mkdir()
    (out/'logs').mkdir()
    write_json(out/'REGISTRATION.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        scope=SCOPE,jobs=jobs,workers=workers,plot=plot,animate=animate,
        alpha=.1,combination='original_add',predictor='ca',noise_std_m=.02,
        original_library_path=str(ORIGINAL_LIBRARY),
        original_library_metadata={key:policy.metadata for key,policy in policies.items()},
        formal_primary_parameters_modified=False,
        comparator='Same two formal A3/A4 add episodes, not replace; all failures retained',
        concurrency_instruction='Launch only after explicit coordination releases CPU slots.'))
    records = {}
    if workers == 1:
        for job in jobs:
            records[job['job_id']] = _execute(job,out,plot,animate)
            print(job['job_id']+': '+records[job['job_id']].get('summary',{}).get('status','PROGRAM_ERROR'),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(_execute,job,out,plot,animate):job for job in jobs}
            for future in as_completed(pending):
                job = pending[future]
                try:
                    records[job['job_id']] = future.result()
                except Exception as exc:
                    records[job['job_id']] = dict(job_id=job['job_id'],program_error=True,
                        error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
                print(job['job_id']+': '+records[job['job_id']].get('summary',{}).get('status','PROGRAM_ERROR'),flush=True)
    ordered = [records[job['job_id']] for job in jobs]
    rows = [metrics(job,record,workers) for job,record in zip(jobs,ordered)]
    write_json(out/'RESULTS.json',ordered)
    write_json(out/'EXPERIMENT_METRICS.json',rows)
    write_csv(out/'EXPERIMENT_SUMMARY.csv',rows,FIELDS)
    summary = dict(executed_jobs=len(jobs),program_error_count=sum(r['program_error'] for r in ordered),
        scope=SCOPE,mixed_main_study_success_rate=None,output_directory=str(out))
    write_json(out/'SUMMARY.json',summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--workers',type=int,choices=(1,2),default=1)
    parser.add_argument('--no-plot',action='store_true')
    parser.add_argument('--animation',action='store_true')
    args = parser.parse_args(argv)
    result = run(args.out,workers=args.workers,plot=not args.no_plot,animate=args.animation)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 2 if result['program_error_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
