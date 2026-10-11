"""Run real new REMUS budget experiments and read-only failure diagnostics."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(ROOT))
from auv3d.stage2_config import Stage2Config

CASE_SPECS={
    'static3d':('static3d',{'max_time_s':150.}),
    'terrain3d':('terrain3d',{'max_time_s':150.}),
    'paper_random3d':('paper_random3d',{'max_time_s':200.}),
    'multi_speed15':('multi_crossing3d',{'max_time_s':150.,'obstacle_speed_scale':1.5}),
    'maneuver_turn8speed14':('maneuvering3d',{'max_time_s':150.,'obstacle_speed_scale':1.4,'obstacle_turn_rate_deg_s':8.}),
}


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def single_case(name,budget,out,worker_limit):
    from auv3d.stage2_failure_analysis import FailureObserver,run_with_observer,diagnose_failures,analyze_obstacle_forecasts,analyze_actual_margin
    from auv3d.stage2_simulation import save_remus_episode
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; experiments are never overwritten')
    scenario,changes=CASE_SPECS[name]
    cfg=replace(Stage2Config(),physics_dt=.02,control_dt=2.,horizon_s=24.,seed=7,
                candidate_budget=budget,command_delay_s=0.,**changes)
    started=now()
    timer=time.perf_counter()
    print(f'START {name} budget={budget} at {started}',flush=True)
    observer=FailureObserver(cfg,lambda line:print(line,flush=True))
    result=run_with_observer(cfg,scenario,observer)
    execution_completed=now()
    save_remus_episode(result,out,animate=False,plot=True)
    print(f'EXECUTION_SAVED status={result["summary"]["status"]} at {execution_completed}',flush=True)
    failures=diagnose_failures(result,observer,out,logger=lambda line:print(line,flush=True))
    forecasts=analyze_obstacle_forecasts(result,observer,out)
    actual_margin=analyze_actual_margin(result,cfg,out)
    record={
        'case':name,'scenario':scenario,'budget':budget,'started_at':started,
        'execution_completed_at':execution_completed,'finished_at':now(),
        'whole_case_wall_clock_s':time.perf_counter()-timer,
        'actual_command_argv':[sys.executable,*sys.argv],
        'closed_loop_summary':result['summary'],'failure_diagnostics':failures,
        'obstacle_forecast_metrics':forecasts,
        'actual_margin_diagnostics':actual_margin,
        'planning_latency_note':f'actual planner timers exclude observer/offline analysis; machine shared by up to {worker_limit} study subprocesses, so P95 is under CPU contention and is not an isolated real-time capability claim',
        'controller_states':'captured in memory at each actual planning call; no restoration/inference from CSV',
        'future_truth_used_by_planner':False,'offline_full_grid_feedback_to_execution':False,
    }
    (out/'case_study.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(record,indent=2,allow_nan=False),flush=True)
    return record


def run_child(name,budget,out,worker_limit):
    case_name=f'{name}_budget{budget}'
    case_out=out/case_name
    log_path=out/'logs'/f'{case_name}.log'
    command=[sys.executable,str(Path(__file__).absolute()),'--single','--case',name,
             '--budget',str(budget),'--out',str(case_out),'--workers',str(worker_limit)]
    environment=os.environ.copy()
    environment['PYTHONIOENCODING']='utf-8'
    environment['PYTHONUNBUFFERED']='1'
    environment['OMP_NUM_THREADS']='1'
    environment['OPENBLAS_NUM_THREADS']='1'
    start=now()
    timer=time.perf_counter()
    with log_path.open('x',encoding='utf-8') as log:
        log.write(f'STARTED_AT {start}\nACTUAL_COMMAND {subprocess.list2cmdline(command)}\n\n')
        log.flush()
        completed=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=False)
        log.write(f'\nFINISHED_AT {now()}\nEXIT_CODE {completed.returncode}\n')
    data_path=case_out/'case_study.json'
    data=json.loads(data_path.read_text(encoding='utf-8')) if data_path.is_file() else None
    summary_path=case_out/'summary.json'
    summary=json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else None
    return {'case':name,'budget':budget,'name':case_name,'actual_command_argv':command,
            'started_at':start,'finished_at':now(),'process_wall_clock_s':time.perf_counter()-timer,
            'exit_code':completed.returncode,'output_directory':str(case_out),'command_log':str(log_path),
            'closed_loop_summary':summary,'case_study_available':data is not None,
            'failure_diagnostics':data['failure_diagnostics'] if data else None,
            'obstacle_forecast_metrics':data['obstacle_forecast_metrics'] if data else None}


def budget_statistics(rows):
    """Pool actual planning-call timers, rather than averaging case P95s."""
    import numpy as np
    budgets={}
    for budget in (9,18,45):
        group=[r for r in rows if int(r['budget'])==budget and not int(r['process_exit_code'])]
        if not group:
            continue
        latency=np.concatenate([np.loadtxt(Path(r['case_directory'])/'planning.csv',delimiter=',',skiprows=1,ndmin=2)[:,1] for r in group])
        truth=lambda value:str(value).lower()=='true'
        total=lambda field:sum(int(r[field]) for r in group)
        value={'actual_cases':len(group),'navigation_arrived':sum(truth(r['reached_goal']) for r in group),
               'safety_accepted':sum(truth(r['avoidance_acceptance_passed']) for r in group),
               'timeouts':sum(r['status']=='timeout' for r in group),
               'actual_collisions':sum('collision' in r['status'] for r in group),
               'actual_margin_failed_cases':sum(not truth(r['actual_margin_met']) for r in group),
               'actual_margin_violating_physics_intervals':total('actual_margin_violating_intervals'),
               'actual_margin_violations_after_feasible_plan':total('actual_margin_violation_after_feasible_plan_intervals'),
               'unsafe_emergencies':total('unsafe_emergency_count'),
               'shortlist_omission_frames':total('shortlist_omission_frames'),
               'no_safe_current_grid_frames':total('no_safe_current_grid_frames'),
               'numerical_failure_frames':total('numerical_failure_frames'),
               'actual_plan_call_count':len(latency),
               'planning_pooled_p95_ms':float(np.percentile(latency,95)),
               'planning_pooled_median_ms':float(np.median(latency))}
        for name,column in (('depth_reference_rmse_range_m','actual_depth_reference_rmse_m'),
                            ('heading_reference_rmse_range_deg','actual_heading_reference_rmse_deg'),
                            ('applied_to_actual_rpm_rmse_range','actual_applied_to_actuator_rpm_rmse')):
            value[name]=[min(float(r[column]) for r in group),max(float(r[column]) for r in group)]
        budgets[str(budget)]=value
    return {'budgets':budgets,
            'latency_note':'pooled actual planner timers under shared concurrent CPU load; physics paused while planning; command delay zero',
            'vehicle_prediction_error_all_rows_m':max((float(r['vehicle_selected_reference_prediction_max_m']) for r in rows if not int(r['process_exit_code'])),default=None)}


def summarize_study(out):
    """Create new derived reports from all recorded runs; never rerun a plant."""
    import numpy as np
    from auv3d.stage2_failure_analysis import analyze_saved_actual_margin
    out=Path(out)
    manifest=json.loads((out/'budget_study.json').read_text(encoding='utf-8'))
    if manifest['completed_count']!=manifest['case_count'] or manifest.get('finished_at') is None:
        raise RuntimeError('all requested actual study processes must complete before final summarization')
    report=out/'derived_reports'
    report.mkdir(exist_ok=False)
    rows=[]
    event_rows=[]
    candidate_rows=[]
    for execution in sorted(manifest['records'],key=lambda r:(list(CASE_SPECS).index(r['case']),r['budget'])):
        path=out/execution.get('name',f'{execution["case"]}_budget{execution["budget"]}')
        if execution['exit_code']!=0:
            rows.append({'case':execution['case'],'budget':execution['budget'],
                         'process_exit_code':execution['exit_code'],'status':'process_error'})
            continue
        summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
        failure=json.loads((path/'failure_analysis_summary.json').read_text(encoding='utf-8'))
        forecast=json.loads((path/'obstacle_forecast_summary.json').read_text(encoding='utf-8'))
        margin_path=path/'actual_margin_diagnostics'/'summary.json'
        margin=json.loads(margin_path.read_text(encoding='utf-8')) if margin_path.is_file() else analyze_saved_actual_margin(path)
        states=np.loadtxt(path/'states.csv',delimiter=',',skiprows=1,ndmin=2)
        controls=np.loadtxt(path/'controls.csv',delimiter=',',skiprows=1,ndmin=2)
        config=json.loads((path/'config.json').read_text(encoding='utf-8'))
        end_times=controls[:,0]+config['physics_dt']
        depth=np.interp(end_times,states[:,0],states[:,3])
        heading=np.interp(end_times,states[:,0],np.unwrap(states[:,6]))
        heading_error=(heading-controls[:,2]+np.pi)%(2*np.pi)-np.pi
        response={'depth_reference_rmse_m':float(np.sqrt(np.mean((depth-controls[:,1])**2))),
                  'heading_reference_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(heading_error**2)))),
                  'applied_to_actual_rpm_rmse':float(np.sqrt(np.mean((controls[:,9]-controls[:,12])**2))),
                  'maximum_actual_abs_rudder_deg':float(np.rad2deg(np.abs(controls[:,10]).max())),
                  'maximum_actual_abs_stern_deg':float(np.rad2deg(np.abs(controls[:,11]).max())),
                  'maximum_actual_rpm':float(controls[:,12].max()),
                  'rudder_saturation_interval_count':int(np.count_nonzero(controls[:,13])),
                  'stern_saturation_interval_count':int(np.count_nonzero(controls[:,14])),
                  'rpm_saturation_interval_count':int(np.count_nonzero(controls[:,15])),
                  'scope':'actual interval-end vehicle/actuator response to the active depth, heading and RPM references; full episode, including reference changes',
                  'distinct_from_vehicle_prediction_error':True,'distinct_from_obstacle_ca_forecast_error':True}
        (report/f'{path.name}_control_response.json').write_text(json.dumps(response,indent=2),encoding='utf-8')
        row={'case':execution['case'],'budget':execution['budget'],'backend':'remus6dof',
             'process_exit_code':execution['exit_code'],'status':summary['status'],
             'reached_goal':summary['reached_goal'],
             'avoidance_acceptance_passed':summary['avoidance_acceptance_passed'],
             'safety_validation_status':summary['safety_validation_status'],
             'duration_s':summary['duration_s'],'minimum_clearance_m':summary['minimum_model_clearance_m'],
             'actual_margin_met':margin['actual_margin_met'],
             'actual_margin_violating_intervals':margin['violating_physics_interval_count'],
             'actual_margin_violation_after_feasible_plan_intervals':margin['violating_intervals_after_feasible_plan'],
             'actual_margin_violation_after_emergency_plan_intervals':margin['violating_intervals_after_emergency_plan'],
             'actual_margin_violation_interval_total_s':margin['violating_interval_total_duration_s'],
             'emergency_count':summary['emergency_count'],'unsafe_emergency_count':summary['unsafe_emergency_forecast_count'],
             'shortlist_omission_frames':failure['categories'].get('shortlist_omission',0),
             'no_safe_current_grid_frames':failure['categories'].get('no_safe_reference_in_current_discrete_grid',0),
             'numerical_failure_frames':failure['categories'].get('no_verified_reference_with_numerical_prediction_failures',0),
             'actual_close_interaction_count':summary['actual_close_interaction_count'],
             'max_simultaneous_close_interactions':summary['maximum_simultaneous_close_interactions'],
             'max_observed_obstacles':summary['maximum_observed_obstacles'],
             'max_tracked_obstacles':summary['maximum_tracked_obstacles'],
             'planning_ms_median':summary['planning_ms_median'],'planning_ms_p95':summary['planning_ms_p95'],
             'planning_ms_max':summary['planning_ms_max'],
             'planning_exceeds_control_period_count':summary['planning_exceeds_control_period_count'],
             'vehicle_selected_reference_prediction_rmse_m':summary['prediction_position_rmse_m'],
             'vehicle_selected_reference_prediction_max_m':summary['prediction_position_max_error_m'],
             'actual_depth_reference_rmse_m':response['depth_reference_rmse_m'],
             'actual_heading_reference_rmse_deg':response['heading_reference_rmse_deg'],
             'actual_applied_to_actuator_rpm_rmse':response['applied_to_actual_rpm_rmse'],
             'obstacle_ca_2s_sample_count':forecast['at_2s']['sample_count'],
             'obstacle_ca_2s_rmse_m':forecast['at_2s']['rmse_m'],
             'obstacle_ca_2s_max_m':forecast['at_2s']['max_error_m'],
             'obstacle_ca_24s_sample_count':forecast['at_24s']['sample_count'],
             'obstacle_ca_24s_rmse_m':forecast['at_24s']['rmse_m'],
             'obstacle_ca_24s_max_m':forecast['at_24s']['max_error_m'],
             'case_directory':str(path),'command_log':execution['command_log']}
        rows.append(row)
        for filename in sorted((path/'failure_diagnostics').glob('event_*.json')):
            event=json.loads(filename.read_text(encoding='utf-8'))
            grid=event['grid']
            candidates=grid['candidates']
            # Regular feasible collision-context records are not emergencies,
            # even if their separate emergency_verified_safe field is false.
            unsafe=not event['selected_feasible'] and not event['emergency_verified_safe']
            entry={'case':execution['case'],'budget':execution['budget'],'time_s':event['time_s'],
                   'trigger':event['trigger'],'category':event['category'],
                   'selected_feasible':event['selected_feasible'],'unsafe_emergency':unsafe,
                   'selected_prediction_clearance_m':event['selected_prediction_clearance_m'],
                   'full_grid_feasible_count':grid['feasible_count'],
                   'grid_evaluation_ms':grid['evaluation_ms'],'evaluation_source':event['evaluation_source'],
                   'full_record':str(filename)}
            for budget in (9,18,45):
                selected=[r for r in candidates if r['in_shortlist'][str(budget)] and r['feasible']]
                entry[f'grid_shortlist{budget}_feasible_count']=len(selected)
                entry[f'grid_shortlist{budget}_best_feasible_cost']=min((r['cost'] for r in selected),default=None)
            event_rows.append(entry)
            for candidate in candidates:
                candidate_rows.append({'case':execution['case'],'budget':execution['budget'],'time_s':event['time_s'],
                                       'index':candidate['index'],'depth_reference_m':candidate['reference'][0],
                                       'heading_reference_rad':candidate['reference'][1],'rpm_reference':candidate['reference'][2],
                                       'fast_clearance_m':candidate['fast_clearance_m'],'fast_cost':candidate['fast_cost'],
                                       'feasible':candidate['feasible'],'full_clearance_m':candidate['minimum_predicted_clearance_m'],
                                       'full_cost':candidate['cost'],'prediction_failure':candidate['prediction_failure'],
                                       **{f'in_shortlist{b}':candidate['in_shortlist'][str(b)] for b in (9,18,45)}})
    for filename,items,empty_columns in (
            ('budget_comparison.csv',rows,('case','budget','status')),
            ('failure_event_comparison.csv',event_rows,('case','budget','time_s')),
            ('candidate_grid_diagnostics.csv',candidate_rows,('case','budget','time_s','index'))):
        columns=list(dict.fromkeys(key for item in items for key in item)) or list(empty_columns)
        with (report/filename).open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader()
            writer.writerows(items)
    totals=budget_statistics(rows)
    (report/'budget_totals.json').write_text(json.dumps(totals,indent=2,allow_nan=False),encoding='utf-8')
    text=['# REMUS candidate-budget failure analysis','',
          f'All {len(rows)} listed conditions were newly executed for this study. Seed 7, horizon 24 s, physical step 0.02 s, control period 2 s; zero configured command delay. Original dynamics and the 1 m safety margin were unchanged.',
          '', 'Arrival and safety acceptance are separate. An unsafe emergency forecast or actual sub-margin interval fails safety acceptance, even when the endpoint is reached. A process exit of zero means that the experiment and its diagnostics completed.',
          '', '| Condition | Budget | Result | Safety accepted | Min clearance (m) | Actual sub-margin intervals | Unsafe emergencies | Missed safe-grid frames | No safe-grid frames | P95 planning (ms) |',
          '|---|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['process_exit_code']:
            text.append(f'| {row["case"]} | {row["budget"]} | process error | unavailable | | | | | | |')
        else:
            text.append(f'| {row["case"]} | {row["budget"]} | {row["status"]} | {row["avoidance_acceptance_passed"]} | {row["minimum_clearance_m"]:.3f} | {row["actual_margin_violating_intervals"]} | {row["unsafe_emergency_count"]} | {row["shortlist_omission_frames"]} | {row["no_safe_current_grid_frames"]} | {row["planning_ms_p95"]:.1f} |')
    text += ['', '## Actual execution and obstacle forecast evidence', '',
             '| Condition | Budget | Sub-margin intervals after feasible plan | CA 2 s RMSE (m) | CA 24 s RMSE (m) | Actual interacted IDs | Max observed / tracked |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['process_exit_code']:
            continue
        rmse2=f'{row["obstacle_ca_2s_rmse_m"]:.3f}' if row['obstacle_ca_2s_rmse_m'] is not None else 'no tracks'
        rmse24=f'{row["obstacle_ca_24s_rmse_m"]:.3f}' if row['obstacle_ca_24s_rmse_m'] is not None else 'no tracks'
        text.append(f'| {row["case"]} | {row["budget"]} | {row["actual_margin_violation_after_feasible_plan_intervals"]} | {rmse2} | {rmse24} | {row["actual_close_interaction_count"]} | {row["max_observed_obstacles"]} / {row["max_tracked_obstacles"]} |')
    text += ['', 'An actual sub-margin interval after a feasible decision is retained as a safety failure. Its presence alongside forecast error does not, by itself, establish which observation, model or decision caused that interval; the per-ID forecast rows and captured planning-state records provide the evidence for a closer audit. Total obstacle population is not used as a substitute for actual interaction or observation density.', '']
    terrain_frames=sorted((out/'terrain3d_budget9'/'failure_diagnostics').glob('event_*.json'))
    if terrain_frames:
        event=json.loads(terrain_frames[0].read_text(encoding='utf-8'))
        candidates=[r for r in event['grid']['candidates'] if r['minimum_predicted_clearance_m'] is not None]
        if candidates and not event['grid']['feasible_count']:
            best=max(candidates,key=lambda r:r['minimum_predicted_clearance_m'])
            text += [f'Terrain example at t={event["time_s"]:g} s (budget 9): all 45 tested moving references fail the unchanged clearance requirement. The largest predicted clearance is {best["minimum_predicted_clearance_m"]:.6f} m at depth/heading/RPM reference {best["reference"]}; emergency coasting predicts {event["selected_prediction_clearance_m"]:.6f} m. This records a limitation of the finite local reference grid while the reachable scene also contains side/upper passages. Keeping a safe coast can still end in task timeout.', '']
    text += ['## Interpretation limits', '',
             '- Failed shortlisted plans were checked offline against all 45 references using a complete controller/actuator/reference-state clone captured at that exact planning call. Budget 45 records reuse the actual live full-grid evaluations. Previous selected references reproduce history cost. No offline result changed the executed action.',
             '- A shortlist omission means a safe reference existed in that finite grid under the same causal CA forecast. No safe grid reference means only that these 45 references failed this model check; it does not prove global unavoidability. Different budgets subsequently visit different states, so their episode differences are not single-decision counterfactuals.',
             '- Actual clearance comes from the same swept physical-interval collision evaluator as execution. The violation CSV includes every interval below the unchanged 1 m margin, including intervals following a feasible plan. Summed interval duration is the duration of intervals containing a breach, not an exact measurement of continuous exposure time.',
             '- Vehicle prediction error compares only a selected held reference until the next planning decision. Small or zero deterministic clone error does not mean immediate reference tracking. Separate control_response JSON reports actual depth/heading reference errors and actuator RPM lag.',
             '- Obstacle forecast error compares CA positions with the same obstacle ID at the same absolute simulation time. All recorded obstacle truth snapshots were numerically checked against an independent same-seed generator before extending truth through the 24 s forecast. These offline obstacle errors are distinct from vehicle execution error and never enter planning.',
             '- Planner P95 values are actual wall-clock timers under up to three concurrent study processes and other shared-machine activity. Observer and offline diagnostics are excluded from those timers. Physics pauses during planning; measured latency was not injected as execution delay. These values are not isolated real-time guarantees.',
             '- The results identify baseline limitations without adding ECP, reinforcement learning, or a new control policy. Collision thresholds, acceptance rules, controller parameters, and stored original experiments were not relaxed.',
             '', '## Files', '',
             '- `budget_comparison.csv`: every requested actual condition/budget and safety, interaction, latency, reference-response, vehicle-error and obstacle-error metrics.',
             '- `failure_event_comparison.csv`: all failure/collision-context events, full-grid counts and 9/18/45 shortlist membership evaluated at the same captured state.',
             '- `candidate_grid_diagnostics.csv`: individual references, fast and physical costs/clearances, feasibility and numerical failure evidence.',
             '- `budget_totals.json`: per-budget arrival/acceptance totals and pooled actual planner P95, plus ranges of actual control-reference and actuator response errors.',
             '- Per-case `failure_diagnostics/event_*.json`: complete captured plant/controller state, causal tracks and all 45 evaluations. Original state/control/planning/error CSV, selected plans, obstacles and PNGs remain alongside them.',
             '- Per-case `actual_margin_diagnostics/`: every violating actual physical interval, contiguous groups and clearance plot.',
             '- Per-case `obstacle_forecast_errors.csv` and summary: same-ID/time errors, ranges, ages and 2 s/24 s RMSE/max, separate from actual-control response.', '']
    (report/'FAILURE_ANALYSIS.md').write_text('\n'.join(text),encoding='utf-8')
    (report/'report_manifest.json').write_text(json.dumps({'created_at':now(),'run_count':len(rows),
        'source_study_manifest':str(out/'budget_study.json'),'run_manifest_finished_at':manifest['finished_at'],
        'controller_restored':False,'new_simulation_performed_during_summarization':False,
        'raw_experiment_files_modified':False,'failure_event_count':len(event_rows),
        'candidate_diagnostic_rows':len(candidate_rows)},indent=2),encoding='utf-8')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='results_stage2/review_20261011_budget')
    parser.add_argument('--workers',type=int,default=1,choices=(1,2,3))
    parser.add_argument('--cases',nargs='+',choices=tuple(CASE_SPECS),default=list(CASE_SPECS))
    parser.add_argument('--budgets',nargs='+',type=int,choices=(9,18,45),default=[9,18,45])
    parser.add_argument('--single',action='store_true')
    parser.add_argument('--summarize-only',action='store_true')
    parser.add_argument('--case',choices=tuple(CASE_SPECS))
    parser.add_argument('--budget',type=int,choices=(9,18,45),default=9)
    args=parser.parse_args(argv)
    out=Path(args.out).absolute()
    if args.summarize_only:
        print(summarize_study(out),flush=True)
        return 0
    if args.single:
        if not args.case:
            parser.error('--single requires --case')
        single_case(args.case,args.budget,out,args.workers)
        return 0
    if out.exists():
        parser.error('study output root already exists; choose a new directory')
    out.mkdir(parents=True,exist_ok=False)
    (out/'logs').mkdir()
    # Prioritize all five baseline budgets, then the static/terrain full-grid
    # runs; every remaining requested pair is still executed in this new study.
    pairs=[(case,9) for case in args.cases if 9 in args.budgets]
    pairs += [(case,45) for case in args.cases if case in ('static3d','terrain3d') and 45 in args.budgets]
    pairs += [(case,budget) for budget in args.budgets for case in args.cases if (case,budget) not in pairs]
    manifest={'started_at':now(),'finished_at':None,'actual_batch_argv':[sys.executable,*sys.argv],
              'purpose':'new actual REMUS experiments; no old experiment reused as a new run',
              'case_count':len(pairs),'completed_count':0,'worker_limit':args.workers,
              'latency_note':'actual observed planning latency under concurrent CPU load; not isolated hardware benchmarking',
              'physics_dt_s':.02,'control_dt_s':2.,'horizon_s':24.,'seed':7,
              'records':[]}
    manifest_path=out/'budget_study.json'
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(run_child,case,budget,out,args.workers):(case,budget) for case,budget in pairs}
        for future in as_completed(jobs):
            case,budget=jobs[future]
            try:
                record=future.result()
            except Exception:
                record={'case':case,'budget':budget,'exit_code':-1,'error':traceback.format_exc()}
            manifest['records'].append(record)
            manifest['completed_count']=len(manifest['records'])
            manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
            summary=record.get('closed_loop_summary') or {}
            print(f'DONE {manifest["completed_count"]}/{len(pairs)} {case} budget={budget} '
                  f'status={summary.get("status")} exit={record["exit_code"]} at {now()}',flush=True)
    manifest['finished_at']=now()
    manifest['process_failure_count']=sum(row['exit_code']!=0 for row in manifest['records'])
    manifest_path.write_text(json.dumps(manifest,indent=2,allow_nan=False),encoding='utf-8')
    summarize_study(out)
    print(f'STUDY_FINISHED {manifest_path}',flush=True)
    return 1 if manifest['process_failure_count'] else 0


if __name__=='__main__':
    raise SystemExit(main())
