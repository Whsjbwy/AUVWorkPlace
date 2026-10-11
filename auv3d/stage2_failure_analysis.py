"""Read-only offline diagnostics for a captured REMUS budget experiment.

Snapshots are cloned at the actual planning call, including controller memory.
Obstacle future truth is generated only after execution for error assessment;
it is never supplied to the planner or used to revise an executed decision.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .adapters.remus_guidance import clone_plant
from .prediction import CAPredictor
from .scenarios import make_stage2_scenario
from .remus_planner import evaluate_reference_grid


def _json_value(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {str(k):_json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    raise TypeError(f'unsupported captured state field {type(value).__name__}')


def captured_plant_record(plant):
    """Complete ordinary plant/vehicle data for evidence, not a CSV inference."""
    return {'time_s':float(plant.time_s), 'eta':plant.eta.tolist(),
            'nu':plant.nu.tolist(), 'actual':plant.actual.tolist(),
            'vehicle_fields':_json_value(vars(plant.vehicle)),
            'capture_note':'full in-memory clone at planning time; offline evaluation uses this clone, not a restored CSV trajectory'}


@dataclass
class PlanningCapture:
    time_s: float
    plant: object
    tracks: list
    previous_action: object
    action: object
    feasible: bool
    emergency_verified_safe: bool
    reason: str
    selected_clearance_m: float
    planning_ms: float
    candidate_evaluations: tuple
    shortlist_indices: dict


class FailureObserver:
    """Optional on_plan observer; it copies data and has no decision authority."""
    def __init__(self,cfg,logger=None):
        self.cfg=cfg
        self.frames=[]
        self.previous_action=None
        self.logger=logger

    def __call__(self,plant,tracks,now,plan,scene):
        captured=PlanningCapture(
            float(now),clone_plant(plant),copy.deepcopy(list(tracks)),
            copy.deepcopy(self.previous_action),copy.deepcopy(plan.action),bool(plan.feasible),
            bool(plan.emergency_verified_safe),str(plan.reason),
            float(plan.minimum_predicted_clearance_m),float(plan.planning_ms),
            tuple(copy.deepcopy(getattr(plan,'candidate_evaluations',()))),
            copy.deepcopy(getattr(plan,'shortlist_indices',{})))
        self.frames.append(captured)
        self.previous_action=copy.deepcopy(plan.action)
        if self.logger:
            self.logger(f'PLAN t={now:.2f}s feasible={plan.feasible} '
                        f'clearance={plan.minimum_predicted_clearance_m:.6f} '
                        f'tracks={len(tracks)} verified={plan.verified_count} '
                        f'planning_ms={plan.planning_ms:.3f}')


def run_with_observer(cfg,scenario,observer):
    """Use the explicit read-only simulation observer interface."""
    from .stage2_simulation import run_remus_episode
    return run_remus_episode(cfg,scenario,on_plan=observer)


def classify_failure(grid,emergency_verified_safe):
    """Separate a shortlist miss from absence in the finite tested grid."""
    feasible=int(grid['feasible_count'])
    numerical=sum(bool(row.get('prediction_failure')) for row in grid['candidates'])
    if feasible:
        category='shortlist_omission'
    elif numerical:
        category='no_verified_reference_with_numerical_prediction_failures'
    else:
        category='no_safe_reference_in_current_discrete_grid'
    return {'category':category, 'full_grid_feasible_count':feasible,
            'full_grid_prediction_failure_count':numerical,
            'unsafe_emergency':not bool(emergency_verified_safe),
            'scope':'same causal CA forecast and full REMUS model; this finite reference grid does not prove global unavoidability'}


def _full_grid_from_cached(frame,cfg):
    rows=list(frame.candidate_evaluations)
    if len(rows)!=45 or len({int(r['index']) for r in rows})!=45:
        return None
    feasible=[row for row in rows if row['feasible']]
    best=min(feasible,key=lambda r:r['cost']) if feasible else None
    return {'candidate_count':len(rows), 'feasible_count':len(feasible),
            'evaluation_ms':frame.planning_ms, 'now_s':frame.time_s,
            'evaluation_timing_scope':'actual full planner timer including screening and any emergency rollout; candidate rollouts are reused',
            'horizon_s':cfg.horizon_s, 'shortlist_indices':frame.shortlist_indices,
            'best_feasible_index':best['index'] if best else None,
            'reason':'live full-grid evaluations reused without another physics integration',
            'candidates':rows}


def diagnose_failures(result,observer,out,logger=None):
    """Evaluate failed planning states using the exact captured controller memory."""
    out=Path(out)
    cfg=observer.cfg
    frames=[(frame,'no_feasible_shortlisted_reference') for frame in observer.frames if not frame.feasible]
    if 'collision' in result['summary']['status'] and observer.frames and observer.frames[-1].feasible:
        frames.append((observer.frames[-1],'collision_after_feasible_selected_plan'))
    evaluations=[]
    started=time.perf_counter()
    directory=out/'failure_diagnostics'
    directory.mkdir(exist_ok=False)
    for index,(frame,trigger) in enumerate(frames):
        evaluation=_full_grid_from_cached(frame,cfg)
        source='reused_actual_live_45_reference_evaluations'
        if evaluation is None:
            evaluation=evaluate_reference_grid(frame.plant,result['goal'],result['terrain'],frame.tracks,
                                               frame.time_s,cfg,previous_action=frame.previous_action)
            source='offline_45_reference_rollouts_from_captured_full_controller_state'
        classification=classify_failure(evaluation,frame.emergency_verified_safe)
        if trigger!='no_feasible_shortlisted_reference':
            classification['category']='collision_context_full_grid_evaluation'
            classification['unsafe_emergency']=False
        record={'time_s':frame.time_s,'trigger':trigger,'budget':cfg.candidate_budget,
                'selected_action':frame.action.as_array().tolist(),'selected_feasible':frame.feasible,
                'selected_prediction_clearance_m':frame.selected_clearance_m,
                'emergency_verified_safe':frame.emergency_verified_safe,
                'selected_reason':frame.reason,'evaluation_source':source,
                **classification,'grid':evaluation,
                'captured_plant':captured_plant_record(frame.plant),
                'previous_action':frame.previous_action.as_array().tolist() if frame.previous_action else None,
                'tracks':[{'object_id':t.object_id,'radius_m':t.radius_m,'times':t.times,
                           'positions':np.asarray(t.positions).tolist()} for t in frame.tracks]}
        path=directory/f'event_{index:03d}_t{frame.time_s:07.2f}.json'
        path.write_text(json.dumps(_json_value(record),indent=2,allow_nan=False),encoding='utf-8')
        evaluations.append(record)
        if logger:
            logger(f'DIAG t={frame.time_s:.2f}s category={classification["category"]} '
                   f'full_grid_feasible={evaluation["feasible_count"]} '
                   f'unsafe_emergency={classification["unsafe_emergency"]} source={source}')
    columns=['time_s','budget','full_grid_feasible_count','full_grid_prediction_failure_count',
             'unsafe_emergency','selected_prediction_clearance_m','full_grid_evaluation_ms']
    rows=[[r['time_s'],r['budget'],r['full_grid_feasible_count'],r['full_grid_prediction_failure_count'],
           int(r['unsafe_emergency']),r['selected_prediction_clearance_m'],r['grid']['evaluation_ms']] for r in evaluations]
    np.savetxt(out/'failure_classification.csv',np.asarray(rows).reshape(-1,len(columns)),delimiter=',',
               header=','.join(columns),comments='')
    categories={name:sum(r['category']==name for r in evaluations) for name in sorted({r['category'] for r in evaluations})}
    summary={'diagnosed_frame_count':len(evaluations),'categories':categories,
             'unsafe_emergency_frames':sum(not f.feasible and not f.emergency_verified_safe for f in observer.frames),
             'offline_diagnostic_wall_clock_s':time.perf_counter()-started,
             'diagnostic_scope':'offline same-forecast finite-grid diagnosis; no executed action is modified',
             'full_state_capture':'actual in-memory plant/controller/reference/actuator clones at on_plan callback',
             'records':[{'time_s':r['time_s'],'trigger':r['trigger'],'category':r['category'],
                         'full_grid_feasible_count':r['full_grid_feasible_count'],
                         'unsafe_emergency':r['unsafe_emergency'], 'evaluation_source':r['evaluation_source']}
                        for r in evaluations]}
    (out/'failure_analysis_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    _plot_failure_grid(evaluations,out)
    return summary


def _plot_failure_grid(evaluations,out):
    fig,ax=plt.subplots(figsize=(11,max(3.,min(10.,len(evaluations)*.22+2))))
    if evaluations:
        values=np.full((len(evaluations),45),np.nan)
        for i,record in enumerate(evaluations):
            for row in record['grid']['candidates']:
                value=row.get('minimum_predicted_clearance_m')
                if value is not None and np.isfinite(value):
                    values[i,int(row['index'])]=value
        image=ax.imshow(values,aspect='auto',cmap='RdYlGn',vmin=-5,vmax=5,origin='lower')
        selected=np.unique(np.linspace(0,len(evaluations)-1,min(12,len(evaluations)),dtype=int))
        ax.set_yticks(selected,[f'{evaluations[i]["time_s"]:g}' for i in selected])
        fig.colorbar(image,ax=ax,label='Predicted enclosing-sphere clearance (m); colors clipped to +/-5 m')
    else:
        ax.text(.5,.5,'No failed planning frames or collision context to diagnose',ha='center',transform=ax.transAxes)
    ax.set(xlabel='Reference grid candidate index (full detail in event JSON)',ylabel='Captured planning time (s)',
           title='Offline full-grid diagnosis; causal obstacle forecast, original REMUS 6-DOF\nStored candidate data retain full numerical clearance values')
    fig.tight_layout()
    fig.savefig(out/'failure_grid.png',dpi=150)
    plt.close(fig)


def analyze_actual_margin(result,cfg,out):
    """Report every swept physical interval below margin, even without fallback.

    The minimum is evaluated over each physical interval. Its recorded end
    time does not assert that the minimum occurred exactly at that endpoint.
    This is an execution assessment; no controller state is reconstructed.
    """
    out=Path(out)
    directory=out/'actual_margin_diagnostics'
    directory.mkdir(exist_ok=False)
    clearance=np.asarray(result['clearance'],dtype=float).reshape(-1,3)
    planning=np.asarray(result['planning'],dtype=float).reshape(-1,9)
    states=np.asarray(result['states'],dtype=float).reshape(-1,16)
    if not len(clearance) or not np.isfinite(clearance).all():
        raise ValueError('actual swept clearance log must be nonempty and finite')
    if np.any(np.diff(clearance[:,0])<=0.):
        raise ValueError('actual swept clearance intervals must have increasing end times')
    diagnostics={float(row['time_s']):row for row in result.get('planning_diagnostics',[])}
    rows=[]
    violating=np.flatnonzero(clearance[:,1]<cfg.safety_margin_m)
    emergency_intervals=0
    for index in violating:
        end=float(clearance[index,0])
        start=float(clearance[index-1,0]) if index else max(0.,end-cfg.physics_dt)
        plan_index=int(np.searchsorted(planning[:,0],start+1e-9,side='right')-1)
        if plan_index<0:
            raise ValueError('an actual margin violation has no preceding planning record')
        plan=planning[plan_index]
        feasible=bool(plan[7])
        emergency_intervals+=not feasible
        detail=diagnostics.get(float(plan[0]),{})
        position=[float(np.interp(end,states[:,0],states[:,column])) for column in (1,2,3)]
        rows.append([start,end,float(clearance[index,1]),float(clearance[index,1]-cfg.safety_margin_m),
                     float(plan[0]),int(feasible),int(not feasible),
                     int(bool(detail.get('emergency_verified_safe',False))),*position])
    columns=['interval_start_s','interval_end_s','swept_actual_clearance_m',
             'clearance_minus_safety_margin_m','preceding_plan_time_s',
             'preceding_plan_feasible','preceding_plan_emergency',
             'preceding_emergency_verified_safe','endpoint_north_m','endpoint_east_m','endpoint_depth_m']
    np.savetxt(directory/'violating_intervals.csv',np.asarray(rows).reshape(-1,len(columns)),
               delimiter=',',header=','.join(columns),comments='')
    groups=[]
    for group in np.split(violating,np.flatnonzero(np.diff(violating)>1)+1) if len(violating) else []:
        first,last=int(group[0]),int(group[-1])
        groups.append({'first_interval_start_s':float(clearance[first-1,0]) if first else max(0.,float(clearance[first,0])-cfg.physics_dt),
                       'last_interval_end_s':float(clearance[last,0]),'interval_count':len(group),
                       'minimum_swept_clearance_m':float(clearance[group,1].min())})
    interval_total=sum(row[1]-row[0] for row in rows)
    summary={'safety_margin_m':cfg.safety_margin_m,'actual_margin_met':len(violating)==0,
             'minimum_swept_actual_clearance_m':float(clearance[:,1].min()),
             'violating_physics_interval_count':len(violating),
             'violating_intervals_after_feasible_plan':len(violating)-emergency_intervals,
             'violating_intervals_after_emergency_plan':emergency_intervals,
             'collision_physics_interval_count':int(np.count_nonzero(clearance[:,1]<=0.)),
             'violating_interval_total_duration_s':float(interval_total),'contiguous_interval_groups':groups,
             'time_convention':'each CSV row is an interval [start,end] whose swept minimum is below the unchanged configured margin; minimum time within that interval is unknown',
             'duration_convention':'sum of durations of intervals containing a violation; not exact continuous time spent below threshold',
             'control_context':f'preceding chosen decision at interval start; configured command_delay_s={cfg.command_delay_s:g}; a positive delay means that this decision may not yet be the applied reference',
             'source':'actual recorded clearance.csv physical-interval swept collision evaluator',
             'controller_state_reconstructed':False}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot(clearance[:,0],clearance[:,1],label='REMUS actual swept interval clearance',linewidth=1.)
    ax.axhline(cfg.safety_margin_m,color='orange',linestyle='--',label=f'Required safety margin {cfg.safety_margin_m:g} m')
    ax.axhline(0.,color='red',linestyle=':',label='Collision boundary')
    if len(violating):
        ax.scatter(clearance[violating,0],clearance[violating,1],s=8,color='red',label='Actual margin violation interval')
    ax.set(xlabel='Physical interval end time (s)',ylabel='Minimum swept clearance (m)',
           title='Actual execution safety margin; independent of emergency classification')
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory/'actual_clearance.png',dpi=150)
    plt.close(fig)
    return summary


def analyze_saved_actual_margin(source):
    """Derive margin diagnostics from a completed run without replaying a plant."""
    from .stage2_config import Stage2Config
    source=Path(source)
    cfg=Stage2Config(**json.loads((source/'config.json').read_text(encoding='utf-8')))
    result={name:np.loadtxt(source/f'{name}.csv',delimiter=',',skiprows=1,ndmin=2)
            for name in ('clearance','planning','states')}
    result['planning_diagnostics']=json.loads((source/'planning_diagnostics.json').read_text(encoding='utf-8'))
    return analyze_actual_margin(result,cfg,source)


def _truth_at_times(cfg,scenario,ticks,recorded_snapshots):
    scene=make_stage2_scenario(scenario,cfg)
    requested=set(ticks)
    recorded={round(float(row['time_s'])/cfg.physics_dt):row for row in recorded_snapshots}
    maximum=max(requested|set(recorded),default=0)
    truth={}
    difference=0.
    for tick in range(maximum+1):
        if tick in requested or tick in recorded:
            snapshot=scene.obstacles.snapshot()
            if tick in requested:
                truth[tick]=snapshot
            if tick in recorded:
                row=recorded[tick]
                positions=np.asarray(row['positions'],dtype=float).reshape(-1,3)
                radii=np.asarray(row['radii'],dtype=float)
                if not np.array_equal(snapshot.ids,np.asarray(row['ids'])) or not np.allclose(snapshot.positions,positions,rtol=1e-10,atol=1e-8) or not np.allclose(snapshot.radii,radii,rtol=1e-10,atol=1e-8):
                    raise ValueError('offline obstacle truth differs from an actual recorded snapshot')
                if positions.size:
                    difference=max(difference,float(np.max(np.abs(snapshot.positions-positions))))
        if tick<maximum:
            scene.obstacles.step(cfg.physics_dt)
    return truth,{'verified_recorded_truth_snapshots':len(recorded),
                  'maximum_recorded_truth_position_difference_m':difference,
                  'offline_truth_advanced_through_s':maximum*cfg.physics_dt}


def _error_statistics(rows):
    rows=np.asarray(rows,dtype=float).reshape(-1,16)
    values=rows[:,-1]
    result={'sample_count':len(rows),'rmse_m':float(np.sqrt(np.mean(values**2))) if len(rows) else None,
            'max_error_m':float(values.max()) if len(rows) else None}
    for horizon in (2.,24.):
        selected=rows[np.isclose(rows[:,5],horizon,rtol=0.,atol=1e-9)]
        error=selected[:,-1]
        result[f'at_{horizon:g}s']={'sample_count':len(selected),
                                   'rmse_m':float(np.sqrt(np.mean(error**2))) if len(error) else None,
                                   'max_error_m':float(error.max()) if len(error) else None}
    return result


def analyze_obstacle_forecasts(result,observer,out):
    """Compare CA forecasts with same-time obstacle truth, separately from AUV error."""
    cfg=observer.cfg
    offsets=np.linspace(0.,cfg.horizon_s,round(cfg.horizon_s/cfg.prediction_output_dt)+1)
    ticks={round((frame.time_s+offset)/cfg.physics_dt) for frame in observer.frames for offset in offsets}
    truth,audit=_truth_at_times(cfg,result['summary']['scenario'],ticks,result['obstacles'])
    predictor=CAPredictor(cfg.observation_period_s)
    states=result['states']
    rows=[]
    for frame in observer.frames:
        for track in frame.tracks:
            predicted=predictor.predict(track,frame.time_s+offsets)
            observation_time=track.times[-1]
            observation_auv=np.array([np.interp(observation_time,states[:,0],states[:,column]) for column in (1,2,3)])
            observed_range=float(np.linalg.norm(track.positions[-1]-observation_auv))
            for offset,point in zip(offsets,predicted):
                target_time=frame.time_s+offset
                snapshot=truth[round(target_time/cfg.physics_dt)]
                found=np.flatnonzero(snapshot.ids==track.object_id)
                if len(found)!=1:
                    raise ValueError('tracked obstacle ID is absent/ambiguous in offline ground truth')
                actual=snapshot.positions[found[0]]
                error=float(np.linalg.norm(point-actual))
                rows.append([frame.time_s,track.object_id,observation_time,frame.time_s-observation_time,
                             observed_range,offset,target_time,*point,*actual,
                             float(np.linalg.norm(point[:2]-actual[:2])),float(point[2]-actual[2]),error])
    rows=np.asarray(rows,dtype=float).reshape(-1,16)
    columns=['plan_time_s','object_id','last_observation_time_s','observation_age_s','range_at_observation_m',
             'offset_s','target_time_s','predicted_north_m','predicted_east_m','predicted_depth_m',
             'truth_north_m','truth_east_m','truth_depth_m','horizontal_error_m','signed_depth_error_m','position_error_m']
    out=Path(out)
    np.savetxt(out/'obstacle_forecast_errors.csv',rows,delimiter=',',header=','.join(columns),comments='')
    by_id={str(int(object_id)):_error_statistics(rows[rows[:,1]==object_id]) for object_id in np.unique(rows[:,1])}
    ranges={}
    for low,high in zip((0.,30.,60.,90.,150.),(30.,60.,90.,150.,float('inf'))):
        selected=rows[(rows[:,4]>=low)&(rows[:,4]<high)]
        ranges[f'{low:g}_to_{high:g}_m']=_error_statistics(selected)
    summary={**_error_statistics(rows),'by_object_id':by_id,'by_observation_range':ranges,**audit,
             'scope':'CA predicted obstacle positions vs the same scenario ID at exactly the same target simulation time',
             'vehicle_prediction_error_included':False,
             'future_truth_used_by_planner':False,
             'forecast_sampling_dt_s':cfg.prediction_output_dt,
             'truth_note':'independent same-seed obstacle generator continued beyond episode end where needed; offline evaluation only'}
    (out/'obstacle_forecast_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for ax,horizon in zip(axes,(2.,24.)):
        selected=rows[np.isclose(rows[:,5],horizon,rtol=0.,atol=1e-9)]
        for object_id in np.unique(selected[:,1]):
            item=selected[selected[:,1]==object_id]
            ax.plot(item[:,0],item[:,-1],marker='.',linewidth=.8,label=f'Obstacle ID {int(object_id)}')
        ax.set_ylabel(f'{horizon:g}s position error (m)')
        ax.grid(alpha=.25)
        if len(np.unique(selected[:,1]))<=12 and len(selected):
            ax.legend(fontsize=7)
    axes[0].set_title('Offline obstacle CA forecast errors; distinct from REMUS execution-prediction errors')
    axes[-1].set_xlabel('Planning simulation time (s)')
    fig.tight_layout()
    fig.savefig(out/'obstacle_forecast_error.png',dpi=150)
    plt.close(fig)
    return summary
