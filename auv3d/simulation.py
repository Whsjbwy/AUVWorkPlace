from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import json, platform, time
import numpy as np
from .config import ExperimentConfig
from .geometry import kinematic_step,swept_sphere_clearance
from .terrain import HeightField
from .obstacles import LinearObstacles,PIObstacles
from .sensors import IdealObjectSonar,TrackMemory
from .planner import DWA3D

SCENARIOS=('empty3d','static3d','crossing3d','paper_random3d')

def make_scenario(name,cfg):
    start=np.array([60.,80.,35.,1.,0.,0.]);goal=np.array([240.,130.,55.])
    if name=='paper_random3d':
        terrain=HeightField.perlin(cfg.terrain,cfg.seed)
        start=np.array([60.,60.,25.,1.,0.,np.pi/6]);goal=np.array([340.,270.,30.])
        obstacles=PIObstacles(cfg.obstacles,terrain,cfg.seed+1,exclude=(start[:3],goal))
    else:
        terrain=HeightField.flat(cfg.terrain.width,cfg.terrain.height,cfg.terrain.max_depth)
        if name=='empty3d':obstacles=LinearObstacles([],[],[])
        elif name=='static3d':obstacles=LinearObstacles([[142.,101.,44.]],[[0,0,0]],[5.])
        elif name=='crossing3d':obstacles=LinearObstacles([[150.,123.,41.]],[[0.,-.45,.06]],[5.])
        else:raise ValueError(f'unknown scenario {name}')
    for point in (start[:3],goal):
        if terrain.clearance(point,cfg.vehicle.body_radius_m)<=0:raise ValueError('invalid start/goal in terrain')
    return terrain,obstacles,start,goal

def run_episode(cfg:ExperimentConfig,scenario='empty3d',predictor=None,calibration=None):
    cfg.validate();terrain,obstacles,state,goal=make_scenario(scenario,cfg)
    sensor=IdealObjectSonar(cfg.sonar,cfg.seed+11);memory=TrackMemory(cfg.planner.track_max_age_s)
    planner=DWA3D(cfg,predictor,calibration)
    dt=cfg.planner.integration_dt;control_n=round(cfg.planner.control_dt/dt);obs_n=round(cfg.sonar.period_s/dt)
    if abs(obs_n*dt-cfg.sonar.period_s)>1e-9:raise ValueError('sonar period must align with integration dt')
    total_steps=round(cfg.planner.max_time_s/dt);command=np.array([state[3],0.,0.])
    logs=[];planlogs=[];snapshots=[];selected_plans=[];status='timeout';actual_min=np.inf;path_length=0.;effort=0.
    t_start=time.perf_counter();emergencies=0;fallbacks=0
    for step in range(total_steps+1):
        t=step*dt;snapshot=obstacles.snapshot()
        if step%obs_n==0:
            eta=np.array([*state[:3],0.,state[4],state[5]])
            observations=sensor.observe(eta,t,snapshot,terrain);memory.update(observations,t)
        if step%control_n==0:
            plan=planner.plan(state,goal,terrain,memory.current(t),t)
            command=plan.command;emergencies+=int(not plan.feasible);fallbacks+=plan.prediction_fallback_count
            planlogs.append([t,plan.planning_ms,plan.candidate_count,plan.valid_count,
                             plan.minimum_predicted_clearance_m,len(memory.current(t)),int(plan.feasible)])
            snapshots.append({'time_s':t,'ids':snapshot.ids.tolist(),'positions':snapshot.positions.tolist(),'radii':snapshot.radii.tolist()})
            selected_plans.append({'time_s':t,'command':plan.command.tolist(),
                                   'states':plan.states.tolist(),'feasible':plan.feasible})
        logs.append([t,*state,*command])
        if np.linalg.norm(state[:3]-goal)<=cfg.planner.goal_tolerance_m:
            status='success';break
        if step==total_steps:break
        old=state.copy();obstacles.step(dt);new_snapshot=obstacles.snapshot()
        state=kinematic_step(state,command,dt,cfg.current_ned_mps)
        path_length+=float(np.linalg.norm(state[:3]-old[:3]));effort+=float(command[0]**3*dt)
        c=terrain.segment_clearance(old[:3],state[:3],cfg.vehicle.body_radius_m)
        c=min(c,cfg.vehicle.depth_limit_m-cfg.vehicle.body_radius_m-state[2])
        actual_min=min(actual_min,c)
        if c<=0:
            status='terrain_or_domain_collision';logs.append([(step+1)*dt,*state,*command]);break
        if len(snapshot.ids):
            c=float(np.min(swept_sphere_clearance(old[:3],state[:3],snapshot.positions,new_snapshot.positions,
                                                snapshot.radii+cfg.vehicle.body_radius_m)))
            actual_min=min(actual_min,c)
            if c<=0:
                status='obstacle_collision';logs.append([(step+1)*dt,*state,*command]);break
    elapsed=time.perf_counter()-t_start
    lat=np.asarray(planlogs)[:,1]
    summary={'stage':'engineering_validation_not_paper_result','backend':'p25_kinematic_3d_rk4',
       'scenario':scenario,'seed':cfg.seed,'status':status,'reached_goal':status=='success',
       'duration_s':float(logs[-1][0]),'path_length_m':path_length,'goal_distance_m':float(np.linalg.norm(state[:3]-goal)),
       'minimum_model_clearance_m':float(actual_min),'emergency_deceleration_count':emergencies,
       'stale_learned_prediction_fallback_count':fallbacks,'predictor':planner.predictor.name,
       'calibration':'empirical_speed_bin' if calibration is not None else 'uncalibrated_debug_margin',
       'surge_cubic_effort_proxy':effort,'energy_joules':None,
       'energy_note':'u^3 integral is a proxy, NOT electrical energy; no calibrated power coefficients supplied',
       'planning_ms_median':float(np.median(lat)),'planning_ms_p95':float(np.percentile(lat,95)),
       'planning_ms_max':float(lat.max()),'wall_clock_s':elapsed,'python':platform.python_version(),
       'numpy':np.__version__,'model_change':'none; paper kinematics implemented separately',
       'configuration_label':cfg.label,
       'failure_reason':None if status=='success' else {
           'timeout':'goal tolerance was not reached before max_time_s',
           'terrain_or_domain_collision':'swept trajectory crossed terrain, surface or configured domain',
           'obstacle_collision':'time-aligned swept enclosing spheres overlap'}[status],
       'execution_model':'P25 simplified kinematics; no REMUS actuator tracking',
       'obstacle_boundary_events':getattr(obstacles,'boundary_events',0)}
    return {'summary':summary,'states':np.asarray(logs),'planning':np.asarray(planlogs),'goal':goal,
            'terrain':terrain,'obstacles':snapshots,'selected_plans':selected_plans,'config':cfg.to_dict()}

def save_episode(result,out,overwrite=False,plot=True):
    out=Path(out)
    if out.exists() and any(out.iterdir()) and not overwrite:raise FileExistsError(f'{out} not empty; use --overwrite')
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result['summary'],indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'config.json').write_text(json.dumps(result['config'],indent=2),encoding='utf-8')
    np.savetxt(out/'trajectory.csv',result['states'],delimiter=',',header='time_s,north_m,east_m,depth_m,u_mps,pitch_rad,yaw_rad,command_u_mps,command_q_radps,command_r_radps',comments='')
    np.savetxt(out/'planning.csv',result['planning'],delimiter=',',header='time_s,planning_ms,candidates,valid_candidates,predicted_clearance_m,tracked_objects,feasible',comments='')
    (out/'obstacles.json').write_text(json.dumps(result['obstacles']),encoding='utf-8')
    (out/'selected_plans.json').write_text(json.dumps(result['selected_plans']),encoding='utf-8')
    np.savez_compressed(out/'terrain.npz',xs=result['terrain'].xs,ys=result['terrain'].ys,depths=result['terrain'].depths)
    if plot:
        from .visualization import plot_episode
        plot_episode(result,out)
