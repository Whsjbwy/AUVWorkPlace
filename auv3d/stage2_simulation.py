"""REMUS state-feedback DWA execution; the plant is always the original 6-DOF model."""
import copy
import json
from pathlib import Path
import time

import numpy as np

from .adapters.remus_guidance import RemusGuidance, ReferenceAction, clone_plant
from .geometry import swept_sphere_clearance, wrap_pi
from .remus import RemusPlant
from .remus_planner import RemusDWA
from .scenarios import make_stage2_scenario
from .sensors import IdealObjectSonar, TrackMemory
from .stage2_config import Stage2Config

STATE_COLUMNS = ['time_s','north_m','east_m','depth_m','roll_rad','pitch_rad','yaw_rad',
                 'u_mps','v_mps','w_mps','p_radps','q_radps','r_radps',
                 'actual_rudder_rad','actual_stern_rad','actual_rpm']
CONTROL_COLUMNS = ['time_s','ref_depth_m','ref_heading_rad','ref_rpm',
                   'raw_rudder_rad','raw_stern_rad','raw_rpm',
                   'applied_rudder_rad','applied_stern_rad','applied_rpm',
                   'actual_rudder_rad','actual_stern_rad','actual_rpm',
                   'saturated_rudder','saturated_stern','saturated_rpm']
ERROR_COLUMNS = ['time_s','plan_time_s','elapsed_s','position_error_m','depth_error_m',
                 'heading_error_rad','surge_error_mps']
PLANNING_COLUMNS = ['time_s','planning_ms','candidates','verified_candidates','valid_candidates',
                    'predicted_clearance_m','tracked_objects','feasible','fallback_count']


def initialize_plant(scene, cfg):
    """Documented steady-cruise initial condition, not a hidden ideal-state executor."""
    eta = np.asarray(scene.start_eta, dtype=float)
    plant = RemusPlant(depth=float(eta[2]), heading_deg=float(np.rad2deg(eta[5])),
                       rpm=cfg.nominal_rpm, current_speed=cfg.current_speed,
                       current_direction_deg=cfg.current_direction_deg,
                       eta0=eta, initialize_references=True)
    guidance = RemusGuidance()
    reference = ReferenceAction(float(eta[2]), float(eta[5]), cfg.nominal_rpm)
    for _ in range(round(cfg.warmup_s / cfg.physics_dt)):
        guidance.step(plant, reference, cfg.physics_dt)
    # Preconditioning is position-invariant in this constant-current model.
    # Translate back to the scenario start; retain all velocities, roll/pitch,
    # controller/filter/integrator and actuator states from the actual warmup.
    plant.eta[:3] = eta[:3]
    plant.time_s = 0.0
    return plant


def run_remus_episode(cfg=None, scenario='crossing3d', *, on_plan=None):
    """Run the actual plant; optional diagnostics receive isolated copies only.

    ``on_plan(plant, tracks, now, plan, scene)`` is an offline observation hook.
    Its return value is ignored and it cannot change live controller/obstacle
    state. Wall time spent in this hook is excluded from planner latency.
    """
    cfg = cfg or Stage2Config()
    cfg.validate()
    scene = make_stage2_scenario(scenario, cfg)
    terrain, obstacles, goal = scene.terrain, scene.obstacles, np.asarray(scene.goal)
    plant = initialize_plant(scene, cfg)
    sonar_cfg = cfg.sensor_config()
    sonar = IdealObjectSonar(sonar_cfg, cfg.seed)
    memory = TrackMemory(cfg.track_max_age_s)
    guidance, planner = RemusGuidance(), RemusDWA(cfg)
    dt = cfg.physics_dt
    control_n, obs_n = round(cfg.control_dt/dt), round(cfg.observation_period_s/dt)
    total = round(cfg.max_time_s/dt)
    state_logs, controls, planning, snapshots, plans, errors, events = [], [], [], [], [], [], []
    plan_diagnostics, observation_log, clearance_log = [], [], []
    pending = []
    pending_commands = []
    actual_min = float('inf')
    interacted = set()
    observed_unique = set()
    maximum_observed = maximum_tracked = maximum_close = 0
    maximum_observation_age = 0.
    observation_counts = []
    status = 'timeout'
    plan = None
    plan_time = 0.0
    executing_plan = None
    action = ReferenceAction(float(plant.eta[2]), float(plant.eta[5]), cfg.nominal_rpm)
    started = time.perf_counter()
    path_length = 0.0
    failure_reason = 'goal tolerance was not reached before max_time_s'
    for step in range(total+1):
        t = step*dt
        snapshot = obstacles.snapshot()
        if len(snapshot.ids):
            distances = np.linalg.norm(snapshot.positions-plant.eta[:3], axis=1)-snapshot.radii-cfg.body_radius_m
            actual_min = min(actual_min, float(np.min(distances)))
            interacted.update(int(i) for i in snapshot.ids[distances <= cfg.interaction_distance_m])
            maximum_close = max(maximum_close,int(np.sum(distances <= cfg.interaction_distance_m)))
        state_logs.append([t, *plant.eta, *plant.nu, *plant.actual])
        terrain_c = min(float(terrain.clearance(plant.eta[:3],cfg.body_radius_m)),
                        cfg.depth_limit_m-cfg.body_radius_m-plant.eta[2])
        actual_min = min(actual_min, terrain_c)
        if terrain_c <= 0 or (len(snapshot.ids) and np.min(distances) <= 0):
            status = 'terrain_or_domain_collision' if terrain_c <= 0 else 'obstacle_collision'
            failure_reason = 'initial/current actual enclosing sphere overlaps terrain/domain or obstacle'
            events.append({'time_s':t,'kind':'collision','position':plant.eta[:3].tolist(),'reason':failure_reason})
            break
        if np.linalg.norm(plant.eta[:3]-goal) <= cfg.goal_tolerance_m:
            status, failure_reason = 'success', None
            break
        if step == total:
            break
        if step % obs_n == 0:
            observations = sonar.observe(plant.eta, t, snapshot, terrain)
            packet = {'measurement_time_s':t, 'scheduled_delivery_time_s':t+cfg.observation_delay_s,
                      'delivered_time_s':None, 'objects':[
                          {'object_id':int(o.object_id),'position':np.asarray(o.position).tolist(),
                           'radius':float(o.radius_m)} for o in observations]}
            observation_log.append(packet)
            pending.append((t+cfg.observation_delay_s, observations, packet))
            maximum_observed = max(maximum_observed, len(observations))
            observation_counts.append(len(observations))
            observed_unique.update(o.object_id for o in observations)
            snapshots.append({'time_s':t,'ids':snapshot.ids.tolist(),
                              'positions':snapshot.positions.tolist(),'radii':snapshot.radii.tolist()})
        delivered = []
        while pending and pending[0][0] <= t+1e-9:
            _, observations, packet = pending.pop(0)
            packet['delivered_time_s'] = t
            delivered.extend(observations)
        # Timestamps remain measurement time, never delivery time or future truth.
        memory.update(delivered, t)
        if step % control_n == 0:
            tracks = memory.current(t)
            maximum_tracked = max(maximum_tracked, len(tracks))
            maximum_observation_age = max(maximum_observation_age,
                                          max((t-track.times[-1] for track in tracks),default=0.))
            plan = planner.plan(plant, goal, terrain, tracks, t)
            pending_commands.append((t+cfg.command_delay_s, t, plan))
            planning.append([t,plan.planning_ms,plan.candidate_count,plan.verified_count,plan.valid_count,
                             plan.minimum_predicted_clearance_m,len(tracks),int(plan.feasible),plan.prediction_fallback_count])
            plans.append({'time_s':t,'command_application_time_s':t+cfg.command_delay_s,
                          'action':[plan.action.depth_m,plan.action.heading_rad,plan.action.rpm],
                          'states':plan.states.copy(),'feasible':plan.feasible,
                          'validation_backend':'original_remus6dof','reason':plan.reason,
                          'emergency_verified_safe':plan.emergency_verified_safe,
                          'prediction_failure_count':plan.prediction_failure_count})
            plan_diagnostics.append({
                'time_s':t,'reason':plan.reason,'feasible':bool(plan.feasible),
                'candidate_count':plan.candidate_count,'verified_count':plan.verified_count,
                'valid_count':plan.valid_count,'prediction_failure_count':plan.prediction_failure_count,
                'emergency_verified_safe':plan.emergency_verified_safe,
                'tracks':[{'object_id':int(track.object_id),
                           'times_s':np.asarray(track.times).tolist(),
                           'positions':np.asarray(track.positions).tolist(),
                           'radius':float(track.radius_m)} for track in tracks]})
            if on_plan is not None:
                on_plan(clone_plant(plant), copy.deepcopy(tracks), t,
                        copy.deepcopy(plan), copy.deepcopy(scene))
            if not plan.feasible:
                events.append({'time_s':t,'kind':'emergency','position':plant.eta[:3].tolist(),
                               'reason':plan.reason,'forecast_clear':plan.emergency_verified_safe})
        while pending_commands and pending_commands[0][0] <= t+1e-9:
            _, plan_time, executing_plan = pending_commands.pop(0)
            action = executing_plan.action
            if cfg.command_delay_s:
                events.append({'time_s':t,'kind':'delayed_command_applied',
                               'position':plant.eta[:3].tolist(),'plan_time_s':plan_time,
                               'configured_delay_s':cfg.command_delay_s})
        old = plant.eta[:3].copy()
        sample = guidance.step(plant, action, dt)
        obstacles.step(dt)
        next_snapshot = obstacles.snapshot()
        if hasattr(obstacles,'t') and abs(obstacles.t-(t+dt)) > 1e-7:
            raise RuntimeError('obstacle and REMUS physical clocks diverged')
        if abs(plant.time_s-(t+dt)) > 1e-7:
            raise RuntimeError('REMUS physical clock diverged')
        controls.append([t,action.depth_m,action.heading_rad,action.rpm,*sample.raw,*sample.applied,
                         *plant.actual,*np.asarray(sample.saturated,dtype=int)])
        path_length += float(np.linalg.norm(plant.eta[:3]-old))
        if executing_plan is not None:
            elapsed = t+dt-plan_time
            index = min(round(elapsed/dt),len(executing_plan.states)-1)
            predicted = executing_plan.states[index]
            errors.append([t+dt,plan_time,elapsed,float(np.linalg.norm(plant.eta[:3]-predicted[1:4])),
                           float(plant.eta[2]-predicted[3]),float(wrap_pi(plant.eta[5]-predicted[6])),
                           float(plant.nu[0]-predicted[7])])
        clearance = min(terrain.segment_clearance(old,plant.eta[:3],cfg.body_radius_m),
                        cfg.depth_limit_m-cfg.body_radius_m-max(old[2],plant.eta[2]))
        if len(snapshot.ids):
            clearance = min(clearance,float(np.min(swept_sphere_clearance(
                old,plant.eta[:3],snapshot.positions,next_snapshot.positions,
                snapshot.radii+cfg.body_radius_m))))
        actual_min = min(actual_min,clearance)
        clearance_log.append([t+dt,clearance,clearance-cfg.safety_margin_m])
        if clearance <= 0:
            terrain_hit = terrain.segment_clearance(old,plant.eta[:3],cfg.body_radius_m) <= 0 or plant.eta[2]+cfg.body_radius_m >= cfg.depth_limit_m
            status = 'terrain_or_domain_collision' if terrain_hit else 'obstacle_collision'
            failure_reason = 'actual time-aligned swept trajectory collision; no safety threshold was relaxed'
            events.append({'time_s':t+dt,'kind':'collision','position':plant.eta[:3].tolist(),'reason':failure_reason})
            state_logs.append([t+dt,*plant.eta,*plant.nu,*plant.actual])
            break
    states = np.asarray(state_logs)
    terminal_time = float(states[-1,0])
    if not snapshots or abs(snapshots[-1]['time_s']-terminal_time)>1e-9:
        terminal = obstacles.snapshot()
        # Offline display truth at the actual endpoint, never a sensor delivery.
        snapshots.append({'time_s':terminal_time,'ids':terminal.ids.tolist(),
                          'positions':terminal.positions.tolist(),'radii':terminal.radii.tolist()})
    planning = np.asarray(planning).reshape(-1,len(PLANNING_COLUMNS))
    errors = np.asarray(errors).reshape(-1,len(ERROR_COLUMNS))
    latencies = planning[:,1] if len(planning) else np.array([0.])
    unsafe_emergencies = sum(e['kind']=='emergency' and not e.get('forecast_clear',False) for e in events)
    if 'collision' in status:
        safety_status = 'failed_actual_collision'
    elif unsafe_emergencies:
        safety_status = 'failed_unsafe_emergency_forecast'
    elif actual_min < cfg.safety_margin_m-1e-9:
        safety_status = 'failed_actual_safety_margin'
    elif status != 'success':
        safety_status = 'not_completed'
    else:
        safety_status = 'passed_finite_horizon_model_checks'
    summary = {
        'backend':'remus6dof','scenario':scenario,'seed':cfg.seed,'status':status,
        'reached_goal':status=='success','failure_reason':failure_reason,
        'avoidance_acceptance_passed':status=='success' and unsafe_emergencies==0 and actual_min>=cfg.safety_margin_m-1e-9,
        'safety_validation_status':safety_status,
        'safety_failure_reason':('one or more emergency coasting predictions violated the configured safety clearance; arrival does not erase this failure'
                                 if unsafe_emergencies else 'actual trajectory fell below the configured safety margin' if actual_min<cfg.safety_margin_m-1e-9 else failure_reason),
        'actual_safety_margin_met':actual_min>=cfg.safety_margin_m-1e-9,
        'duration_s':float(states[-1,0]),'goal_distance_m':float(np.linalg.norm(plant.eta[:3]-goal)),
        'goal_ned_m':goal.tolist(),'actual_start_ned_m':states[0,1:4].tolist(),
        'path_length_m':path_length,'minimum_model_clearance_m':float(actual_min),
        'actual_close_interaction_count':len(interacted),'interacted_obstacle_ids':sorted(interacted),
        'maximum_simultaneous_close_interactions':maximum_close,
        'interaction_threshold_m':cfg.interaction_distance_m,
        'obstacle_count':len(obstacles.snapshot().ids),'observed_unique_count':len(observed_unique),
        'maximum_observed_obstacles':maximum_observed,'maximum_tracked_obstacles':maximum_tracked,
        'mean_observed_obstacles':float(np.mean(observation_counts)) if observation_counts else 0.,
        'maximum_used_observation_age_s':maximum_observation_age,
        'emergency_count':sum(e['kind']=='emergency' for e in events),
        'unsafe_emergency_forecast_count':unsafe_emergencies,
        'valid_candidates_min':int(np.min(planning[:,4])) if len(planning) else 0,
        'valid_candidates_mean':float(np.mean(planning[:,4])) if len(planning) else 0.,
        'valid_candidates_max':int(np.max(planning[:,4])) if len(planning) else 0,
        'candidate_prediction_failure_count':sum(p['prediction_failure_count'] for p in plan_diagnostics),
        'planning_ms_median':float(np.median(latencies)),'planning_ms_p95':float(np.percentile(latencies,95)),
        'planning_ms_max':float(np.max(latencies)),
        'planning_exceeds_control_period_count':int(np.sum(latencies > cfg.control_dt*1000)),
        'planning_execution_timing':('fixed command-delay sensitivity experiment; previous reference remains active during configured delay; wall-clock planning latency is measured separately'
                                     if cfg.command_delay_s else 'synchronous offline simulation; physics time pauses during planning; measured wall-clock latency is not injected as an execution delay'),
        'command_delay_s':cfg.command_delay_s,
        'prediction_position_rmse_m':float(np.sqrt(np.mean(errors[:,3]**2))) if len(errors) else None,
        'prediction_position_max_error_m':float(np.max(errors[:,3])) if len(errors) else None,
        'prediction_error_scope':('active selected prediction at its absolute decision time; includes mismatch from fixed command delay'
                                  if cfg.command_delay_s else 'selected reference held until next replanning; matching deterministic original plant'),
        'control_log_time_convention':'time_s is interval start; raw/applied commands act over [time_s,time_s+physics_dt]; actual actuator columns are interval-end samples',
        'all_states_finite':bool(np.isfinite(states).all()),'wall_clock_s':time.perf_counter()-started,
        'physics_dt_s':dt,'control_dt_s':cfg.control_dt,'candidate_budget':cfg.candidate_budget,
        'maximum_full_rollouts_per_plan':cfg.candidate_budget+1,
        'initialization':'physical warmup then translation to start; controller, velocity and actuator states retained',
        'warmup_s':cfg.warmup_s,'hydrodynamic_parameters_modified':False,
        'prediction':'CA obstacle forecast + full REMUS shortlisted vehicle rollout; no ECP',
        'safety_claim':'finite-horizon model check only; emergency does not imply safe stopping',
        'energy_joules':None,'scene_metadata':scene.metadata,
    }
    return {'summary':summary,'states':states,'state_columns':STATE_COLUMNS,
            'controls':np.asarray(controls).reshape(-1,len(CONTROL_COLUMNS)),'control_columns':CONTROL_COLUMNS,
            'planning':planning,'planning_columns':PLANNING_COLUMNS,
            'prediction_errors':errors,'error_columns':ERROR_COLUMNS,'selected_plans':plans,
            'planning_diagnostics':plan_diagnostics,'observations':observation_log,
            'clearance':np.asarray(clearance_log).reshape(-1,3),
            'terrain':terrain,'obstacles':snapshots,'goal':goal,'config':cfg.to_dict(),'events':events}


def save_remus_episode(result,out,animate=True,plot=True):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'{out} is not empty; choose a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    for name in ('summary','config','events','obstacles'):
        (out/f'{name}.json').write_text(json.dumps(result[name],indent=2,ensure_ascii=False),encoding='utf-8')
    for name in ('planning_diagnostics','observations'):
        (out/f'{name}.json').write_text(json.dumps(result.get(name,[]),indent=2,ensure_ascii=False),encoding='utf-8')
    np.savetxt(out/'clearance.csv',result.get('clearance',np.empty((0,3))),delimiter=',',
               header='time_s,actual_interval_clearance_m,clearance_minus_safety_margin_m',comments='')
    for name,columns in [('states','state_columns'),('controls','control_columns'),
                         ('planning','planning_columns'),('prediction_errors','error_columns')]:
        np.savetxt(out/f'{name}.csv',result[name],delimiter=',',header=','.join(result[columns]),comments='')
    stride=max(1,round(result['config']['prediction_output_dt']/result['config']['physics_dt']))
    plans=[]
    for plan in result['selected_plans']:
        row=dict(plan)
        full_states=np.asarray(row['states'])
        indices=np.unique(np.r_[np.arange(0,len(full_states),stride),len(full_states)-1])
        row['states']=full_states[indices].tolist()
        row['saved_sampling']='display samples including final endpoint; collision verification used every physics tick'
        plans.append(row)
    (out/'selected_plans.json').write_text(json.dumps(plans),encoding='utf-8')
    np.savez_compressed(out/'terrain.npz',xs=result['terrain'].xs,ys=result['terrain'].ys,depths=result['terrain'].depths)
    if plot:
        from .stage2_visualization import plot_stage2_episode
        from .stage2_replay import export_stage2_replay
        plot_stage2_episode(result,out,animate=animate)
        export_stage2_replay(result,out/'replay_3d.html')
