"""P25 kinematic backend on the shared stage 2 scenes and sensor timeline.

This comparison backend retains DWA3D scoring and collision checks. It shares
terrain, obstacle truth, requested start/goal, observation settings and swept
collision evaluation with the REMUS experiment. Its instantaneous commanded
speed and absent actuators are different dynamics, not a fair algorithm-only
comparison and never a six-DOF execution result.
"""
from __future__ import annotations

from dataclasses import replace
import platform
import time
import numpy as np

from .config import ExperimentConfig
from .geometry import kinematic_step, swept_sphere_clearance
from .planner import DWA3D as OriginalDWA3D, PlanResult, candidate_commands, rollout
from .prediction import CAPredictor
from .scenarios import make_stage2_scenario
from .sensors import IdealObjectSonar, TrackMemory, validate_track
from .stage2_config import Stage2Config


def pitch_legal_commands(state, commands, cfg):
    """Reject illegal full-horizon pitch rates before evaluating Euler dynamics.

    P25 pitch evolves linearly under a fixed q command, so checking both
    endpoints is sufficient. The original planner applies the same pitch
    envelope after rollout, which is too late for a 24 s singular candidate.
    No accepted trajectory, collision threshold or physical parameter changes.
    """
    commands = np.asarray(commands, dtype=float).reshape(-1, 3)
    limit = np.deg2rad(cfg.vehicle.pitch_max_deg)
    pitch = float(state[4])
    if abs(pitch) > limit+1e-9:
        raise ValueError('actual P25 pitch exceeds the original configured envelope')
    end_pitch = pitch+commands[:, 1]*cfg.planner.horizon_s
    return commands[np.abs(end_pitch) <= limit+1e-12]


class DWA3D(OriginalDWA3D):
    """Stage 2 pitch adapter around the validated stage 1 P25 baseline.

    Only the analytical full-horizon pitch rejection moves ahead of rollout.
    The original candidate grid, kinematics, CA obstacle forecasts and scoring
    are retained. Both backends reject malformed inputs and apply the same
    clearance threshold at the configured operating-depth boundary.
    """

    def _evaluate_paths(self, paths, terrain, tracks, now):
        cfg, p, v = self.cfg, self.cfg.planner, self.cfg.vehicle
        paths = np.asarray(paths,dtype=float)
        if paths.ndim != 3 or paths.shape[2] != 6 or paths.shape[1] < 2:
            raise ValueError('P25 paths require [candidate,time,6] states with at least two times')
        if (not np.isfinite(paths).all() or not np.isfinite(terrain.depths).all()
                or not np.isfinite(terrain.slope_bound) or terrain.slope_bound < 0):
            raise FloatingPointError('P25 trajectory and terrain must be finite with nonnegative slope')
        if not np.isfinite(now):
            raise ValueError('P25 planning time must be finite')
        pts = paths[..., :3]
        times = np.arange(paths.shape[1])*p.rollout_dt
        clear = terrain.clearance(pts, v.body_radius_m).min(axis=1)
        gaps = np.linalg.norm(np.diff(pts, axis=1), axis=-1).max(axis=1)
        clear -= .5*gaps*np.sqrt(1+terrain.slope_bound**2)
        depth_clear = np.min(v.depth_limit_m-v.body_radius_m-paths[...,2],axis=1)
        clear = np.minimum(clear,depth_clear)
        valid = np.all(np.abs(paths[..., 4]) <= np.deg2rad(v.pitch_max_deg)+1e-9, axis=1)
        valid &= np.all(paths[..., 2]+v.body_radius_m <= v.depth_limit_m, axis=1)
        fallback_count = 0
        for track in tracks:
            validate_track(track, now)
            age = max(0., now-track.times[-1])
            if times[-1]+age > getattr(self.predictor, 'max_horizon_s', float('inf'))+1e-8:
                pred = CAPredictor(cfg.prediction_dt_s).predict(track, now+times)
                fallback_count += 1
            else:
                pred = self.predictor.predict(track, now+times)
            if np.shape(pred) != (len(times),3) or not np.isfinite(pred).all():
                raise FloatingPointError('P25 obstacle prediction must contain finite [time,3] positions')
            margin = p.base_prediction_margin_m+cfg.obstacles.speed_max*age
            total = track.radius_m+v.body_radius_m+margin
            clearance = swept_sphere_clearance(pts[:, :-1], pts[:, 1:], pred[:-1], pred[1:], total)
            clear = np.minimum(clear, clearance.min(axis=1))
        if not np.isfinite(clear).all():
            raise FloatingPointError('P25 trajectory clearance must be finite')
        return clear, valid & (clear >= p.safety_margin_m), fallback_count

    def plan(self, state, goal, terrain, tracks, now):
        started = time.perf_counter()
        cfg, p, v = self.cfg, self.cfg.planner, self.cfg.vehicle
        state,goal = np.asarray(state,dtype=float),np.asarray(goal,dtype=float)
        if state.shape != (6,) or goal.shape != (3,) or not np.isfinite(np.r_[state,goal,self.previous]).all():
            raise ValueError('P25 state, goal and previous command must have valid finite coordinates')
        if not np.isfinite(now):
            raise ValueError('P25 planning time must be finite')
        original_commands = candidate_commands(state, self.previous, cfg)
        commands = pitch_legal_commands(state, original_commands, cfg)
        fallback_count = 0
        if len(commands):
            paths = rollout(state, commands, cfg)
            clear, valid, fallback_count = self._evaluate_paths(paths, terrain, tracks, now)
        else:
            valid = np.zeros(0, dtype=bool)
        if valid.any():
            progress = np.linalg.norm(paths[:, -1, :3]-goal, axis=-1)
            cost = p.goal_weight*progress+p.clearance_weight/np.maximum(clear, .1)-p.speed_weight*commands[:, 0]
            scale = np.array([v.max_speed, np.deg2rad(v.q_max_deg_s), np.deg2rad(v.r_max_deg_s)])
            cost += p.smooth_weight*np.sum(((commands-self.previous)/scale)**2, axis=1)
            selected = int(np.argmin(np.where(valid, cost, np.inf)))
            command, path, minimum, feasible = commands[selected], paths[selected], float(clear[selected]), True
        else:
            angular_step = np.deg2rad(v.angular_acceleration_deg_s2)*p.control_dt
            rates = np.clip(np.zeros(2), self.previous[1:]-angular_step, self.previous[1:]+angular_step)
            limit = np.deg2rad(v.pitch_max_deg)
            rates[0] = np.clip(rates[0], (-limit-state[4])/p.horizon_s, (limit-state[4])/p.horizon_s)
            command = np.array([max(v.min_speed, state[3]-v.acceleration_max*p.control_dt), *rates])
            path = rollout(state, command[None], cfg)[0]
            clearance, _, extra_fallbacks = self._evaluate_paths(path[None], terrain, tracks, now)
            fallback_count = max(fallback_count, extra_fallbacks)
            minimum, feasible = float(clearance[0]), False
        self.previous = command.copy()
        return PlanResult(command, path, feasible, len(original_commands), int(valid.sum()), minimum,
                          (time.perf_counter()-started)*1000, fallback_count)


def p25_config(cfg: Stage2Config) -> ExperimentConfig:
    """Map experiment timing/settings, retaining the original P25 rate limits.

    The output sampling interval is reduced to a common divisor when necessary
    to preserve the original planner's integer-grid timing contract. RPM
    reference samples have no physical interpretation in this backend.
    """
    cfg.validate()
    if cfg.command_delay_s:
        raise ValueError('command_delay_s sensitivity is implemented only for the REMUS backend')
    base = ExperimentConfig()
    divisor = np.gcd(round(cfg.control_dt/cfg.physics_dt),
                     round(cfg.prediction_output_dt/cfg.physics_dt))
    rollout_dt = float(divisor*cfg.physics_dt)
    direction = np.deg2rad(cfg.current_direction_deg)
    planner = replace(base.planner, control_dt=cfg.control_dt,
                      integration_dt=cfg.physics_dt, rollout_dt=rollout_dt,
                      horizon_s=cfg.horizon_s, safety_margin_m=cfg.safety_margin_m,
                      base_prediction_margin_m=cfg.prediction_margin_m,
                      track_max_age_s=cfg.track_max_age_s,
                      goal_tolerance_m=cfg.goal_tolerance_m, max_time_s=cfg.max_time_s,
                      goal_weight=cfg.goal_weight, clearance_weight=cfg.clearance_weight,
                      speed_weight=cfg.speed_weight, smooth_weight=cfg.smooth_weight)
    mapped = replace(base, sonar=cfg.sensor_config(), planner=planner,
                     vehicle=replace(base.vehicle, body_radius_m=cfg.body_radius_m,
                                     depth_limit_m=cfg.depth_limit_m),
                     seed=cfg.seed, prediction_dt_s=cfg.horizon_s/3.,
                     current_ned_mps=(float(cfg.current_speed*np.cos(direction)),
                                      float(cfg.current_speed*np.sin(direction)), 0.),
                     label="stage2_shared_scene_p25_kinematics_not_remus6dof")
    mapped.validate()
    return mapped


def _snapshot_record(snapshot, stamp):
    return {'time_s':stamp, 'ids':snapshot.ids.tolist(),
            'positions':snapshot.positions.tolist(), 'radii':snapshot.radii.tolist()}


def run_p25_episode(cfg: Stage2Config, scenario='crossing3d'):
    """Execute P25 kinematics using causal shared observations and ground truth.

    Return the original stage 1 result shape so ``save_episode`` and its P25
    plotter remain usable. Additional events are available for the stage 2 CLI.
    The fixed initial water surge of 1 m/s follows the stage 1 P25 convention;
    REMUS warmup, reference/RPM sampling and actuator memory do not apply here.
    """
    mapped = p25_config(cfg)
    scene = make_stage2_scenario(scenario, cfg)
    terrain, obstacles = scene.terrain, scene.obstacles
    goal = np.asarray(scene.goal, dtype=float)
    eta = np.asarray(scene.start_eta, dtype=float)
    if abs(eta[3]) > 1e-12:
        raise ValueError('P25 shared-scene backend cannot represent nonzero initial roll')
    state = np.array([*eta[:3], 1.0, eta[4], eta[5]])
    sonar = IdealObjectSonar(cfg.sensor_config(), cfg.seed)
    memory = TrackMemory(cfg.track_max_age_s)
    planner = DWA3D(mapped)
    dt = cfg.physics_dt
    control_n = round(cfg.control_dt/dt)
    obs_n = round(cfg.observation_period_s/dt)
    total = round(cfg.max_time_s/dt)
    command = np.array([state[3], 0., 0.])
    logs, planning, snapshots, plans, events, pending = [], [], [], [], [], []
    interacted, observed_unique = set(), set()
    max_observed = max_tracked = max_close = 0
    maximum_age = 0.
    observation_counts = []
    actual_min = float('inf')
    path_length = effort = 0.
    fallbacks = 0
    status = 'timeout'
    failure_reason = 'goal tolerance was not reached before max_time_s'
    started = time.perf_counter()
    for step in range(total+1):
        stamp = step*dt
        snapshot = obstacles.snapshot()
        if len(snapshot.ids):
            distance = np.linalg.norm(snapshot.positions-state[:3], axis=1)-snapshot.radii-cfg.body_radius_m
            actual_min = min(actual_min, float(distance.min()))
            interacted.update(int(i) for i in snapshot.ids[distance <= cfg.interaction_distance_m])
            max_close = max(max_close, int(np.sum(distance <= cfg.interaction_distance_m)))
        logs.append([stamp, *state, *command])
        terrain_clear = min(float(terrain.clearance(state[:3], cfg.body_radius_m)),
                            cfg.depth_limit_m-cfg.body_radius_m-state[2])
        actual_min = min(actual_min, terrain_clear)
        if terrain_clear <= 0 or (len(snapshot.ids) and distance.min() <= 0):
            status = 'terrain_or_domain_collision' if terrain_clear <= 0 else 'obstacle_collision'
            failure_reason = 'initial/current actual enclosing sphere overlaps terrain/domain or obstacle'
            events.append({'time_s':stamp, 'kind':'collision', 'position':state[:3].tolist(), 'reason':failure_reason})
            break
        if np.linalg.norm(state[:3]-goal) <= cfg.goal_tolerance_m:
            status, failure_reason = 'success', None
            break
        if step == total:
            break
        if step % obs_n == 0:
            sensor_eta = np.array([*state[:3], 0., state[4], state[5]])
            observations = sonar.observe(sensor_eta, stamp, snapshot, terrain)
            pending.append((stamp+cfg.observation_delay_s, observations))
            max_observed = max(max_observed, len(observations))
            observation_counts.append(len(observations))
            observed_unique.update(o.object_id for o in observations)
            snapshots.append(_snapshot_record(snapshot, stamp))
        delivered = []
        while pending and pending[0][0] <= stamp+1e-9:
            _, observations = pending.pop(0)
            delivered.extend(observations)
        memory.update(delivered, stamp)
        if step % control_n == 0:
            tracks = memory.current(stamp)
            max_tracked = max(max_tracked, len(tracks))
            maximum_age = max(maximum_age, max((stamp-t.times[-1] for t in tracks), default=0.))
            plan = planner.plan(state, goal, terrain, tracks, stamp)
            command = plan.command
            fallbacks += plan.prediction_fallback_count
            planning.append([stamp, plan.planning_ms, plan.candidate_count, plan.valid_count,
                             plan.minimum_predicted_clearance_m, len(tracks), int(plan.feasible)])
            plans.append({'time_s':stamp, 'command':command.tolist(), 'states':plan.states.tolist(),
                          'feasible':plan.feasible, 'validation_backend':'p25_kinematic_not_remus6dof'})
            if not plan.feasible:
                events.append({'time_s':stamp, 'kind':'emergency', 'position':state[:3].tolist(),
                               'reason':'P25 kinematic deceleration fallback; no physically verified REMUS stopping claim',
                               'forecast_clear':bool(plan.minimum_predicted_clearance_m>=cfg.safety_margin_m)})
        old = state.copy()
        state = kinematic_step(state, command, dt, mapped.current_ned_mps)
        obstacles.step(dt)
        if hasattr(obstacles, 't') and abs(obstacles.t-(stamp+dt)) > 1e-7:
            raise RuntimeError('obstacle and P25 physical clocks diverged')
        next_snapshot = obstacles.snapshot()
        path_length += float(np.linalg.norm(state[:3]-old[:3]))
        effort += float(command[0]**3*dt)
        terrain_clear = min(terrain.segment_clearance(old[:3], state[:3], cfg.body_radius_m),
                            cfg.depth_limit_m-cfg.body_radius_m-max(old[2],state[2]))
        clearance = terrain_clear
        if len(snapshot.ids):
            clearance = min(clearance, float(np.min(swept_sphere_clearance(
                old[:3], state[:3], snapshot.positions, next_snapshot.positions,
                snapshot.radii+cfg.body_radius_m))))
        actual_min = min(actual_min, clearance)
        if clearance <= 0:
            status = 'terrain_or_domain_collision' if terrain_clear <= 0 else 'obstacle_collision'
            failure_reason = 'actual time-aligned swept trajectory collision; no safety threshold was relaxed'
            events.append({'time_s':stamp+dt, 'kind':'collision', 'position':state[:3].tolist(), 'reason':failure_reason})
            logs.append([stamp+dt, *state, *command])
            break
    states = np.asarray(logs).reshape(-1, 10)
    if not snapshots or abs(snapshots[-1]['time_s']-states[-1, 0]) > 1e-9:
        # Offline truth at the actual terminal time; never a new observation.
        snapshots.append(_snapshot_record(obstacles.snapshot(), float(states[-1, 0])))
    planning = np.asarray(planning).reshape(-1, 7)
    latencies = planning[:, 1] if len(planning) else np.array([0.])
    emergencies = sum(e['kind']=='emergency' for e in events)
    unsafe = sum(e['kind']=='emergency' and not e.get('forecast_clear',False) for e in events)
    safety_status = ('failed_actual_collision' if 'collision' in status
                     else 'failed_unsafe_emergency_forecast' if unsafe
                     else 'failed_actual_safety_margin' if actual_min<cfg.safety_margin_m-1e-9
                     else 'not_completed' if status!='success'
                     else 'passed_finite_horizon_kinematic_checks')
    summary = {
        'stage':'stage2_shared_scene_engineering_comparison',
        'backend':'p25_kinematic_3d_rk4', 'scenario':scenario, 'seed':cfg.seed,
        'status':status, 'reached_goal':status=='success', 'failure_reason':failure_reason,
        'avoidance_acceptance_passed':status=='success' and unsafe==0 and actual_min>=cfg.safety_margin_m-1e-9,
        'actual_safety_margin_met':actual_min>=cfg.safety_margin_m-1e-9,
        'safety_validation_status':safety_status,
        'unsafe_emergency_forecast_count':unsafe,
        'execution_model':'P25 simplified kinematics; no REMUS actuator tracking',
        'comparison_scope':'shared scene/sensor/time/collision evaluation; different vehicle dynamics and command spaces; not an algorithm-only comparison',
        'remus6dof_avoidance_claim':False,
        'initialization':'same requested scene pose; fixed P25 water surge 1 m/s; no physical warmup',
        'duration_s':float(states[-1,0]), 'path_length_m':path_length,
        'goal_distance_m':float(np.linalg.norm(state[:3]-goal)),
        'minimum_model_clearance_m':float(actual_min),
        'actual_close_interaction_count':len(interacted), 'interacted_obstacle_ids':sorted(interacted),
        'maximum_simultaneous_close_interactions':max_close,
        'interaction_threshold_m':cfg.interaction_distance_m,
        'obstacle_count':len(obstacles.snapshot().ids), 'observed_unique_count':len(observed_unique),
        'maximum_observed_obstacles':max_observed, 'maximum_tracked_obstacles':max_tracked,
        'mean_observed_obstacles':float(np.mean(observation_counts)) if observation_counts else 0.,
        'maximum_used_observation_age_s':maximum_age,
        'emergency_count':emergencies, 'emergency_deceleration_count':emergencies,
        'stale_learned_prediction_fallback_count':fallbacks,
        'predictor':planner.predictor.name, 'calibration':'uncalibrated_debug_margin',
        'planning_ms_median':float(np.median(latencies)),
        'planning_ms_p95':float(np.percentile(latencies,95)), 'planning_ms_max':float(latencies.max()),
        'planning_exceeds_control_period_count':int(np.sum(latencies > cfg.control_dt*1000)),
        'physics_dt_s':dt, 'control_dt_s':cfg.control_dt,
        'all_states_finite':bool(np.isfinite(states).all()),
        'wall_clock_s':time.perf_counter()-started, 'python':platform.python_version(), 'numpy':np.__version__,
        'configuration_label':mapped.label, 'model_change':'none; original P25 kinematics run on shared scenes',
        'pitch_candidate_filter':'analytical current_pitch+q*horizon within original +/-30deg before Euler rollout; zero-q retained',
        'safety_claim':'P25 finite-horizon kinematic check only; emergency is not a REMUS safe-stop guarantee',
        'surge_cubic_effort_proxy':effort, 'energy_joules':None,
        'energy_note':'u^3 integral is a proxy, NOT electrical energy',
        'obstacle_boundary_events':getattr(obstacles,'boundary_events',0),
        'scene_metadata':scene.metadata,
    }
    return {'summary':summary, 'states':states, 'planning':planning, 'goal':goal,
            'terrain':terrain, 'obstacles':snapshots, 'selected_plans':plans,
            'config':{'shared_stage2':cfg.to_dict(), 'p25_mapping':mapped.to_dict()}, 'events':events}
