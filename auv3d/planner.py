from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .config import ExperimentConfig
from .geometry import kinematic_step,swept_sphere_clearance,wrap_pi
from .sensors import Track,validate_track
from .prediction import CAPredictor,SphericalCalibration,make_history
from .terrain import HeightField

@dataclass
class PlanResult:
    command: np.ndarray
    states: np.ndarray
    feasible: bool
    candidate_count: int
    valid_count: int
    minimum_predicted_clearance_m: float
    planning_ms: float
    prediction_fallback_count: int

def candidate_commands(state,previous,cfg:ExperimentConfig):
    p,v=cfg.planner,cfg.vehicle
    u=np.linspace(max(v.min_speed,state[3]-v.acceleration_max*p.control_dt),
                  min(v.max_speed,state[3]+v.acceleration_max*p.control_dt),p.speed_samples)
    delta=np.deg2rad(v.angular_acceleration_deg_s2)*p.control_dt
    def rates(prev,limit,n):
        lo=max(-np.deg2rad(limit),prev-delta);hi=min(np.deg2rad(limit),prev+delta)
        vals=np.linspace(lo,hi,n)
        if lo<=0<=hi:vals=np.unique(np.r_[vals,0.])
        return vals
    q=rates(previous[1],v.q_max_deg_s,p.pitch_samples)
    r=rates(previous[2],v.r_max_deg_s,p.yaw_samples)
    grid=np.meshgrid(u,q,r,indexing='ij')
    return np.column_stack([a.ravel() for a in grid])

def rollout(state,commands,cfg:ExperimentConfig):
    p=cfg.planner;steps=round(p.horizon_s/p.rollout_dt)
    x=np.repeat(np.asarray(state)[None],len(commands),axis=0)
    result=[x.copy()]
    for _ in range(steps):
        x=kinematic_step(x,commands,p.rollout_dt,cfg.current_ned_mps)
        result.append(x.copy())
    return np.stack(result,axis=1)

class DWA3D:
    """Shared 3D candidate planner. This stage is a baseline, not ECP.

    Hard trajectory checks and a smoothness cost are disclosed strengthening
    choices, not a claim to the authors' exact DWA weights or implementation.
    """
    def __init__(self,cfg:ExperimentConfig,predictor=None,calibration=None):
        cfg.validate();self.cfg=cfg;self.predictor=predictor or CAPredictor(cfg.prediction_dt_s)
        self.calibration=calibration;self.previous=np.zeros(3)
        if not np.isfinite(self.predictor.dt) or abs(self.predictor.dt-cfg.prediction_dt_s)>1e-9:
            raise ValueError('predictor physical time step mismatch')
        if calibration is not None:
            calibration.validate_for(self.predictor,cfg.prediction_dt_s,3)
    def reset(self):self.previous=np.zeros(3)
    def _evaluate_paths(self,paths,terrain:HeightField,tracks:list[Track],now):
        """Apply the same checks to ordinary candidates and the fallback itself."""
        cfg=self.cfg;p,v=cfg.planner,cfg.vehicle
        paths=np.asarray(paths,dtype=float)
        if paths.ndim!=3 or paths.shape[2]!=6 or paths.shape[1]<2:
            raise ValueError('P25 paths require [candidate,time,6] states with at least two times')
        if not np.isfinite(now):
            raise ValueError('P25 planning time must be finite')
        times=np.arange(paths.shape[1])*p.rollout_dt;pts=paths[...,:3]
        if (not np.isfinite(paths).all() or not np.isfinite(terrain.depths).all()
                or not np.isfinite(terrain.slope_bound) or terrain.slope_bound<0):
            raise FloatingPointError('P25 trajectory and terrain must be finite with nonnegative slope')
        clear=terrain.clearance(pts,v.body_radius_m).min(axis=1)
        valid=np.all(np.abs(paths[...,4])<=np.deg2rad(v.pitch_max_deg)+1e-9,axis=1)
        valid &= np.all(paths[...,2]+v.body_radius_m<=v.depth_limit_m,axis=1)
        # Intersample terrain bound, under the same piecewise smooth heightfield.
        gaps=np.linalg.norm(np.diff(pts,axis=1),axis=-1).max(axis=1)
        clear-=.5*gaps*np.sqrt(1+terrain.slope_bound**2)
        # The 100 m operating boundary is a domain constraint even when the
        # physical seafloor lies deeper. Apply the configured clearance there.
        depth_clear=np.min(v.depth_limit_m-v.body_radius_m-paths[...,2],axis=1)
        clear=np.minimum(clear,depth_clear)
        fallback_count=0
        for track in tracks:
            validate_track(track,now)
            age=max(0.,now-track.times[-1])
            if times[-1]+age > getattr(self.predictor,'max_horizon_s',float('inf'))+1e-8:
                # Explicit stale-track fallback only; do not hide other data/model errors.
                pred=CAPredictor(cfg.prediction_dt_s).predict(track,now+times);fallback_count+=1
            else:
                pred=self.predictor.predict(track,now+times)
            if np.shape(pred)!=(len(times),3) or not np.isfinite(pred).all():
                raise FloatingPointError('P25 obstacle prediction must contain finite [time,3] positions')
            radius=np.full(len(times),p.base_prediction_margin_m+cfg.obstacles.speed_max*age)
            if self.calibration is not None and age<1e-8:
                hist=make_history(track.times,track.positions,cfg.prediction_dt_s)
                speed=np.linalg.norm(hist.features[-1,3:6])
                radius=self.calibration.radius(speed,times,cfg.prediction_dt_s)
            total=track.radius_m+v.body_radius_m+np.maximum(radius[:-1],radius[1:])
            c=swept_sphere_clearance(pts[:,:-1],pts[:,1:],pred[:-1],pred[1:],total)
            clear=np.minimum(clear,c.min(axis=1))
        if not np.isfinite(clear).all():
            raise FloatingPointError('P25 trajectory clearance must be finite')
        valid &= clear>=p.safety_margin_m
        return clear,valid,fallback_count

    def plan(self,state,goal,terrain:HeightField,tracks:list[Track],now):
        start=time.perf_counter();cfg=self.cfg;p,v=cfg.planner,cfg.vehicle
        state,goal=np.asarray(state,dtype=float),np.asarray(goal,dtype=float)
        if state.shape!=(6,) or goal.shape!=(3,) or not np.isfinite(np.r_[state,goal,self.previous]).all():
            raise ValueError('P25 state, goal and previous command must have valid finite coordinates')
        if not np.isfinite(now):
            raise ValueError('P25 planning time must be finite')
        commands=candidate_commands(state,self.previous,cfg);paths=rollout(state,commands,cfg)
        clear,valid,fallback_count=self._evaluate_paths(paths,terrain,tracks,now)
        progress=np.linalg.norm(paths[:,-1,:3]-goal,axis=-1)
        cost=p.goal_weight*progress+p.clearance_weight/np.maximum(clear,.1)-p.speed_weight*commands[:,0]
        scale=np.array([v.max_speed,np.deg2rad(v.q_max_deg_s),np.deg2rad(v.r_max_deg_s)])
        cost+=p.smooth_weight*np.sum(((commands-self.previous)/scale)**2,axis=1)
        if valid.any():
            chosen=int(np.argmin(np.where(valid,cost,np.inf)));command=commands[chosen];path=paths[chosen];min_c=float(clear[chosen]);ok=True
        else:
            # Emergency deceleration is logged. It is NOT a proven safe backup,
            # and does not instantaneously stop a vehicle drifting in current.
            angular_step=np.deg2rad(v.angular_acceleration_deg_s2)*p.control_dt
            rates=np.clip(np.zeros(2),self.previous[1:]-angular_step,self.previous[1:]+angular_step)
            command=np.array([max(v.min_speed,state[3]-v.acceleration_max*p.control_dt),*rates])
            path=rollout(state,command[None],cfg)[0]
            fallback_clear,_,extra_fallbacks=self._evaluate_paths(path[None],terrain,tracks,now)
            fallback_count=max(fallback_count,extra_fallbacks)
            min_c=float(fallback_clear[0]);ok=False
        self.previous=command.copy()
        return PlanResult(command,path,ok,len(commands),int(valid.sum()),min_c,(time.perf_counter()-start)*1e3,fallback_count)
