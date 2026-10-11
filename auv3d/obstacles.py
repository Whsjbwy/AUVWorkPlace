from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import ObstacleConfig
from .geometry import wrap_pi, kinematic_step
from .terrain import HeightField

@dataclass
class ObstacleSnapshot:
    ids: np.ndarray
    positions: np.ndarray
    radii: np.ndarray

class LinearObstacles:
    """Transparent controlled-scenario obstacle generator, NOT P25 random PI."""
    def __init__(self,positions,velocities,radii):
        self.positions=np.asarray(positions,dtype=float).reshape(-1,3)
        self.velocities=np.asarray(velocities,dtype=float).reshape(-1,3)
        self.radii=np.asarray(radii,dtype=float).reshape(-1)
        if self.positions.shape != self.velocities.shape or len(self.radii)!=len(self.positions):
            raise ValueError('obstacle array lengths differ')
        if np.any(self.radii<=0): raise ValueError('nonpositive radius')
    def snapshot(self):
        return ObstacleSnapshot(np.arange(len(self.radii)),self.positions.copy(),self.radii.copy())
    def step(self,dt): self.positions+=self.velocities*dt

class PIObstacles:
    """P25 Table 2/4 values; disclosed saturation/boundary rules in docs."""
    def __init__(self,cfg:ObstacleConfig,terrain:HeightField,seed:int,exclude=()):
        self.cfg=cfg;self.terrain=terrain;self.rng=np.random.default_rng(seed);self.t=0.;self.next_target=cfg.target_period_s
        accepted=[];radii=[]
        for _ in range(max(1000, cfg.count*2000)):
            if len(accepted)==cfg.count: break
            rad=self.rng.uniform(cfg.radius_min_m,cfg.radius_max_m)
            p=np.array([self.rng.uniform(terrain.xs[0]+rad,terrain.xs[-1]-rad),
                        self.rng.uniform(terrain.ys[0]+rad,terrain.ys[-1]-rad),
                        self.rng.uniform(rad+1,float(terrain.depths.max())-rad)])
            if terrain.clearance(p,rad)<1:continue
            if len(accepted) and np.any(np.linalg.norm(np.asarray(accepted)-p,axis=1)<cfg.min_separation_m):continue
            if any(np.linalg.norm(p-np.asarray(q))<rad+10 for q in exclude):continue
            accepted.append(p);radii.append(rad)
        if len(accepted)!=cfg.count:raise RuntimeError('cannot place obstacles; do not silently reduce count')
        n=cfg.count
        self.states=np.zeros((n,6));self.states[:,:3]=np.asarray(accepted).reshape(n,3)
        self.states[:,3]=self.rng.uniform(cfg.speed_min,cfg.speed_max,n)
        self.states[:,4]=self.rng.uniform(-cfg.pitch_max_deg,cfg.pitch_max_deg,n)*np.pi/180
        self.states[:,5]=self.rng.uniform(-np.pi,np.pi,n)
        self.radii=np.asarray(radii);self.integrals=np.zeros((n,3));self.targets=self._targets()
        self.boundary_events=0
    def _targets(self):
        c=self.cfg;n=len(self.radii)
        return np.column_stack([self.rng.uniform(c.speed_min,c.speed_max,n),
            self.rng.uniform(-c.pitch_max_deg,c.pitch_max_deg,n)*np.pi/180,
            self.rng.uniform(-np.pi,np.pi,n)])
    def snapshot(self):return ObstacleSnapshot(np.arange(len(self.radii)),self.states[:,:3].copy(),self.radii.copy())
    def step(self,dt):
        if not len(self.radii):self.t+=dt;return
        c=self.cfg
        if self.t+1e-9>=self.next_target:
            self.targets=self._targets();self.next_target+=c.target_period_s
        err=self.targets-self.states[:,3:6];err[:,2]=wrap_pi(err[:,2])
        self.integrals=np.clip(self.integrals+err*dt,-10,10)  # disclosed anti-windup choice
        rates=np.asarray(c.kp)*err+np.asarray(c.ki)*self.integrals
        rates=np.clip(rates,[-c.acceleration_max,-np.deg2rad(c.q_max_deg_s),-np.deg2rad(c.r_max_deg_s)],
                       [c.acceleration_max,np.deg2rad(c.q_max_deg_s),np.deg2rad(c.r_max_deg_s)])
        speed=np.clip(self.states[:,3]+dt*rates[:,0],c.speed_min,c.speed_max)
        nxt=kinematic_step(self.states,np.column_stack([speed,rates[:,1:]]),dt)
        nxt[:,4]=np.clip(nxt[:,4],-np.deg2rad(c.pitch_max_deg),np.deg2rad(c.pitch_max_deg))
        bad=self.terrain.clearance(nxt[:,:3],self.radii)<.1
        # Boundary retreat: reject positional move, reverse direction and reset PI.
        # Not authors' specified behavior; boundary events remain in diagnostics.
        if np.any(bad):
            nxt[bad,:3]=self.states[bad,:3];nxt[bad,5]=wrap_pi(self.states[bad,5]+np.pi)
            nxt[bad,4]=-self.states[bad,4];self.targets[bad,1:]=nxt[bad,4:6];self.integrals[bad]=0
            self.boundary_events+=int(bad.sum())
        self.states=nxt;self.t+=dt
