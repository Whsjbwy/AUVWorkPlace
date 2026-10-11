from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .config import SonarConfig
from .geometry import rotation_zyx
from .obstacles import ObstacleSnapshot
from .terrain import HeightField

@dataclass(frozen=True)
class Observation:
    object_id: int
    time_s: float
    position: np.ndarray
    radius_m: float

class IdealObjectSonar:
    """Geometric, body-fixed 3D object sensor; NOT an acoustic image simulator.

    Range/angles are paper values. Sphere angular extent is included. Object
    association and sizes are idealized. Angular resolution is metadata only;
    this stage does not claim beam-level reconstruction or acoustic uncertainty.
    """
    def __init__(self,cfg:SonarConfig,seed:int):self.cfg=cfg;self.rng=np.random.default_rng(seed)
    def observe(self,eta,time_s,snapshot:ObstacleSnapshot,terrain:HeightField):
        e=np.asarray(eta);R=rotation_zyx(*e[3:6]);rel=(snapshot.positions-e[:3])@R
        d=np.linalg.norm(rel,axis=1);rad=snapshot.radii
        az=np.arctan2(rel[:,1],rel[:,0]);el=np.arctan2(rel[:,2],np.hypot(rel[:,0],rel[:,1]))
        pad=np.arcsin(np.clip(rad/np.maximum(d,1e-9),0,1))
        visible=(d-rad<=self.cfg.range_m)&(np.abs(az)<=np.deg2rad(self.cfg.hfov_deg/2)+pad)&(np.abs(el)<=np.deg2rad(self.cfg.vfov_deg/2)+pad)
        out=[]
        for i in np.flatnonzero(visible):
            if self.cfg.terrain_occlusion and not terrain.visible_line(e[:3],snapshot.positions[i]):continue
            p=snapshot.positions[i]+self.rng.normal(0,self.cfg.position_noise_std_m,3)
            out.append(Observation(int(snapshot.ids[i]),float(time_s),p.copy(),float(rad[i])))
        return out

@dataclass
class Track:
    object_id: int
    radius_m: float
    times: list[float] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)
    def append(self,o:Observation):
        if self.times and o.time_s <= self.times[-1]:raise ValueError('observations must be strictly time ordered')
        self.times.append(o.time_s);self.positions.append(o.position.copy());self.radius_m=o.radius_m
        self.times=self.times[-32:];self.positions=self.positions[-32:]

def validate_track(track: Track, now: float) -> None:
    """Validate a causal sphere track before forecasting or collision checking.

    Measurements retain their acquisition times. Duplicate/reversed times are
    rejected before velocity division, and a nonpositive size cannot shrink a
    collision envelope. This check does not modify the supplied track.
    """
    if not np.isfinite(now):
        raise ValueError('obstacle track validation requires finite now')
    observed_times=np.asarray(track.times,dtype=float)
    observed_positions=np.asarray(track.positions,dtype=float)
    if observed_times.ndim!=1 or len(observed_times)==0 or not np.isfinite(observed_times).all():
        raise ValueError('planning requires causal, nonempty obstacle tracks')
    if observed_positions.shape!=(len(observed_times),3) or not np.isfinite(observed_positions).all():
        raise ValueError('obstacle track requires matching finite [time,3] positions')
    if np.any(np.diff(observed_times)<=0):
        raise ValueError('obstacle observation times must be strictly increasing')
    if observed_times[-1]>now:
        raise ValueError('planning requires causal obstacle tracks; future observation rejected')
    if not np.isfinite(track.radius_m) or track.radius_m<=0:
        raise ValueError('obstacle radius must be finite and positive')

class TrackMemory:
    def __init__(self,max_age_s:float=3.):self.max_age_s=max_age_s;self.tracks={}
    def update(self,observations:list[Observation],now:float):
        for o in observations:
            if o.time_s>now+1e-9:raise ValueError('future observation leak')
            if o.object_id not in self.tracks:self.tracks[o.object_id]=Track(o.object_id,o.radius_m)
            self.tracks[o.object_id].append(o)
        self.tracks={i:t for i,t in self.tracks.items() if now-t.times[-1]<=self.max_age_s+1e-9}
    def current(self,now):
        return [t for t in self.tracks.values() if now-t.times[-1]<=self.max_age_s+1e-9]
