from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json
import math
import numpy as np

@dataclass(frozen=True)
class TerrainConfig:
    width: float = 1000.0
    height: float = 1000.0
    max_depth: float = 200.0
    amplitude: float = 150.0
    scale: float = 150.0
    octaves: int = 2
    exponent: int = 3
    grid_spacing: float = 5.0  # reconstruction choice, NOT a paper value

@dataclass(frozen=True)
class SonarConfig:
    range_m: float = 150.0
    hfov_deg: float = 90.0
    vfov_deg: float = 6.0
    horizontal_resolution_deg: float = 1.0
    vertical_resolution_deg: float = 6.0
    period_s: float = 1.0  # paper does not specify physical delta-t
    position_noise_std_m: float = 0.0
    terrain_occlusion: bool = True

@dataclass(frozen=True)
class ObstacleConfig:
    count: int = 120
    min_separation_m: float = 25.0
    radius_min_m: float = 2.0
    radius_max_m: float = 12.0
    speed_min: float = 0.2
    speed_max: float = 2.5
    pitch_max_deg: float = 30.0
    acceleration_max: float = 0.5
    q_max_deg_s: float = 8.0
    r_max_deg_s: float = 15.0
    target_period_s: float = 15.0
    kp: tuple[float, ...] = (0.6, 1.0, 0.8)
    ki: tuple[float, ...] = (0.08, 0.15, 0.10)

@dataclass(frozen=True)
class VehicleConfig:
    # AUV command limits below are DEVELOPMENT choices, NOT obstacle Table 2.
    min_speed: float = 0.0
    max_speed: float = 2.5
    acceleration_max: float = 0.3
    pitch_max_deg: float = 30.0
    q_max_deg_s: float = 4.0
    r_max_deg_s: float = 5.0
    angular_acceleration_deg_s2: float = 4.0
    body_radius_m: float = math.hypot(0.8, 0.095)  # REMUS enclosing sphere
    depth_limit_m: float = 100.0  # preserve upstream operating restriction

@dataclass(frozen=True)
class PlannerConfig:
    control_dt: float = 1.0
    integration_dt: float = 0.05
    rollout_dt: float = 0.2
    horizon_s: float = 9.0  # 3 future prediction steps at provisional 3-s sampling
    speed_samples: int = 5
    pitch_samples: int = 7
    yaw_samples: int = 9
    goal_weight: float = 1.0
    clearance_weight: float = 0.8
    speed_weight: float = 0.35
    smooth_weight: float = 0.1
    safety_margin_m: float = 1.0
    base_prediction_margin_m: float = 0.75  # debug only, not 95%-calibrated
    track_max_age_s: float = 3.0
    goal_tolerance_m: float = 5.0
    max_time_s: float = 250.0

@dataclass(frozen=True)
class ExperimentConfig:
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    sonar: SonarConfig = field(default_factory=SonarConfig)
    obstacles: ObstacleConfig = field(default_factory=ObstacleConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    seed: int = 7
    current_ned_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    prediction_dt_s: float = 3.0  # not supplied in P25; independent of sensor update period
    label: str = "stage1_development_not_exact_reproduction"

    def validate(self) -> None:
        t, s, o, v, p = self.terrain, self.sonar, self.obstacles, self.vehicle, self.planner
        def positive(x, name):
            if not math.isfinite(float(x)) or x <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for n in ('width','height','max_depth','amplitude','scale','grid_spacing'):
            positive(getattr(t,n), f'terrain.{n}')
        if not 0 < t.amplitude < t.max_depth or t.octaves < 1 or t.exponent < 1:
            raise ValueError("invalid terrain amplitude/octaves/exponent")
        if not 0 < s.hfov_deg <= 180 or not 0 < s.vfov_deg <= 180:
            raise ValueError("sonar full angles must be in (0,180]")
        positive(s.range_m,'range'); positive(s.period_s,'observation period')
        if not math.isfinite(s.position_noise_std_m) or s.position_noise_std_m < 0: raise ValueError('invalid noise')
        if not 0 <= v.min_speed < v.max_speed: raise ValueError('invalid AUV speed')
        if not 0 < v.pitch_max_deg < 85: raise ValueError('pitch singularity guard')
        for n in ('acceleration_max','q_max_deg_s','r_max_deg_s','angular_acceleration_deg_s2','body_radius_m','depth_limit_m'):
            positive(getattr(v,n), f'vehicle.{n}')
        if o.count < 0 or not 0 <= o.speed_min <= o.speed_max: raise ValueError('invalid obstacles')
        if not 0 < o.radius_min_m <= o.radius_max_m: raise ValueError('invalid obstacle radius')
        for n in ('control_dt','integration_dt','rollout_dt','horizon_s','max_time_s','goal_tolerance_m'):
            positive(getattr(p,n), f'planner.{n}')
        for d in (p.integration_dt,p.rollout_dt):
            if abs(p.control_dt/d-round(p.control_dt/d)) > 1e-8:
                raise ValueError('control_dt must be an integer multiple of integration/rollout dt')
        if abs(p.horizon_s/p.rollout_dt-round(p.horizon_s/p.rollout_dt)) > 1e-8:
            raise ValueError('horizon must contain an integer number of rollout steps')
        if p.horizon_s < p.control_dt: raise ValueError('horizon < control interval')
        if any(not isinstance(n,int) or n<2 for n in (p.speed_samples,p.pitch_samples,p.yaw_samples)): raise ValueError('invalid candidate sample counts')
        for n in ('goal_weight','clearance_weight','speed_weight','smooth_weight','safety_margin_m','base_prediction_margin_m','track_max_age_s'):
            value=getattr(p,n)
            if not math.isfinite(value) or value<0: raise ValueError(f'invalid planner.{n}')
        positive(self.prediction_dt_s,'prediction_dt_s')
        if abs(p.horizon_s-3*self.prediction_dt_s)>1e-8:
            raise ValueError('stage 1 requires exactly 3 prediction steps covering the DWA horizon')
        if len(self.current_ned_mps)!=3 or not np.isfinite(self.current_ned_mps).all():
            raise ValueError('current requires 3 finite NED components')

    def to_dict(self): return asdict(self)

    @classmethod
    def load(cls, path: str | Path):
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        data.pop('_provenance',None)
        types={'terrain':TerrainConfig,'sonar':SonarConfig,'obstacles':ObstacleConfig,
               'vehicle':VehicleConfig,'planner':PlannerConfig}
        for name,typ in types.items():
            if name in data: data[name]=typ(**data[name])
        cfg=cls(**data); cfg.validate(); return cfg
