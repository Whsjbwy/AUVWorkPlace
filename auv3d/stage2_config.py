"""Independent engineering settings for the REMUS closed-loop baseline.

These values configure the experiment and the reference-command sampler.  They
do not replace any coefficient, actuator limit or time constant in REMUS 100.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from .config import SonarConfig


@dataclass(frozen=True)
class Stage2Config:
    physics_dt: float = 0.02
    control_dt: float = 2.0
    command_delay_s: float = 0.0
    horizon_s: float = 24.0
    prediction_output_dt: float = 0.2
    candidate_budget: int = 9
    nominal_rpm: float = 1100.0
    rpm_values: tuple[float, ...] = (800.0, 1100.0, 1400.0)
    heading_offsets_deg: tuple[float, ...] = (-55.0, -30.0, 0.0, 30.0, 55.0)
    depth_offsets_m: tuple[float, ...] = (-8.0, 0.0, 8.0)
    body_radius_m: float = math.hypot(0.8, 0.095)
    depth_limit_m: float = 100.0
    safety_margin_m: float = 1.0
    prediction_margin_m: float = 0.75
    emergency_rpm: float = 0.0
    goal_weight: float = 1.0
    clearance_weight: float = 0.8
    speed_weight: float = 0.35
    smooth_weight: float = 0.1
    max_time_s: float = 200.0
    goal_tolerance_m: float = 5.0
    current_speed: float = 0.0
    current_direction_deg: float = 90.0
    sensor_range_m: float = 150.0
    sensor_hfov_deg: float = 120.0
    sensor_vfov_deg: float = 30.0
    observation_period_s: float = 1.0
    observation_delay_s: float = 0.0
    sensor_noise_std_m: float = 0.0
    sensor_terrain_occlusion: bool = True
    track_max_age_s: float = 4.0
    seed: int = 7
    warmup_s: float = 20.0
    interaction_distance_m: float = 30.0
    obstacle_speed_scale: float = 1.0
    obstacle_turn_rate_deg_s: float = 4.0
    obstacle_radius_m: float = 4.0
    scenario_width_m: float = 260.0
    scenario_height_m: float = 200.0
    terrain_depth_m: float = 90.0
    terrain_amplitude_m: float = 60.0
    terrain_grid_spacing_m: float = 2.0
    paper_obstacle_count: int = 120
    label: str = "stage2_remus6dof_engineering_baseline"

    @property
    def planning_dt(self) -> float:
        return self.control_dt

    @property
    def shortlist_count(self) -> int:
        return self.candidate_budget

    @property
    def rpm_choices(self) -> tuple[float, ...]:
        return self.rpm_values

    def sensor_config(self) -> SonarConfig:
        return SonarConfig(range_m=self.sensor_range_m,
                           hfov_deg=self.sensor_hfov_deg,
                           vfov_deg=self.sensor_vfov_deg,
                           period_s=self.observation_period_s,
                           position_noise_std_m=self.sensor_noise_std_m,
                           terrain_occlusion=self.sensor_terrain_occlusion)

    def validate(self) -> None:
        positive = (
            "physics_dt", "control_dt", "horizon_s", "prediction_output_dt",
            "body_radius_m", "depth_limit_m", "max_time_s", "goal_tolerance_m",
            "sensor_range_m", "observation_period_s", "track_max_age_s",
            "interaction_distance_m", "obstacle_radius_m", "scenario_width_m",
            "scenario_height_m", "terrain_depth_m", "terrain_grid_spacing_m",
        )
        nonnegative = (
            "safety_margin_m", "prediction_margin_m", "current_speed",
            "observation_delay_s", "sensor_noise_std_m", "warmup_s",
            "command_delay_s",
            "obstacle_speed_scale", "obstacle_turn_rate_deg_s",
            "terrain_amplitude_m", "goal_weight", "clearance_weight",
            "speed_weight", "smooth_weight",
        )
        for name in positive + nonnegative:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name in positive and value == 0):
                raise ValueError(f"invalid {name}")
        if self.physics_dt > 0.02:
            raise ValueError("REMUS physics_dt must be <= 0.02 s")
        if self.horizon_s < self.control_dt:
            raise ValueError("horizon_s must cover at least one control interval")
        if self.command_delay_s + self.control_dt > self.horizon_s + 1e-9:
            raise ValueError("horizon_s must cover command_delay_s plus control_dt")
        for name in ("control_dt", "prediction_output_dt", "observation_period_s",
                     "observation_delay_s", "command_delay_s", "warmup_s", "max_time_s"):
            ratio = getattr(self, name) / self.physics_dt
            if not math.isclose(ratio, round(ratio), abs_tol=1e-8):
                raise ValueError(f"{name} must align with physics_dt")
        ratio = self.horizon_s / self.prediction_output_dt
        if not math.isclose(ratio, round(ratio), abs_tol=1e-8):
            raise ValueError("horizon_s must align with prediction_output_dt")
        if not 0 < self.depth_limit_m <= 100:
            raise ValueError("REMUS depth_limit_m must remain <= 100 m")
        if self.depth_limit_m <= 2 * (self.body_radius_m + self.safety_margin_m):
            raise ValueError("depth envelope must contain the vehicle and its safety margins")
        for name in ("nominal_rpm", "emergency_rpm"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1525:
                raise ValueError(f"{name} must be in REMUS's [0, 1525] RPM envelope")
        for name in ("rpm_values", "heading_offsets_deg", "depth_offsets_m"):
            values = getattr(self, name)
            if not values or any(not math.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain finite samples")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} contains duplicate samples")
        if any(not 0 <= rpm <= 1525 for rpm in self.rpm_values):
            raise ValueError("rpm_values exceed the original REMUS limit")
        if any(abs(a) > 180 for a in self.heading_offsets_deg):
            raise ValueError("heading_offsets_deg must be within +/-180 degrees")
        if not math.isfinite(self.current_direction_deg):
            raise ValueError("current_direction_deg must be finite")
        for name in ("sensor_hfov_deg", "sensor_vfov_deg"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 180:
                raise ValueError(f"{name} must be in (0, 180]")
        if not isinstance(self.candidate_budget, int) or isinstance(self.candidate_budget, bool) or self.candidate_budget < 1:
            raise ValueError("candidate_budget must be a positive integer")
        if not isinstance(self.paper_obstacle_count, int) or isinstance(self.paper_obstacle_count, bool) or self.paper_obstacle_count < 0:
            raise ValueError("paper_obstacle_count must be a nonnegative integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.terrain_amplitude_m >= self.terrain_depth_m:
            raise ValueError("terrain_amplitude_m must be smaller than terrain_depth_m")
        # Controlled scenes contain a 180 m route plus space for turning.
        if self.scenario_width_m < 240 or self.scenario_height_m < 180:
            raise ValueError("controlled scenes require at least 240 x 180 m")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "Stage2Config":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("_provenance", None)
        for name in ("rpm_values", "heading_offsets_deg", "depth_offsets_m"):
            if name in data:
                data[name] = tuple(data[name])
        cfg = cls(**data)
        cfg.validate()
        return cfg
