"""Shared deterministic ground truth for the two stage 2 backends.

Only the sensor receives obstacle snapshots.  A trajectory planner must receive
observations/tracks, never this module's motion schedule or future snapshots.
Stage 1's original scene generator remains available and unmodified.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import numpy as np

from .config import ExperimentConfig
from .obstacles import LinearObstacles, ObstacleSnapshot
from .stage2_config import Stage2Config
from .terrain import HeightField


STAGE2_SCENARIOS = (
    "empty3d", "static3d", "head_on3d", "crossing3d", "multi_crossing3d",
    "maneuvering3d", "terrain3d", "tracking_mismatch3d", "paper_random3d",
)
SCENARIOS = STAGE2_SCENARIOS


@dataclass
class Scene:
    name: str
    terrain: HeightField
    obstacles: object
    start_eta: np.ndarray  # N,E,D,roll,pitch,yaw; radians, NED positive down
    goal: np.ndarray
    metadata: dict = field(default_factory=dict)


class ManeuveringObstacles:
    """Analytic curved ground truth with speed and heading changes.

    Positions are evaluated at the accumulated simulation time, so dividing a
    physics step does not change their path.  This is a controlled stress case,
    not an inferred obstacle model or P25 Table 2 parameters.  Transiting objects
    may leave the scene; no unmodelled reflection is imposed at the boundary.
    """
    def __init__(self, positions, velocities, radii, turn_rate_deg_s=4.0,
                 amplitudes_m=None, phases=None):
        self.initial_positions = np.asarray(positions, float).reshape(-1, 3)
        self.base_velocities = np.asarray(velocities, float).reshape(-1, 3)
        self.radii = np.asarray(radii, float).reshape(-1)
        n = len(self.radii)
        if self.initial_positions.shape != self.base_velocities.shape or n != len(self.initial_positions):
            raise ValueError("obstacle array lengths differ")
        if not np.isfinite(np.r_[self.initial_positions.ravel(), self.base_velocities.ravel(), self.radii]).all() or np.any(self.radii <= 0):
            raise ValueError("obstacles must have finite positions/velocities and positive radii")
        if not np.isfinite(turn_rate_deg_s) or turn_rate_deg_s < 0:
            raise ValueError("turn_rate_deg_s must be finite and nonnegative")
        self.omega = float(np.deg2rad(turn_rate_deg_s))
        self.amplitudes = np.full(n, 8.0) if amplitudes_m is None else np.asarray(amplitudes_m, float).reshape(n)
        self.phases = np.zeros(n) if phases is None else np.asarray(phases, float).reshape(n)
        speed = np.linalg.norm(self.base_velocities[:, :2], axis=1)
        direction = self.base_velocities[:, :2] / np.maximum(speed[:, None], 1e-12)
        self.lateral = np.column_stack([-direction[:, 1], direction[:, 0], np.zeros(n)])
        self.t = 0.0
        self.positions = self.initial_positions.copy()
        self.velocities = self._velocity(0.0)

    def _position(self, t):
        excursion = self.amplitudes * (np.sin(self.omega * t + self.phases) - np.sin(self.phases))
        return self.initial_positions + t * self.base_velocities + excursion[:, None] * self.lateral

    def _velocity(self, t):
        rate = self.amplitudes * self.omega * np.cos(self.omega * t + self.phases)
        return self.base_velocities + rate[:, None] * self.lateral

    def snapshot(self):
        return ObstacleSnapshot(np.arange(len(self.radii)), self.positions.copy(), self.radii.copy())

    def step(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("obstacle dt must be finite and positive")
        self.t += dt
        self.positions = self._position(self.t)
        self.velocities = self._velocity(self.t)


def _ridge_terrain(cfg):
    xs = np.linspace(0, cfg.scenario_width_m,
                     int(math.ceil(cfg.scenario_width_m / cfg.terrain_grid_spacing_m)) + 1)
    ys = np.linspace(0, cfg.scenario_height_m,
                     int(math.ceil(cfg.scenario_height_m / cfg.terrain_grid_spacing_m)) + 1)
    x, y = np.meshgrid(xs, ys)
    # A broad finite ridge leaves both an upper-water and a lateral passage.
    ridge = np.exp(-0.5 * ((x - 130) / 27) ** 2 - 0.5 * ((y - 100) / 42) ** 2)
    depth = cfg.terrain_depth_m - cfg.terrain_amplitude_m * ridge
    depth += 0.025 * cfg.terrain_amplitude_m * np.sin(x / 35) * np.sin(y / 31)
    gx = np.diff(depth, axis=1) / np.diff(xs)[None, :]
    gy = np.diff(depth, axis=0) / np.diff(ys)[:, None]
    bound = float(np.hypot(np.abs(gx).max(), np.abs(gy).max()))
    return HeightField(xs, ys, depth, bound)


def make_stage2_scenario(name: str, cfg: Stage2Config) -> Scene:
    cfg.validate()
    if name not in STAGE2_SCENARIOS:
        raise ValueError(f"unknown stage 2 scenario {name}")
    metadata = {
        "coordinates": "NED; depth positive down; eta angles in radians",
        "obstacle_motion_information": "truth reserved for sensor/collision/metrics; no future schedule supplied to planner",
        "initialization": "same requested start pose for every backend; REMUS warmup is configured separately",
        "obstacle_speed_scale": cfg.obstacle_speed_scale,
        "obstacle_turn_rate_parameter_deg_s": cfg.obstacle_turn_rate_deg_s,
        "boundary_rule": "controlled transit objects may leave the domain; no reflection",
        "method_independent_scene": True,
    }
    start_eta = np.array([30.0, 100.0, 35.0, 0.0, 0.0, 0.0])
    goal = np.array([210.0, 100.0, 35.0])
    terrain = HeightField.flat(cfg.scenario_width_m, cfg.scenario_height_m, cfg.terrain_depth_m)
    radius = cfg.obstacle_radius_m
    s = cfg.obstacle_speed_scale
    if name in ("empty3d", "tracking_mismatch3d"):
        obstacles = LinearObstacles([], [], [])
        if name == "tracking_mismatch3d":
            goal = np.array([210.0, 125.0, 43.0])
            metadata.update({
                "diagnostic_only": True,
                "diagnostic": "compare an instantaneous ideal velocity/heading change with actuator-limited REMUS reference tracking",
                "reference_step_test": {"time_s": 20.0, "duration_s": 8.0,
                                        "heading_offset_deg": 70.0, "depth_offset_m": 8.0,
                                        "rpm": 1400.0},
                "safety_claim": "none; an open reachable scene exposes tracking error without inserting an unavoidable obstacle",
            })
    elif name == "static3d":
        obstacles = LinearObstacles([[115.0, 100.0, 35.0]], [[0, 0, 0]], [radius])
    elif name == "head_on3d":
        obstacles = LinearObstacles([[170.0, 100.0, 35.0]], [[-1.0 * s, 0, 0]], [radius])
    elif name == "crossing3d":
        obstacles = LinearObstacles([[110.0, 135.0, 35.0]], [[0, -0.8 * s, 0]], [radius])
    elif name == "multi_crossing3d":
        obstacles = LinearObstacles([[100, 132, 35], [115, 62, 35], [128, 142, 35]],
                                    [[0, -0.8 * s, 0], [0, 0.85 * s, 0], [0, -0.8 * s, 0]],
                                    [radius] * 3)
        metadata["nominal_interaction_window_s"] = [35.0, 58.0]
    elif name == "maneuvering3d":
        obstacles = ManeuveringObstacles([[105, 135, 35], [128, 59, 38], [150, 140, 32]],
                                        [[0, -0.8 * s, 0], [0, 0.85 * s, 0], [0, -0.65 * s, 0]],
                                        [radius] * 3,
                                        turn_rate_deg_s=cfg.obstacle_turn_rate_deg_s,
                                        amplitudes_m=[8 * s, 7 * s, 10 * s],
                                        phases=[0, np.pi / 3, 2 * np.pi / 3])
    elif name == "terrain3d":
        terrain = _ridge_terrain(cfg)
        obstacles = LinearObstacles([], [], [])
        metadata["terrain_constraint"] = "finite ridge has a safe upper-water passage and side passages; original conservative terrain clearance is retained"
    else:  # original full-size scene; leave all stage 1 entry points unchanged
        from .simulation import make_scenario
        legacy = ExperimentConfig(seed=cfg.seed)
        legacy = replace(legacy, obstacles=replace(legacy.obstacles,
                          count=cfg.paper_obstacle_count,
                          speed_min=legacy.obstacles.speed_min * s,
                          speed_max=legacy.obstacles.speed_max * s))
        terrain, obstacles, state, goal = make_scenario("paper_random3d", legacy)
        start_eta = np.array([*state[:3], 0.0, state[4], state[5]])
        metadata.update({"scene_source": "unchanged stage 1 generator, same seed and terrain settings",
                         "boundary_rule": "original PI obstacle retreat; boundary events recorded",
                         "obstacle_count": cfg.paper_obstacle_count})
    snap = obstacles.snapshot()
    for label, point in (("start", start_eta[:3]), ("goal", goal)):
        if terrain.clearance(point, cfg.body_radius_m) <= cfg.safety_margin_m:
            raise ValueError(f"{name} {label} violates terrain clearance")
        if not cfg.body_radius_m < point[2] < cfg.depth_limit_m - cfg.body_radius_m:
            raise ValueError(f"{name} {label} violates REMUS depth envelope")
        if len(snap.ids) and np.any(np.linalg.norm(snap.positions - point, axis=1) <= snap.radii + cfg.body_radius_m + cfg.safety_margin_m):
            raise ValueError(f"{name} {label} initially overlaps an obstacle")
    metadata["obstacle_count"] = len(snap.ids)
    return Scene(name, terrain, obstacles, start_eta, goal, metadata)
