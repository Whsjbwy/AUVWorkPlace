"""Ordinary, deterministic known-seabed navigation for a shared experiment.

This module does not know obstacle identities, schedules, or future motion.
The coarse A* route is geometric guidance, never a claim that REMUS can track
it safely. Every accepted local command still needs the original six-DOF
forward verification and the actual execution collision monitor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import itertools
import math
import time

import numpy as np

from .terrain import HeightField


class TerrainRouteError(RuntimeError):
    """No verified coarse-grid terrain route was found within the fixed budget."""


@dataclass(frozen=True)
class TerrainGuideConfig:
    mode: str = "direct"
    xy_step_m: float = 15.0
    depth_step_m: float = 3.0
    max_grade: float = 0.25
    waypoint_tolerance_m: float = 6.0
    lookahead_m: float = 25.0
    max_nodes: int = 60000

    def validate(self):
        if self.mode not in ("direct", "coarse3d_astar"):
            raise ValueError("unknown terrain guidance mode")
        for name in ("xy_step_m", "depth_step_m", "max_grade",
                     "waypoint_tolerance_m", "lookahead_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_grade > 0.5:
            raise ValueError("terrain route grade must be <= 0.5; reachability still requires REMUS validation")
        if not isinstance(self.max_nodes, int) or isinstance(self.max_nodes, bool) or self.max_nodes < 1:
            raise ValueError("max_nodes must be a positive integer")


def _point(value, name):
    point = np.asarray(value, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must be a finite NED point")
    return point.copy()


def _validate_terrain(terrain):
    for values in (terrain.xs, terrain.ys, terrain.depths):
        if not np.isfinite(values).all():
            raise ValueError("terrain arrays must be finite")
    if (np.ndim(terrain.xs) != 1 or np.ndim(terrain.ys) != 1
            or len(terrain.xs) < 2 or len(terrain.ys) < 2
            or np.any(np.diff(terrain.xs) <= 0) or np.any(np.diff(terrain.ys) <= 0)
            or terrain.depths.shape != (len(terrain.ys), len(terrain.xs))
            or not np.isfinite(terrain.slope_bound) or terrain.slope_bound < 0):
        raise ValueError("invalid terrain grid or conservative slope bound")


def _grade_ok(a, b, max_grade):
    delta = np.asarray(b) - a
    horizontal = np.linalg.norm(delta[:2])
    return abs(delta[2]) <= max_grade * horizontal + 1e-10


def _segment_gap(terrain, a, b, radius, depth_limit):
    # This is exactly the existing conservative swept-heightfield check,
    # including the surface and horizontal map boundaries. Maximum depth is
    # linear along a segment and therefore attains its worst value at an end.
    return min(terrain.segment_clearance(a, b, radius),
               float(depth_limit - radius - max(a[2], b[2])))


def _route_segments(terrain, points, radius, margin, depth_limit):
    gaps = [_segment_gap(terrain, a, b, radius, depth_limit)
            for a, b in zip(points[:-1], points[1:])]
    if gaps and min(gaps) < margin - 1e-10:
        raise TerrainRouteError("coarse route contains an edge below the unchanged safety margin")
    return min(gaps) if gaps else float(terrain.clearance(points[0], radius))


def _coarse_astar(terrain, start, goal, radius, margin, depth_limit, cfg):
    begun = time.perf_counter()
    edge_checks = 0
    if _grade_ok(start, goal, cfg.max_grade):
        edge_checks += 1
        gap = _segment_gap(terrain, start, goal, radius, depth_limit)
        if gap >= margin:
            return np.stack([start, goal]), {"expanded_nodes": 0, "edge_checks": edge_checks,
                "minimum_route_clearance_m": gap, "search_ms": (time.perf_counter()-begun)*1000}

    low = radius + margin + 0.25
    high = depth_limit - radius - margin - 0.25
    if low >= high:
        raise TerrainRouteError("no water depth remains for the requested sphere and margin")
    # A misconfigured submillimeter grid must fail before allocating enormous
    # arrays. This memory guard is separate from the disclosed expansion budget
    # and never causes an unchecked fallback or changes the safety criterion.
    estimated_shape = (max(0, int(math.ceil((terrain.xs[-1]-terrain.xs[0]-2*low)/cfg.xy_step_m))+1),
                       max(0, int(math.ceil((terrain.ys[-1]-terrain.ys[0]-2*low)/cfg.xy_step_m))+1),
                       int(math.ceil((high-low)/cfg.depth_step_m))+3)
    if math.prod(estimated_shape) > 2_000_000:
        raise TerrainRouteError("declared coarse grid exceeds the fixed 2,000,000-node allocation guard")
    # Use uniform horizontal spacing: a final short boundary cell is avoided.
    axes = [np.arange(terrain.xs[0]+low, terrain.xs[-1]-low+1e-10, cfg.xy_step_m),
            np.arange(terrain.ys[0]+low, terrain.ys[-1]-low+1e-10, cfg.xy_step_m),
            np.unique(np.r_[np.arange(low, high+1e-10, cfg.depth_step_m), start[2], goal[2]])]
    if any(len(axis) == 0 for axis in axes):
        raise TerrainRouteError("coarse grid has no nodes")
    shape = tuple(len(axis) for axis in axes)
    mesh = np.meshgrid(*axes, indexing="ij")
    all_points = np.stack(mesh, axis=-1)
    terrain_gap = terrain.clearance(all_points, radius)
    valid = (terrain_gap >= margin) & (depth_limit-radius-all_points[..., 2] >= margin)
    if not np.isfinite(terrain_gap).all():
        raise ValueError("nonfinite terrain node clearance")

    def position(node):
        return all_points[node]

    def connections(point):
        nonlocal edge_checks
        nearest = [np.argsort(abs(axis-coordinate), kind="stable")[:5]
                   for axis, coordinate in zip(axes, point)]
        result = {}
        for node_values in itertools.product(*nearest):
            node = tuple(int(i) for i in node_values)
            target = position(node)
            if not valid[node] or not _grade_ok(point, target, cfg.max_grade):
                continue
            edge_checks += 1
            if _segment_gap(terrain, point, target, radius, depth_limit) >= margin:
                result[node] = float(np.linalg.norm(target-point))
        return result

    starts, ends = connections(start), connections(goal)
    if not starts or not ends:
        raise TerrainRouteError("start or goal cannot connect to the verified coarse grid")
    queue = []
    serial = itertools.count()
    costs = dict(starts)
    parent = {node: None for node in starts}
    for node, cost in starts.items():
        heapq.heappush(queue, (cost+np.linalg.norm(position(node)-goal), next(serial), cost, node))
    neighbors = tuple(offset for offset in itertools.product((-1, 0, 1), repeat=3)
                      if (offset[0] or offset[1]))
    expanded = set()
    best_cost = float("inf")
    best_end = None
    edge_cache = {}
    while queue:
        estimate, _, cost, node = heapq.heappop(queue)
        if estimate >= best_cost - 1e-10:
            break
        if node in expanded or cost > costs[node] + 1e-10:
            continue
        if len(expanded) >= cfg.max_nodes:
            raise TerrainRouteError(f"A* exhausted the fixed {cfg.max_nodes}-node budget; no route certified")
        expanded.add(node)
        if node in ends and cost+ends[node] < best_cost:
            best_cost, best_end = cost+ends[node], node
        p = position(node)
        for offset in neighbors:
            target_node = tuple(index+delta for index, delta in zip(node, offset))
            if any(index < 0 or index >= size for index, size in zip(target_node, shape)):
                continue
            if not valid[target_node] or target_node in expanded:
                continue
            q = position(target_node)
            if not _grade_ok(p, q, cfg.max_grade):
                continue
            new_cost = cost+float(np.linalg.norm(q-p))
            if new_cost >= costs.get(target_node, float("inf")) - 1e-10:
                continue
            key = tuple(sorted((node, target_node)))
            if key not in edge_cache:
                edge_checks += 1
                edge_cache[key] = _segment_gap(terrain, p, q, radius, depth_limit) >= margin
            if not edge_cache[key]:
                continue
            costs[target_node], parent[target_node] = new_cost, node
            estimate = new_cost+float(np.linalg.norm(q-goal))
            heapq.heappush(queue, (estimate, next(serial), new_cost, target_node))
    if best_end is None:
        raise TerrainRouteError("A* found no safe route on the declared coarse grid")
    nodes = []
    node = best_end
    while node is not None:
        nodes.append(position(node).copy())
        node = parent[node]
    route = np.vstack([start, *reversed(nodes), goal])
    route = route[np.r_[True, np.linalg.norm(np.diff(route, axis=0), axis=1) > 1e-10]]
    # Deterministic line-of-sight shortening. A shortcut is accepted only after
    # the same original swept check, with the unchanged sphere and margin.
    shortened = [route[0]]
    index = 0
    while index < len(route)-1:
        selected = index+1
        for end in range(len(route)-1, index, -1):
            if not _grade_ok(route[index], route[end], cfg.max_grade):
                continue
            edge_checks += 1
            if _segment_gap(terrain, route[index], route[end], radius, depth_limit) >= margin:
                selected = end
                break
        shortened.append(route[selected])
        index = selected
    points = np.stack(shortened)
    gap = _route_segments(terrain, points, radius, margin, depth_limit)
    return points, {"expanded_nodes": len(expanded), "edge_checks": edge_checks,
        "minimum_route_clearance_m": gap, "grid_shape": list(shape),
        "raw_waypoints": len(route), "search_ms": (time.perf_counter()-begun)*1000}


class TerrainGuidance:
    """Monotone, actual-position-based progress over a geometric terrain route.

    This object cannot mark a navigation task successful. The execution loop
    must continue evaluating distance to ``final_goal`` using its original
    arrival criterion, including when an intermediate waypoint is reached.
    """
    def __init__(self, terrain, path, final_goal, radius, margin, depth_limit, config, metrics):
        self.terrain = terrain
        self.path = np.asarray(path, dtype=float).copy()
        self.path.setflags(write=False)
        self.final_goal = _point(final_goal, "final_goal")
        self.final_goal.setflags(write=False)
        self.config, self.metrics = config, dict(metrics)
        self.radius, self.margin, self.depth_limit = radius, margin, depth_limit
        lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self._lengths = lengths
        self._distances = np.r_[0., np.cumsum(lengths)]
        self._progress = 0.0
        self._segment = 0
        self.unreachable_target_count = 0

    @property
    def progress_m(self):
        return self._progress

    def _at_distance(self, distance):
        if distance >= self._distances[-1]-1e-10:
            return self.final_goal.copy()
        index = min(len(self._lengths)-1, max(0, int(np.searchsorted(self._distances, distance, side="right")-1)))
        fraction = (distance-self._distances[index])/max(self._lengths[index], 1e-12)
        return self.path[index]+fraction*(self.path[index+1]-self.path[index])

    def current_target(self, actual_position):
        actual = _point(actual_position, "actual_position")
        if self.config.mode == "direct" or len(self.path) <= 1:
            return self.final_goal.copy()
        # Consider only the current and immediately adjacent route segment.
        # Cross-segment lookahead may round a corner outside the waypoint ball;
        # requiring that ball forever would lock progress and leave the target
        # behind an actually advancing vehicle. A distant self-near route leg
        # must not become a shortcut over unvisited intermediate corners.
        while self._segment < len(self._lengths):
            start, end = self.path[self._segment:self._segment+2]
            length = self._lengths[self._segment]
            fraction = float(np.clip(np.dot(actual-start, end-start)/max(length**2, 1e-12), 0, 1))
            self._progress = max(self._progress, self._distances[self._segment]+fraction*length)
            advance = np.linalg.norm(actual-end) <= self.config.waypoint_tolerance_m
            if not advance and self._segment < len(self._lengths)-1:
                following = self.path[self._segment+2]
                next_delta = following-end
                next_fraction = float(np.clip(np.dot(actual-end, next_delta)
                    /max(np.dot(next_delta, next_delta), 1e-12), 0, 1))
                next_projection = end+next_fraction*next_delta
                current_projection = start+fraction*(end-start)
                nearby_corner = np.linalg.norm(actual-end) <= self.config.lookahead_m
                passed_plane = np.dot(actual-end, end-start) >= 0
                joined_next = (next_fraction > 0 and
                    np.linalg.norm(actual-next_projection) < np.linalg.norm(actual-current_projection)-1e-10)
                # This is a local, actually reached transition. The same
                # original swept terrain check must permit joining the next
                # segment; neither lookahead nor proximity certifies safety.
                advance = bool(nearby_corner and (passed_plane or joined_next)
                    and _segment_gap(self.terrain, actual, next_projection,
                        self.radius, self.depth_limit) >= self.margin)
            if advance and self._segment < len(self._lengths)-1:
                self._segment += 1
                self._progress = max(self._progress, self._distances[self._segment])
                continue
            break
        desired = min(self._distances[-1], self._progress+self.config.lookahead_m)
        # A geometric shortcut from the actual deviated state is checked too.
        # Reducing lookahead may preserve a nearby safe target. If the vehicle
        # has left the route's safe tube, return the next route point and expose
        # this fact; the REMUS planner must reject unsafe executions itself.
        candidates = [desired]
        candidates.extend(float(distance) for distance in self._distances
                          if self._progress+1e-9 < distance < desired-1e-9)
        for distance in sorted(set(candidates), reverse=True):
            target = self._at_distance(distance)
            if _segment_gap(self.terrain, actual, target, self.radius, self.depth_limit) >= self.margin:
                return target
        self.unreachable_target_count += 1
        index = min(self._segment+1, len(self.path)-1)
        return self.path[index].copy()


def build_terrain_guidance(terrain, start, goal, *, body_radius_m,
                           safety_margin_m, depth_limit_m, config=None):
    config = config or TerrainGuideConfig()
    config.validate()
    _validate_terrain(terrain)
    start, goal = _point(start, "start"), _point(goal, "goal")
    values = np.array([body_radius_m, safety_margin_m, depth_limit_m], dtype=float)
    if (not np.isfinite(values).all() or body_radius_m <= 0 or safety_margin_m < 0
            or depth_limit_m <= 0 or depth_limit_m > 100):
        raise ValueError("terrain route requires a positive sphere, nonnegative margin and depth limit <=100")
    for point in (start, goal):
        if min(float(terrain.clearance(point, body_radius_m)), depth_limit_m-body_radius_m-point[2]) < safety_margin_m:
            raise TerrainRouteError("start or goal violates the original terrain/depth clearance")
    if config.mode == "direct":
        points = np.stack([start, goal])
        metrics = {"expanded_nodes": 0, "edge_checks": 0, "search_ms": 0.,
                   "route_verified_safe": False, "reason": "original direct final-goal guidance"}
    else:
        points, metrics = _coarse_astar(terrain, start, goal, body_radius_m,
                                       safety_margin_m, depth_limit_m, config)
        metrics["route_verified_safe"] = True
    metrics.update({"guidance": config.mode, "config": asdict(config),
                    "route_length_m": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
                    "waypoints": len(points),
                    "safety_margin_m": float(safety_margin_m),
                    "validation_scope": "known terrain geometry only; not dynamic obstacles or REMUS reachability",
                    "obstacle_future_information_used": False})
    return TerrainGuidance(terrain, points, goal, body_radius_m, safety_margin_m,
                           depth_limit_m, config, metrics)


def make_constrained_perlin_scene(cfg):
    """A predeclared 180-m terrain-constrained route on the original Perlin map.

    Scene selection uses only the seeded seabed, never an algorithm run. Both
    route ends have fixed depth 70 m and are 180 m apart along a map axis. Among
    all admissible pairs on the fixed 25-m center grid, choose the pair closest
    to the map center (then north, east, axis), requiring safe endpoints and an
    unsafe straight line. No seed retry or method-dependent placement occurs.
    Dynamic obstacles are omitted here to isolate the terrain experiment; the
    existing paper_random3d scene remains untouched.
    """
    from .config import TerrainConfig
    from .obstacles import LinearObstacles
    from .scenarios import Scene

    cfg.validate()
    terrain_cfg = TerrainConfig()
    terrain = HeightField.perlin(terrain_cfg, cfg.seed)
    fixed_depth, half_length, center_step = 70.0, 90.0, 25.0
    map_center = np.array([terrain_cfg.width/2, terrain_cfg.height/2])
    centers = itertools.product(np.arange(100., terrain_cfg.width-99., center_step),
                                np.arange(100., terrain_cfg.height-99., center_step), (0, 1))
    ordered = sorted(centers, key=lambda item: (np.sum((np.array(item[:2])-map_center)**2), *item))
    examined = 0
    for north, east, axis in ordered:
        center = np.array([north, east, fixed_depth])
        offset = np.zeros(3)
        offset[axis] = half_length
        start, goal = center-offset, center+offset
        examined += 1
        if any(min(float(terrain.clearance(p, cfg.body_radius_m)),
                   cfg.depth_limit_m-cfg.body_radius_m-p[2]) <= cfg.safety_margin_m for p in (start, goal)):
            continue
        straight_gap = _segment_gap(terrain, start, goal, cfg.body_radius_m, cfg.depth_limit_m)
        if straight_gap >= cfg.safety_margin_m:
            continue
        yaw = math.atan2(goal[1]-start[1], goal[0]-start[0])
        return Scene("perlin_constrained3d", terrain, LinearObstacles([], [], []),
                     np.r_[start, 0., 0., yaw], goal,
                     {"coordinates": "NED; depth positive down; eta angles in radians",
                      "method_independent_scene": True, "obstacle_count": 0,
                      "scene_source": "original TerrainConfig/HeightField.perlin; original paper_random3d is unchanged",
                      "scenario_seed": cfg.seed, "terrain_config": asdict(terrain_cfg),
                      "selection_rule": "safe fixed-depth 180-m axis pairs on a 25-m center grid; first straight-line-blocked pair sorted by center distance, north, east, axis",
                      "fixed_depth_m": fixed_depth, "route_chord_m": 2*half_length,
                      "candidate_pairs_examined": examined,
                      "straight_line_clearance_m": straight_gap,
                      "terrain_constraint": "straight line violates the original swept sphere plus 1-m criterion; known seabed guidance may route above or around it",
                      "motion_information": "no dynamic obstacles in this terrain-isolation scene",
                      "obstacle_future_information_used": False})
    raise TerrainRouteError("this seed has no admissible terrain-constrained 180-m pair under the predeclared selection rule; seed was not changed")


def make_cruise_constrained_perlin_scene(cfg):
    """Independent Perlin protocol with original-model initialization admission.

    Preserve ``perlin_constrained3d`` and its recorded initialization failure.
    This new version uses the same fixed geometric pair order, then requires
    an unchanged-REMUS neutral cruise over the fixed 24-second horizon to
    satisfy the original swept sphere/margin/depth checks. Selection never
    calls DWA or an obstacle predictor and never consults navigation outcomes.
    Admission is finite-horizon initialization evidence, not a guarantee of
    entire-task reachability or dynamic-obstacle avoidance.
    """
    from .adapters.remus_guidance import ReferenceAction, predict_reference
    from .obstacles import LinearObstacles
    from .scenarios import Scene
    from .stage2_simulation import initialize_plant

    cfg.validate()
    initialization_horizon_s = 24.0
    original = make_constrained_perlin_scene(cfg)
    terrain = original.terrain
    fixed_depth = original.metadata["fixed_depth_m"]
    half_length = original.metadata["route_chord_m"]/2
    map_center = np.array([(terrain.xs[0]+terrain.xs[-1])/2,
                           (terrain.ys[0]+terrain.ys[-1])/2])
    centers = itertools.product(np.arange(100., terrain.xs[-1]-99., 25.),
                                np.arange(100., terrain.ys[-1]-99., 25.), (0, 1))
    ordered = sorted(centers, key=lambda item: (np.sum((np.array(item[:2])-map_center)**2), *item))
    admissions = []
    for examined, (north, east, axis) in enumerate(ordered, 1):
        center = np.array([north, east, fixed_depth])
        offset = np.zeros(3)
        offset[axis] = half_length
        start, goal = center-offset, center+offset
        if any(min(float(terrain.clearance(p, cfg.body_radius_m)),
                   cfg.depth_limit_m-cfg.body_radius_m-p[2]) <= cfg.safety_margin_m for p in (start, goal)):
            continue
        straight_gap = _segment_gap(terrain, start, goal, cfg.body_radius_m, cfg.depth_limit_m)
        if straight_gap >= cfg.safety_margin_m:
            continue
        yaw = math.atan2(goal[1]-start[1], goal[0]-start[0])
        scene = Scene("perlin_cruise_constrained3d", terrain, LinearObstacles([], [], []),
                      np.r_[start, 0., 0., yaw], goal)
        plant = initialize_plant(scene, cfg)
        neutral = ReferenceAction(float(start[2]), yaw, cfg.nominal_rpm)
        predicted = predict_reference(plant, neutral, initialization_horizon_s, cfg.physics_dt,
                                      record_controls=False)
        positions = predicted.positions
        point_gap = min(float(terrain.clearance(positions, cfg.body_radius_m).min()),
                        float((cfg.depth_limit_m-cfg.body_radius_m-positions[:, 2]).min()))
        segment_gap = min(_segment_gap(terrain, a, b, cfg.body_radius_m, cfg.depth_limit_m)
                          for a, b in zip(positions[:-1], positions[1:]))
        gap = min(point_gap, segment_gap)
        if not np.isfinite(gap):
            raise FloatingPointError("neutral-cruise initialization check produced nonfinite clearance")
        admission = {"pair_index": examined, "start_ned_m": start.tolist(),
                     "goal_ned_m": goal.tolist(), "straight_line_clearance_m": straight_gap,
                     "neutral_cruise_clearance_m": gap,
                     "initialization_cruise_verified": bool(gap >= cfg.safety_margin_m)}
        admissions.append(admission)
        if gap < cfg.safety_margin_m:
            continue
        scene.metadata = {
            **original.metadata,
            "scene_source": "independent initialization-admitted protocol on the unchanged original Perlin map; old failed perlin_constrained3d retained",
            "selection_rule": "same fixed geometric pair order; additionally admit original-REMUS neutral cruise over a fixed 24-second initialization horizon before selecting the first pair",
            "candidate_pairs_examined": examined,
            "straight_line_clearance_m": straight_gap,
            "initialization_cruise_verified": True,
            "initialization_validation_scope": "original-model neutral cruise over a fixed 24-second horizon only; no entire-task reachability or dynamic-obstacle guarantee",
            "initialization_cruise_horizon_s": initialization_horizon_s,
            "initialization_physics_dt_s": cfg.physics_dt,
            "initialization_warmup_s": cfg.warmup_s,
            "initialization_cruise_reference": neutral.as_array().tolist(),
            "initialization_cruise_clearance_m": gap,
            "initialization_speed_mps": float(plant.nu[0]),
            "initialization_checked_physical_intervals": len(positions)-1,
            "initialization_admission_attempts": admissions,
            "planner_or_predictor_used_for_scene_selection": False,
        }
        return scene
    raise TerrainRouteError("no fixed Perlin pair passed original-model neutral-cruise initialization admission; no seed or geometry was changed")
