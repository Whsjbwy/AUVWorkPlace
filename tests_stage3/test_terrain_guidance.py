"""Geometric guidance tests do not establish closed-loop six-DOF safety."""
from dataclasses import replace

import numpy as np
import pytest

from auv3d.scenarios import make_stage2_scenario
from auv3d.stage2_config import Stage2Config
from auv3d.stage3_terrain import (
    TerrainGuideConfig, TerrainRouteError, build_terrain_guidance,
    make_constrained_perlin_scene,
)
from auv3d.terrain import HeightField


def build(scene, config=None):
    cfg = Stage2Config()
    return build_terrain_guidance(scene.terrain, scene.start_eta[:3], scene.goal,
        body_radius_m=cfg.body_radius_m, safety_margin_m=cfg.safety_margin_m,
        depth_limit_m=cfg.depth_limit_m, config=config)


@pytest.fixture(scope="module")
def ridge_route():
    scene = make_stage2_scenario("terrain3d", Stage2Config())
    return scene, build(scene, TerrainGuideConfig(mode="coarse3d_astar"))


def test_direct_is_original_final_goal_even_if_terrain_blocks_line():
    scene = make_stage2_scenario("terrain3d", Stage2Config())
    guide = build(scene)
    assert scene.terrain.segment_clearance(scene.start_eta[:3], scene.goal, Stage2Config().body_radius_m) < 0
    np.testing.assert_array_equal(guide.current_target(scene.start_eta[:3]), scene.goal)
    assert guide.metrics["route_verified_safe"] is False
    assert guide.metrics["edge_checks"] == 0


def test_ridge_astar_has_safe_original_swept_edges_and_mild_grade(ridge_route):
    scene, guide = ridge_route
    cfg = Stage2Config()
    np.testing.assert_array_equal(guide.path[0], scene.start_eta[:3])
    np.testing.assert_array_equal(guide.path[-1], scene.goal)
    assert len(guide.path) > 2
    for a, b in zip(guide.path[:-1], guide.path[1:]):
        assert scene.terrain.segment_clearance(a, b, cfg.body_radius_m) >= cfg.safety_margin_m
        assert cfg.depth_limit_m-cfg.body_radius_m-max(a[2], b[2]) >= cfg.safety_margin_m
        assert abs(b[2]-a[2]) <= guide.config.max_grade*np.linalg.norm(b[:2]-a[:2])+1e-10
    assert guide.metrics["route_verified_safe"] is True
    assert guide.metrics["obstacle_future_information_used"] is False
    assert "not dynamic obstacles or REMUS reachability" in guide.metrics["validation_scope"]


def test_astar_repeated_scene_has_identical_route_without_random_search(ridge_route):
    scene, guide = ridge_route
    repeated = build(scene, guide.config)
    np.testing.assert_array_equal(guide.path, repeated.path)
    for key in ("expanded_nodes", "edge_checks", "minimum_route_clearance_m", "route_length_m"):
        assert guide.metrics[key] == repeated.metrics[key]


def test_original_segment_clearance_is_used_for_all_route_shortcuts(monkeypatch):
    scene = make_stage2_scenario("terrain3d", Stage2Config())
    calls = []
    original = scene.terrain.segment_clearance
    def observed(a, b, radius=0., spacing=.25):
        result = original(a, b, radius, spacing)
        calls.append((np.asarray(a).copy(), np.asarray(b).copy(), radius, result))
        return result
    monkeypatch.setattr(scene.terrain, "segment_clearance", observed)
    guide = build(scene, TerrainGuideConfig(mode="coarse3d_astar"))
    assert len(calls) >= guide.metrics["edge_checks"]
    for a, b in zip(guide.path[:-1], guide.path[1:]):
        assert any(np.array_equal(a, checked_a) and np.array_equal(b, checked_b)
                   and radius == Stage2Config().body_radius_m and clearance >= 1
                   for checked_a, checked_b, radius, clearance in calls)


def test_progress_uses_actual_position_and_does_not_mark_waypoint_arrival_success(ridge_route):
    scene, original = ridge_route
    guide = build(scene, original.config)
    initial = guide.current_target(scene.start_eta[:3])
    assert np.linalg.norm(initial-scene.start_eta[:3]) <= guide.config.lookahead_m+1e-9
    target = guide.current_target(guide.path[1])
    assert guide.progress_m > 0
    assert np.linalg.norm(guide.path[1]-scene.goal) > Stage2Config().goal_tolerance_m
    assert not hasattr(guide, "navigation_success")
    assert not hasattr(guide, "success")
    assert np.linalg.norm(target-scene.goal) > 0
    np.testing.assert_array_equal(guide.final_goal, scene.goal)
    previous_progress = guide.progress_m
    guide.current_target(scene.start_eta[:3])
    assert guide.progress_m >= previous_progress


def test_route_and_goal_are_not_mutated_by_target_consumer(ridge_route):
    scene, original = ridge_route
    guide = build(scene, original.config)
    target = guide.current_target(scene.start_eta[:3])
    target[:] = -999
    assert np.all(guide.path >= 0)
    with pytest.raises(ValueError):
        guide.path[0, 0] = -999
    with pytest.raises(ValueError):
        guide.final_goal[0] = -999


def test_deviated_unsafe_state_is_exposed_and_not_claimed_safe(ridge_route):
    scene, original = ridge_route
    guide = build(scene, original.config)
    target = guide.current_target([130, 100, 45])
    assert scene.terrain.clearance([130, 100, 45], Stage2Config().body_radius_m) < 0
    assert guide.unreachable_target_count == 1
    assert np.isfinite(target).all()


def test_astar_fixed_search_budget_exhaustion_is_explicit():
    scene = make_stage2_scenario("terrain3d", Stage2Config())
    with pytest.raises(TerrainRouteError, match="fixed 1-node budget"):
        build(scene, TerrainGuideConfig(mode="coarse3d_astar", max_nodes=1))


def test_impossible_full_width_seabed_barrier_never_returns_unchecked_route():
    terrain = HeightField(np.array([0., 20., 25., 35., 40., 60.]),
        np.array([0., 60.]), np.array([[90., 90., 0., 0., 90., 90.]]*2), 18.)
    with pytest.raises(TerrainRouteError, match="no safe route"):
        build_terrain_guidance(terrain, [10, 30, 30], [50, 30, 30],
            body_radius_m=Stage2Config().body_radius_m, safety_margin_m=1,
            depth_limit_m=100, config=TerrainGuideConfig(mode="coarse3d_astar",
                xy_step_m=10, depth_step_m=10, max_grade=.5, max_nodes=5000))


def test_excessive_grid_is_rejected_before_unbounded_allocation():
    scene = make_stage2_scenario("terrain3d", Stage2Config())
    with pytest.raises(TerrainRouteError, match="allocation guard"):
        build(scene, TerrainGuideConfig(mode="coarse3d_astar", xy_step_m=.001,
                                       depth_step_m=.001))


def test_depth_envelope_cannot_be_bypassed_by_deeper_terrain():
    terrain = HeightField.flat(260, 200, 200)
    with pytest.raises(TerrainRouteError, match="depth clearance"):
        build_terrain_guidance(terrain, [30, 100, 99], [210, 100, 99],
            body_radius_m=Stage2Config().body_radius_m, safety_margin_m=1,
            depth_limit_m=100, config=TerrainGuideConfig(mode="coarse3d_astar"))


def test_unsafe_surface_or_domain_endpoints_are_rejected():
    terrain = HeightField.flat(260, 200, 200)
    for start in ([30, 100, 1], [0.5, 100, 35], [30, 200, 35]):
        with pytest.raises(TerrainRouteError, match="start or goal"):
            build_terrain_guidance(terrain, start, [210, 100, 35],
                body_radius_m=Stage2Config().body_radius_m, safety_margin_m=1,
                depth_limit_m=100)


def test_perlin_scene_is_method_independent_and_actually_terrain_constrained():
    cfg = Stage2Config()
    scene = make_constrained_perlin_scene(cfg)
    repeated = make_constrained_perlin_scene(cfg)
    np.testing.assert_array_equal(scene.terrain.depths, repeated.terrain.depths)
    np.testing.assert_array_equal(scene.start_eta, repeated.start_eta)
    np.testing.assert_array_equal(scene.goal, repeated.goal)
    assert scene.metadata["method_independent_scene"] is True
    assert scene.metadata["scenario_seed"] == 7
    assert np.linalg.norm(scene.goal-scene.start_eta[:3]) == pytest.approx(180)
    assert scene.terrain.depths.min() == pytest.approx(50)
    assert scene.terrain.depths.max() == pytest.approx(200)
    assert scene.terrain.segment_clearance(scene.start_eta[:3], scene.goal, cfg.body_radius_m) < cfg.safety_margin_m
    assert len(scene.obstacles.snapshot().ids) == 0
    for point in (scene.start_eta[:3], scene.goal):
        assert scene.terrain.clearance(point, cfg.body_radius_m) > cfg.safety_margin_m
    guide = build(scene, TerrainGuideConfig(mode="coarse3d_astar"))
    for a, b in zip(guide.path[:-1], guide.path[1:]):
        assert scene.terrain.segment_clearance(a, b, cfg.body_radius_m) >= cfg.safety_margin_m


def test_perlin_factory_uses_original_terrain_generator():
    from auv3d.config import TerrainConfig
    cfg = replace(Stage2Config(), seed=11)
    scene = make_constrained_perlin_scene(cfg)
    original = HeightField.perlin(TerrainConfig(), 11)
    np.testing.assert_array_equal(scene.terrain.xs, original.xs)
    np.testing.assert_array_equal(scene.terrain.ys, original.ys)
    np.testing.assert_array_equal(scene.terrain.depths, original.depths)
    assert scene.terrain.slope_bound == original.slope_bound


@pytest.mark.parametrize("changes", [
    {"mode": "unknown"}, {"xy_step_m": 0}, {"depth_step_m": float("nan")},
    {"max_grade": .6}, {"waypoint_tolerance_m": -1}, {"lookahead_m": float("inf")},
    {"max_nodes": 0}, {"max_nodes": True},
])
def test_terrain_guidance_rejects_invalid_config(changes):
    with pytest.raises(ValueError):
        replace(TerrainGuideConfig(), **changes).validate()


def test_invalid_terrain_cannot_produce_false_safe_route():
    terrain = HeightField.flat(260, 200, 200.)
    terrain.depths[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_terrain_guidance(terrain, [30, 100, 35], [210, 100, 35],
            body_radius_m=.81, safety_margin_m=1, depth_limit_m=100)
