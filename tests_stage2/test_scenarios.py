from dataclasses import replace
import json

import numpy as np
import pytest

from auv3d.scenarios import STAGE2_SCENARIOS, ManeuveringObstacles, make_stage2_scenario
from auv3d.sensors import IdealObjectSonar, TrackMemory
from auv3d.stage2_config import Stage2Config


@pytest.mark.parametrize("name", STAGE2_SCENARIOS)
def test_scene_starts_and_goals_are_valid(name):
    cfg = Stage2Config()
    scene = make_stage2_scenario(name, cfg)
    assert scene.start_eta.shape == (6,)
    assert scene.goal.shape == (3,)
    assert np.isfinite(np.r_[scene.start_eta, scene.goal]).all()
    snap = scene.obstacles.snapshot()
    for point in (scene.start_eta[:3], scene.goal):
        assert scene.terrain.clearance(point, cfg.body_radius_m) > cfg.safety_margin_m
        assert np.all(np.linalg.norm(snap.positions - point, axis=1) > snap.radii + cfg.body_radius_m + cfg.safety_margin_m)
    assert scene.metadata["method_independent_scene"] is True


def test_dense_scene_contains_simultaneously_observable_real_interactions():
    cfg = Stage2Config()
    scene = make_stage2_scenario("multi_crossing3d", cfg)
    sensor = IdealObjectSonar(cfg.sensor_config(), cfg.seed)
    observations = sensor.observe(scene.start_eta, 0, scene.obstacles.snapshot(), scene.terrain)
    assert len(observations) == 3
    # A nominal straight transit would encounter all three objects within 30 m
    # in the same interval, rather than merely counting distant random objects.
    t = 44.0
    scene.obstacles.step(t)
    nominal_position = scene.start_eta[:3] + [1.84 * t, 0, 0]
    center_distances = np.linalg.norm(scene.obstacles.snapshot().positions - nominal_position, axis=1)
    assert np.count_nonzero(center_distances < cfg.interaction_distance_m) == 3
    assert center_distances.min() < cfg.obstacle_radius_m + cfg.body_radius_m


def test_crossing_scene_geometry_is_an_actual_conflict():
    cfg = Stage2Config()
    scene = make_stage2_scenario("crossing3d", cfg)
    t = 43.75
    scene.obstacles.step(t)
    nominal_position = scene.start_eta[:3] + [1.84 * t, 0, 0]
    assert np.linalg.norm(scene.obstacles.snapshot().positions[0] - nominal_position) < cfg.obstacle_radius_m + cfg.body_radius_m


def test_maneuvering_path_has_speed_and_heading_changes_without_future_observations():
    cfg = Stage2Config()
    scene = make_stage2_scenario("maneuvering3d", cfg)
    obj = scene.obstacles
    v0 = obj.velocities.copy()
    obj.step(20.0)
    v1 = obj.velocities.copy()
    assert np.any(abs(np.linalg.norm(v0, axis=1) - np.linalg.norm(v1, axis=1)) > 0.02)
    assert np.any(abs(np.arctan2(v0[:, 1], v0[:, 0]) - np.arctan2(v1[:, 1], v1[:, 0])) > 0.1)
    sensor = IdealObjectSonar(cfg.sensor_config(), cfg.seed)
    obs = sensor.observe(scene.start_eta, obj.t, obj.snapshot(), scene.terrain)
    assert all(o.time_s == 20.0 for o in obs)
    memory = TrackMemory(cfg.track_max_age_s)
    memory.update(obs, obj.t)
    assert all(max(track.times) <= obj.t for track in memory.current(obj.t))
    assert all(not hasattr(o, "future_positions") for o in obs)


def test_curved_obstacle_motion_is_time_aligned_and_step_partition_invariant():
    cfg = Stage2Config()
    a = make_stage2_scenario("maneuvering3d", cfg).obstacles
    b = make_stage2_scenario("maneuvering3d", cfg).obstacles
    a.step(10.0)
    for _ in range(500):
        b.step(cfg.physics_dt)
    assert b.t == pytest.approx(a.t, abs=1e-10)
    np.testing.assert_allclose(a.snapshot().positions, b.snapshot().positions, atol=1e-10)
    np.testing.assert_allclose(a.velocities, b.velocities, atol=1e-10)


def test_obstacle_snapshot_is_isolated_from_scene_state():
    scene = make_stage2_scenario("maneuvering3d", Stage2Config())
    snapshot = scene.obstacles.snapshot()
    expected = snapshot.positions.copy()
    snapshot.positions[:] = -999
    snapshot.radii[:] = 999
    np.testing.assert_array_equal(scene.obstacles.snapshot().positions, expected)
    assert np.all(scene.obstacles.snapshot().radii == Stage2Config().obstacle_radius_m)


def test_speed_and_turn_parameters_change_truth_without_changing_task():
    cfg = Stage2Config()
    a = make_stage2_scenario("maneuvering3d", cfg)
    b = make_stage2_scenario("maneuvering3d", replace(cfg, obstacle_speed_scale=1.5, obstacle_turn_rate_deg_s=7))
    np.testing.assert_array_equal(a.start_eta, b.start_eta)
    np.testing.assert_array_equal(a.goal, b.goal)
    np.testing.assert_array_equal(a.obstacles.snapshot().positions, b.obstacles.snapshot().positions)
    a.obstacles.step(10)
    b.obstacles.step(10)
    assert np.linalg.norm(a.obstacles.positions - b.obstacles.positions) > 1


def test_zero_speed_scale_leaves_maneuvering_obstacles_static():
    scene = make_stage2_scenario("maneuvering3d", replace(Stage2Config(), obstacle_speed_scale=0))
    original = scene.obstacles.snapshot().positions
    scene.obstacles.step(100)
    np.testing.assert_array_equal(scene.obstacles.snapshot().positions, original)


def test_terrain_ridge_requires_vertical_or_lateral_avoidance_but_has_a_passage():
    cfg = Stage2Config()
    scene = make_stage2_scenario("terrain3d", cfg)
    route = np.column_stack([np.linspace(30, 210, 181), np.full(181, 100), np.full(181, 35)])
    assert scene.terrain.clearance(route, cfg.body_radius_m).min() < 0
    upper_passage = route.copy()
    upper_passage[:, 2] = 10
    assert scene.terrain.clearance(upper_passage, cfg.body_radius_m).min() > cfg.safety_margin_m
    assert scene.terrain.depths.min() > 0
    assert scene.terrain.slope_bound > 0


def test_sensor_range_changes_observations_without_changing_ground_truth():
    cfg = Stage2Config()
    scene = make_stage2_scenario("multi_crossing3d", cfg)
    far_sensor = IdealObjectSonar(cfg.sensor_config(), cfg.seed)
    near_sensor = IdealObjectSonar(replace(cfg, sensor_range_m=40).sensor_config(), cfg.seed)
    snapshot = scene.obstacles.snapshot()
    assert len(far_sensor.observe(scene.start_eta, 0, snapshot, scene.terrain)) == 3
    assert len(near_sensor.observe(scene.start_eta, 0, snapshot, scene.terrain)) == 0
    np.testing.assert_array_equal(scene.obstacles.snapshot().positions, snapshot.positions)


def test_diagnostic_scene_is_reachable_and_explicitly_labelled():
    scene = make_stage2_scenario("tracking_mismatch3d", Stage2Config())
    assert scene.metadata["diagnostic_only"] is True
    assert scene.metadata["reference_step_test"]["heading_offset_deg"] == 70
    assert len(scene.obstacles.snapshot().ids) == 0
    assert scene.terrain.segment_clearance(scene.start_eta[:3], scene.goal, Stage2Config().body_radius_m) > 1


def test_config_roundtrip_and_sensor_mapping(tmp_path):
    cfg = replace(Stage2Config(), observation_delay_s=1.2, current_speed=0.3, current_direction_deg=270)
    path = tmp_path / "stage2.json"
    path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    loaded = Stage2Config.load(path)
    assert loaded == cfg
    assert isinstance(loaded.rpm_values, tuple)
    assert loaded.sensor_config().range_m == cfg.sensor_range_m
    assert loaded.sensor_config().period_s == cfg.observation_period_s
    assert loaded.planning_dt == loaded.control_dt
    assert loaded.shortlist_count == loaded.candidate_budget


@pytest.mark.parametrize("changes", [
    {"physics_dt": 0.03}, {"control_dt": 0.031}, {"horizon_s": 1.0},
    {"depth_limit_m": 101}, {"nominal_rpm": 1526}, {"rpm_values": (-1, 1100)},
    {"candidate_budget": 0}, {"candidate_budget": True},
    {"observation_delay_s": -1}, {"observation_delay_s": 0.011},
    {"current_speed": float("nan")}, {"current_direction_deg": float("inf")},
    {"sensor_vfov_deg": 181}, {"heading_offsets_deg": ()},
    {"depth_offsets_m": (0, 0)}, {"terrain_amplitude_m": 90},
    {"paper_obstacle_count": -1}, {"scenario_width_m": 100}, {"seed": -1},
    {"safety_margin_m": 50},
])
def test_config_rejects_invalid_real_parameters(changes):
    with pytest.raises(ValueError):
        replace(Stage2Config(), **changes).validate()


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_maneuvering_obstacles_reject_invalid_time_step(dt):
    obj = ManeuveringObstacles([[1, 2, 3]], [[0, 1, 0]], [1])
    with pytest.raises(ValueError):
        obj.step(dt)


def test_unknown_scene_is_rejected():
    with pytest.raises(ValueError, match="unknown stage 2 scenario"):
        make_stage2_scenario("invented_scene", Stage2Config())
