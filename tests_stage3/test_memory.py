"""Functional/cause-and-effect tests for causal memory policy ablations."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from auv3d.sensors import Observation, TrackMemory
from auv3d.stage2_config import Stage2Config
from auv3d.stage3_memory import CausalTrackMemory, make_track_memory


def observation(stamp=0.0, position=(0.0, 0.0, 20.0), object_id=1, radius=4.0):
    return Observation(object_id, stamp, np.array(position, dtype=float), radius)


@pytest.mark.parametrize("max_age", [4.0, 8.0, 12.0])
def test_hard_expiry_matches_original_age_boundary_and_tracks(max_age):
    legacy, new = TrackMemory(max_age), CausalTrackMemory(max_age)
    for now, observations in [(0, [observation()]), (1, [observation(1, object_id=2)]),
                              (max_age, []), (max_age + 1e-10, []),
                              (max_age + 0.02, []), (max_age + 2, [])]:
        legacy.update(observations, now)
        new.update(observations, now)
        old_tracks, new_tracks = legacy.current(now), new.current(now)
        assert [t.object_id for t in old_tracks] == [t.object_id for t in new_tracks]
        for old, current in zip(old_tracks, new_tracks):
            assert old.radius_m == current.radius_m
            assert old.times == current.times
            np.testing.assert_array_equal(old.positions, current.positions)
            assert new.uncertainty_for(current, now) == 0


def test_retention_preserves_only_measured_history_and_expands_uncertainty():
    memory = CausalTrackMemory(policy="retain_uncertain", uncertainty_growth_mps=2.5)
    memory.update([observation(0), observation(1, (1, 0, 20))], 1)
    before = memory.current(1)[0]
    memory.update([], 12)
    track = memory.current(12)[0]
    assert track.times == before.times == [0, 1]
    np.testing.assert_array_equal(track.positions, before.positions)
    assert memory.uncertainty_for(track, 12) == 25.0
    diagnostic = memory.diagnostics(12)
    assert diagnostic["retained_beyond_expiry_ids"] == [1]
    assert diagnostic["uncertainty_by_id"] == {1: 25.0}
    assert diagnostic["lost_object_count"] == 1
    assert diagnostic["lost"] == 1
    assert diagnostic["uncertainty_status"] == "configured_heuristic_not_calibrated_coverage"


def test_declared_delay_does_not_count_fresh_delayed_observation_as_lost():
    memory = CausalTrackMemory(policy="retain_uncertain", observation_delay_s=2)
    memory.update([observation(0)], 2)
    assert memory.diagnostics(3)["lost_ids"] == []
    assert memory.uncertainty_for(memory.current(3)[0], 3) == 0
    memory.update([], 3.02)
    assert memory.diagnostics(3.02)["lost_ids"] == [1]
    assert memory.diagnostics(3.02)["maximum_uncertainty_m"] == pytest.approx(0.05)


def test_loss_and_reacquisition_count_transitions_not_physics_steps():
    memory = CausalTrackMemory(policy="retain_uncertain")
    memory.update([observation()], 0)
    memory.update([], 2)
    memory.update([], 3)
    memory.update([observation(4)], 4)
    memory.update([], 6)
    diagnostic = memory.diagnostics(6)
    assert (diagnostic["added"], diagnostic["lost"], diagnostic["reacquired"]) == (1, 2, 1)
    assert diagnostic["lost_object_count"] == 1
    assert [event["event"] for event in diagnostic["events"]] == ["added", "lost", "reacquired", "lost"]


def test_reacquisition_after_expiry_is_logged_without_replaying_old_history():
    memory = CausalTrackMemory()
    memory.update([observation()], 0)
    memory.update([], 5)
    memory.update([observation(6, (5, 0, 20))], 6)
    diagnostic = memory.diagnostics(6)
    assert diagnostic["hard_removed"] == 1
    assert diagnostic["reacquired"] == 1
    assert memory.current(6)[0].times == [6]


def test_returned_history_events_and_input_arrays_do_not_mutate_memory():
    memory = CausalTrackMemory(policy="retain_uncertain")
    supplied = observation()
    memory.update([supplied], 0)
    supplied.position[:] = 99
    track = memory.current(0)[0]
    track.positions[0][:] = -99
    track.radius_m = 200
    diagnostic = memory.diagnostics(0)
    diagnostic["events"][0]["event"] = "corrupted"
    np.testing.assert_array_equal(memory.current(0)[0].positions[0], [0, 0, 20])
    assert memory.current(0)[0].radius_m == 4
    assert memory.diagnostics(0)["events"][0]["event"] == "added"


def test_original_32_measurement_limit_retained():
    memory = CausalTrackMemory(policy="retain_uncertain")
    memory.update([observation(float(t)) for t in range(40)], 39)
    assert memory.current(39)[0].times == list(range(8, 40))
    assert memory.diagnostics(39)["delivered_observations"] == 40


@pytest.mark.parametrize("policy", ["retain_uncertain", "conditional"])
def test_no_delete_of_nearby_lost_objects(policy):
    memory = CausalTrackMemory(policy=policy)
    memory.update([observation(position=(35, 0, 20))], 0, ego_position=[0, 0, 20])
    memory.update([], 20, ego_position=[0, 0, 20])
    assert len(memory.current(20)) == 1
    assert memory.diagnostics(20)["conditional_removed"] == 0


def test_conditional_deletion_requires_actual_ego_and_lost_track():
    memory = CausalTrackMemory(policy="conditional")
    memory.update([observation(position=(1000, 0, 20))], 0, ego_position=[0, 0, 20])
    assert len(memory.current(0)) == 1  # A current observation is not deleted.
    memory.update([], 2)  # No actual pose, no conditional certificate.
    assert len(memory.current(2)) == 1
    memory.update([], 3, ego_position=[0, 0, 20])
    diagnostic = memory.diagnostics(3)
    assert memory.current(3) == []
    assert diagnostic["conditional_removed"] == 1
    event = diagnostic["events"][-1]
    assert event["reason"] == "reachable_balls_disjoint"
    assert event["bound_status"] == "declared_assumption_not_certified"
    assert event["obstacle_reachable_radius_m"] == 4 * (3 + 26)
    assert event["ego_reachable_radius_m"] == 4 * 26
    assert event["excluded_interaction_gap_m"] > 0


def test_conditional_boundary_includes_age_both_speed_bounds_and_control_buffer():
    memory = CausalTrackMemory(policy="conditional", command_delay_s=1)
    radius, age = 4, 3
    threshold = 4 * (age + 27) + 4 * 27 + 30 + memory.body_radius_m + radius + 1
    memory.update([observation(position=(threshold, 0, 20), radius=radius)], 0)
    memory.update([], age, ego_position=[0, 0, 20])
    assert len(memory.current(age)) == 1  # Equality does not exclude interaction.
    far = CausalTrackMemory(policy="conditional", command_delay_s=1)
    far.update([observation(position=(threshold + 0.1, 0, 20), radius=radius)], 0)
    far.update([], age, ego_position=[0, 0, 20])
    assert far.current(age) == []


@pytest.mark.parametrize("bad", [
    observation(2), observation(radius=-1), observation(radius=float("nan")),
    observation(position=(float("inf"), 0, 20)), observation(object_id=True),
    observation(object_id=1.5), observation(stamp=-1),
    Observation(1, 0.5, np.array([0, 1]), 4),
])
def test_invalid_batch_rejected_before_partial_write(bad):
    memory = CausalTrackMemory()
    memory.update([observation()], 0)
    before = deepcopy(memory.diagnostics(0))
    with pytest.raises(ValueError):
        memory.update([observation(0.5, object_id=2), bad], 1)
    assert memory.diagnostics(0) == before
    assert [t.object_id for t in memory.current(0)] == [1]


@pytest.mark.parametrize("stamp", [0, -0.1])
def test_old_duplicate_packet_cannot_reacquire_removed_id(stamp):
    memory = CausalTrackMemory()
    memory.update([observation()], 0)
    memory.update([], 5)
    with pytest.raises(ValueError):
        memory.update([observation(stamp)], 5)
    assert memory.current(5) == []


@pytest.mark.parametrize("method", ["update", "current", "diagnostics"])
def test_memory_clock_cannot_go_backwards(method):
    memory = CausalTrackMemory()
    memory.update([], 2)
    with pytest.raises(ValueError, match="backwards"):
        getattr(memory, method)([], 1) if method == "update" else getattr(memory, method)(1)


@pytest.mark.parametrize("kwargs", [dict(policy="unknown"), dict(max_age_s=0),
    dict(uncertainty_growth_mps=-1), dict(obstacle_speed_bound_mps=0),
    dict(ego_speed_bound_mps=float("inf")), dict(horizon_s=False),
    dict(safety_margin_m=-1), dict(expected_period_s=float("nan"))])
def test_invalid_memory_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        CausalTrackMemory(**kwargs)


def test_factory_defaults_preserve_stage2_and_accept_explicit_policy_fields():
    cfg = Stage2Config()
    default = make_track_memory(cfg)
    assert default.policy == "hard_expiry"
    assert default.max_age_s == 4
    extended = SimpleNamespace(**cfg.to_dict())
    extended.memory_policy = "conditional"
    extended.memory_uncertainty_growth_mps = 3
    extended.memory_obstacle_speed_bound_mps = 5
    extended.memory_ego_speed_bound_mps = 6
    memory = make_track_memory(extended)
    assert memory.policy == "conditional"
    assert memory.uncertainty_growth_mps == 3
    assert memory.obstacle_speed_bound_mps == 5
    assert memory.ego_speed_bound_mps == 6


def test_reject_bad_ego_before_applying_observations():
    memory = CausalTrackMemory(policy="conditional")
    with pytest.raises(ValueError, match="ego_position"):
        memory.update([observation()], 0, ego_position=[0, 0, float("nan")])
    assert memory.current(0) == []


def test_gaussian_observation_noise_disables_conditional_hard_exclusion():
    cfg = SimpleNamespace(**Stage2Config().to_dict())
    cfg.memory_policy = "conditional"
    cfg.sensor_noise_std_m = 0.1
    memory = make_track_memory(cfg)
    memory.update([observation(position=(1000, 0, 20))], 0)
    memory.update([], 3, ego_position=[0, 0, 20])
    assert len(memory.current(3)) == 1
    diagnostic = memory.diagnostics(3)
    assert diagnostic["conditional_removed"] == 0
    assert diagnostic["conditional_measurements_bounded"] is False


def test_malformed_delivered_object_rejected_without_partial_update():
    memory = CausalTrackMemory()
    with pytest.raises(ValueError, match="Observation"):
        memory.update([observation(), {"object_id": 2}], 0)
    assert memory.current(0) == []
