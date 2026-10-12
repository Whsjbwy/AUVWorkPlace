"""Offline diagnostic truth cannot alter live prediction or archived execution."""
from dataclasses import replace
import json

import numpy as np
import pytest

from auv3d.adapters.remus_guidance import clone_plant, predict_reference
from auv3d.remus_planner import RemusDWA
from auv3d.scenarios import make_stage2_scenario
from auv3d.sensors import Track
from auv3d.stage2_config import Stage2Config
from auv3d.stage2_simulation import run_remus_episode, save_remus_episode
from auv3d.stage4_diagnostics import OfflineTruthReplay, replay_anchors
from auv3d.stage3_forecasting import CAObstacleForecaster
from auv3d.stage4_risk import RiskPolicy, ecp_scene_scores


@pytest.fixture
def small_cfg():
    return replace(Stage2Config(), horizon_s=.4, control_dt=.2,
                   observation_period_s=.2, prediction_output_dt=.2,
                   max_time_s=.8, warmup_s=.2, sensor_noise_std_m=.02)


@pytest.fixture
def archived(tmp_path, small_cfg):
    result = run_remus_episode(small_cfg, "crossing3d")
    target = tmp_path/"episode"
    save_remus_episode(result, target, animate=False, plot=False)
    return target, result


def test_replay_restores_real_controller_and_reproduces_b0_same_state(archived, small_cfg):
    directory, result = archived
    frames = list(replay_anchors(directory, small_cfg, "crossing3d", (.2, .4, .6)))
    assert [frame.now for frame in frames] == [.2, .4, .6]
    before = [frame.plant.eta.copy() for frame in frames]
    for frame in frames:
        planner = RemusDWA(small_cfg, risk_policy=RiskPolicy("B0"))
        planner.previous = frame.previous_action
        planner.set_memory_uncertainty(frame.memory_uncertainty)
        shadow = planner.plan(clone_plant(frame.plant), frame.goal, frame.terrain,
                              frame.tracks, frame.now)
        np.testing.assert_allclose(shadow.action.as_array(), frame.original_action.as_array(),
                                   atol=1e-10, rtol=0)
        assert shadow.feasible == frame.original_feasible
        assert shadow.minimum_predicted_clearance_m == pytest.approx(frame.original_predicted_clearance_m, abs=1e-9)
        exact = result["states"][np.isclose(result["states"][:, 0], frame.now)][0]
        np.testing.assert_allclose(np.r_[frame.plant.time_s, frame.plant.eta,
                                        frame.plant.nu, frame.plant.actual], exact,
                                   atol=1e-7, rtol=0)
    frames[0].plant.vehicle.z_int = 12345.
    frames[0].plant.eta[:] = -999.
    assert frames[1].plant.vehicle.z_int != 12345.
    np.testing.assert_array_equal(frames[1].plant.eta, before[1])


def test_replay_checks_every_state_even_after_last_requested_anchor(archived):
    directory, _ = archived
    path = directory/"states.csv"
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    data[-1, 1] += .01
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    with pytest.raises(ValueError, match="actual state"):
        list(replay_anchors(directory, anchors=(.2,)))


def test_replay_checks_controller_requests_and_archived_parameter_match(archived, small_cfg):
    directory, _ = archived
    with pytest.raises(ValueError, match="actual parameters"):
        list(replay_anchors(directory, replace(small_cfg, nominal_rpm=1200.), anchors=(.2,)))
    path = directory/"controls.csv"
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    data[0, 4] += .01
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    with pytest.raises(ValueError, match="actual control"):
        list(replay_anchors(directory, anchors=(.2,)))


def test_replay_rejects_undelivered_or_future_track_samples(archived):
    directory, _ = archived
    path = directory/"planning_diagnostics.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    record = next(row for row in records if abs(row["time_s"]-.2) < 1e-8)
    record["tracks"][0]["positions"][-1][0] += 5.
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="causally delivered"):
        list(replay_anchors(directory, anchors=(.2,)))


def test_missing_selected_plan_is_corruption_and_not_an_empty_anchor(archived):
    directory, _ = archived
    path = directory/"selected_plans.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    records = [row for row in records if abs(row["time_s"]-.2) > 1e-8]
    path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree at requested anchor"):
        list(replay_anchors(directory, anchors=(.2,)))


@pytest.mark.parametrize("anchors", [(0.21,), (.4, .2), (-.2,), (float("nan"),)])
def test_replay_rejects_invalid_anchor_clocks(archived, anchors):
    with pytest.raises(ValueError, match="anchor"):
        list(replay_anchors(archived[0], anchors=anchors))


def test_empty_observed_set_does_not_invent_a_same_state_comparison(tmp_path, small_cfg):
    result = run_remus_episode(small_cfg, "empty3d")
    target = tmp_path/"empty"
    save_remus_episode(result, target, animate=False, plot=False)
    assert list(replay_anchors(target, anchors=(.2, .4, .6))) == []


def _synthetic_candidate(cfg, points):
    """Geometry fixture only, not a REMUS execution or navigation success."""
    rows = np.zeros((round(cfg.horizon_s/cfg.physics_dt)+1, 16))
    rows[:, 0] = np.arange(len(rows))*cfg.physics_dt
    rows[:, 1:4] = points
    return rows


def test_selected_forecast_false_feasible_and_score_have_explicit_offline_scope(small_cfg):
    times = np.arange(round(small_cfg.horizon_s/small_cfg.physics_dt)+1)*small_cfg.physics_dt
    points = np.column_stack([np.full(len(times), 110.), 135.-.8*times, np.full(len(times), 35.)])
    candidate = _synthetic_candidate(small_cfg, points)
    track = Track(0, 4., [0.], [np.array([130., 135., 35.])])
    evaluator = OfflineTruthReplay(small_cfg, "crossing3d")
    before = candidate.copy(), track.positions[0].copy()
    metrics = evaluator.evaluate(candidate, [track], 0., quantile_m=1., feasible=True,
                                 horizons=(.2, .4))
    assert metrics["selected_forecast_false_feasible"]
    assert metrics["selected_truth_collision"]
    assert metrics["selected_future_truth_min_clearance_m"] < 0
    assert metrics["selected_ecp_gap_score_m"] > 19.
    assert metrics["ball_error_max_m"] >= 20.
    assert metrics["coverage_ecp"] is False and metrics["coverage_ball"] is False
    assert "not all-candidate" in metrics["scope"]
    assert not metrics["truth_used_online"]
    for value in metrics["prediction_errors_by_horizon"].values():
        assert value["sample_count"] == 1
        assert value["squared_error_sum_m2"] == pytest.approx(value["rmse_m"]**2)
    np.testing.assert_array_equal(candidate, before[0])
    np.testing.assert_array_equal(track.positions[0], before[1])


def test_unobserved_collision_is_counted_without_inventing_error_labels(small_cfg):
    times = np.arange(round(small_cfg.horizon_s/small_cfg.physics_dt)+1)*small_cfg.physics_dt
    points = np.column_stack([np.full(len(times), 110.), 135.-.8*times, np.full(len(times), 35.)])
    candidate = _synthetic_candidate(small_cfg, points)
    metrics = OfflineTruthReplay(small_cfg, "crossing3d").evaluate(
        candidate, [], 0., quantile_m=100., feasible=True)
    assert metrics["selected_forecast_false_feasible"]
    assert metrics["observed_target_count"] == 0 and metrics["unobserved_true_target_count"] == 1
    assert metrics["selected_ecp_gap_score_m"] is None
    assert metrics["ball_error_max_m"] is None
    assert metrics["coverage_ecp"] is None and metrics["coverage_ball"] is None
    assert all(value["sample_count"] == 0 and value["rmse_m"] is None
               for value in metrics["prediction_errors_by_horizon"].values())


def test_offline_coverage_uses_the_same_nearest_object_ecp_score_as_risk_library(small_cfg):
    times = np.arange(round(small_cfg.horizon_s/small_cfg.physics_dt)+1)*small_cfg.physics_dt
    path = np.tile([115., 100., 35.], (len(times), 1))
    candidate = _synthetic_candidate(small_cfg, path)
    scene = make_stage2_scenario("multi_crossing3d", small_cfg)
    snapshot = scene.obstacles.snapshot()
    biases = np.array([[40., 0., 0.], [-30., 0., 0.], [0., 40., 0.]])
    tracks = [Track(int(object_id), float(radius), [0.], [position+bias])
              for object_id, position, radius, bias in
              zip(snapshot.ids, snapshot.positions, snapshot.radii, biases, strict=True)]
    predictions = np.asarray([CAObstacleForecaster(small_cfg).forecast(track, times, 0.).positions
                              for track in tracks])
    truth = []
    for _ in times:
        truth.append(scene.obstacles.snapshot().positions)
        scene.obstacles.step(small_cfg.physics_dt)
    labels = np.asarray(truth).transpose(1, 0, 2)
    library_score = ecp_scene_scores(
        path, predictions[None, None], labels[None, None],
        snapshot.radii[None, None], np.ones((1, 1, len(tracks)), bool), small_cfg.body_radius_m)
    metrics = OfflineTruthReplay(small_cfg, "multi_crossing3d").evaluate(candidate, tracks, 0.)
    assert metrics["selected_ecp_gap_score_m"] == pytest.approx(library_score[0], abs=1e-10)


def test_offline_replay_matches_saved_snapshots_and_nonzero_anchor(archived, small_cfg):
    directory, result = archived
    frame = next(replay_anchors(directory, anchors=(.2,)))
    states = predict_reference(frame.plant, frame.original_action, small_cfg.horizon_s,
                               small_cfg.physics_dt, record_controls=False).states
    evaluator = OfflineTruthReplay(small_cfg, "crossing3d", result["obstacles"])
    first = evaluator.evaluate(states, frame.tracks, frame.now, feasible=frame.original_feasible,
                               horizons=(.2, .4))
    second = evaluator.evaluate(states, frame.tracks, frame.now, feasible=frame.original_feasible,
                                horizons=(.2, .4))
    assert first == second
    assert first["anchor_s"] == .2 and first["checked_saved_snapshots"] > 0
    assert first["maximum_saved_position_error_m"] == 0.
    assert first["prediction_errors_by_horizon"]["0.4"]["sample_count"] == len(frame.tracks)


def test_offline_truth_replay_rejects_changed_saved_motion(small_cfg):
    scene = make_stage2_scenario("crossing3d", small_cfg)
    snapshot = scene.obstacles.snapshot()
    saved = [{"time_s": 0., "ids": snapshot.ids.tolist(),
              "positions": (snapshot.positions+1.).tolist(), "radii": snapshot.radii.tolist()}]
    candidate = _synthetic_candidate(small_cfg, np.tile([30., 100., 35.], (21, 1)))
    with pytest.raises(ValueError, match="positions disagree"):
        OfflineTruthReplay(small_cfg, "crossing3d", saved).evaluate(candidate, [], 0.)


@pytest.mark.parametrize("change", ["sparse", "absolute", "nan", "future_track", "negative_q"])
def test_offline_evaluator_rejects_ambiguous_time_or_labels(small_cfg, change):
    candidate = _synthetic_candidate(small_cfg, np.tile([30., 100., 35.], (21, 1)))
    tracks, q = [], None
    if change == "sparse":
        candidate = candidate[::2]
    elif change == "absolute":
        candidate[:, 0] += 10.
    elif change == "nan":
        candidate[3, 1] = np.nan
    elif change == "future_track":
        tracks = [Track(0, 4., [1.], [np.array([110., 135., 35.])])]
    elif change == "negative_q":
        q = -1.
    with pytest.raises(ValueError):
        OfflineTruthReplay(small_cfg, "crossing3d").evaluate(candidate, tracks, 0., quantile_m=q)
