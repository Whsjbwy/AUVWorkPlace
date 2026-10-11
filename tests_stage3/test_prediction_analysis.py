"""Offline error comparisons cannot change histories, anchors or truth."""
from dataclasses import replace
import json

import numpy as np
import pytest

from auv3d.stage2_config import Stage2Config
from auv3d.stage2_simulation import run_remus_episode, save_remus_episode
from auv3d.stage3_forecasting import CAObstacleForecaster, ObstacleForecast
from scripts_stage3.analyze_prediction_errors import (
    analyze_sources, error_statistics, paired_case,
)


def tiny_config():
    return replace(Stage2Config(), horizon_s=.4, control_dt=.2, max_time_s=.4,
                   observation_period_s=.4, prediction_output_dt=.2,
                   warmup_s=0, candidate_budget=1)


class FakeTrained(CAObstacleForecaster):
    """Deterministic contract fixture, not an experimental learned model."""
    name = "fake_trained_test_fixture"

    def forecast(self, track, absolute_times, now, extra_uncertainty_m=0):
        base = super().forecast(track, absolute_times, now, extra_uncertainty_m)
        trained = np.asarray(absolute_times)-track.times[-1] <= 9+1e-8
        positions = base.positions + np.array([3.0, 0.0, 0.0])
        return ObstacleForecast(track.object_id, base.absolute_times, positions,
            base.enclosing_radii_m, dict(base.metadata, predictor=self.name,
                trained_horizon_s=9.0, trained_valid_mask=trained.tolist(),
                extension_mode="cv_after9", p25_calibration_radii_m=[None]*len(trained)))


def providers():
    cfg = replace(tiny_config(), prediction_history_dt_s=3.0)
    return {"ca_dt3": CAObstacleForecaster(cfg), "fake_trained": FakeTrained(cfg)}


def saved_actual(tmp_path, scenario="static3d"):
    source = tmp_path/"actual"
    save_remus_episode(run_remus_episode(tiny_config(), scenario), source, plot=False)
    return source


def test_saved_real_remus_histories_have_paired_anchors_and_verified_truth(tmp_path):
    source = saved_actual(tmp_path)
    rows, anchors, summary = paired_case(source, forecasters=providers())
    assert summary["planning_anchor_count"] == summary["object_anchor_count"] == 2
    assert summary["paired_row_count"] == 24  # 2 anchors * 6 horizons * 2 predictors.
    assert summary["verified_recorded_truth_snapshots"] == 2
    assert summary["maximum_recorded_truth_position_difference_m"] == 0
    assert summary["new_vehicle_rollouts"] == 0
    assert not summary["feedback_to_planner"] and not summary["parameter_fitting"]
    assert all(row["position_error_m"] == 0 for row in rows if row["variant"] == "ca_dt3")
    assert all(row["position_error_m"] == 3 for row in rows if row["variant"] == "fake_trained")
    for anchor in anchors:
        variants = anchor["forecasts"]
        assert variants["ca_dt3"]["absolute_times_s"] == variants["fake_trained"]["absolute_times_s"]
        assert anchor["measured_times_s"] == [0.0]
        assert len(anchor["measured_positions"]) == 1


def test_nine_second_range_uses_last_measurement_not_planning_time(tmp_path):
    rows, _, _ = paired_case(saved_actual(tmp_path), forecasters=providers())
    selected = [row for row in rows if row["variant"] == "fake_trained" and row["planning_horizon_s"] == 9]
    assert selected[0]["planning_time_s"] == 0
    assert selected[0]["within_trained_horizon"] is True
    assert selected[0]["from_measurement_horizon_s"] == 9
    assert selected[1]["planning_time_s"] == .2
    assert selected[1]["within_trained_horizon"] is False
    assert selected[1]["from_measurement_horizon_s"] == 9.2
    assert selected[1]["training_scope"] == "explicit_post_training_extension"
    statistics = error_statistics(rows)
    at9 = [row for row in statistics if row["variant"] == "fake_trained" and row["planning_horizon_s"] == 9]
    assert len(at9) == 2  # Do not merge trained support and extended support.
    assert {row["training_scope"] for row in at9} == {
        "within_trained_range_from_measurement", "explicit_post_training_extension"}
    ca9 = [row for row in statistics if row["variant"] == "ca_dt3" and row["planning_horizon_s"] == 9]
    assert len(ca9) == 2  # CA is split into the exact same fresh/stale anchor subsets.
    for learned_row in at9:
        same_scope = [row for row in ca9 if row["common_pplnm_support_scope"] == learned_row["common_pplnm_support_scope"]]
        assert len(same_scope) == 1 and same_scope[0]["sample_count"] == learned_row["sample_count"] == 1


def test_analysis_preserves_source_and_writes_raw_paired_histories(tmp_path):
    source = saved_actual(tmp_path)
    original = (source/"planning_diagnostics.json").read_text(encoding="utf-8")
    out = tmp_path/"analysis"
    result = analyze_sources([source], out, forecasters=providers(), plot=False)
    assert (source/"planning_diagnostics.json").read_text(encoding="utf-8") == original
    assert result["paired_row_count"] == 24
    assert not result["ground_truth_used_for_prediction"]
    assert not result["risk_margins_adjusted"] and not result["calibration_refitted"]
    assert result["common_support_anchor_sets_equal"]
    for name in ("paired_prediction_errors.csv", "prediction_RMSE_by_horizon.csv", "prediction_anchors.json", "summary.json"):
        assert (out/name).is_file()
    with pytest.raises(FileExistsError):
        analyze_sources([source], out, forecasters=providers(), plot=False)


@pytest.mark.parametrize("corruption", ["not_delivered", "pending", "position", "future_track", "early_delivery"])
def test_undelivered_changed_or_future_history_cannot_support_prediction_comparison(tmp_path, corruption):
    source = saved_actual(tmp_path)
    observation_path, detail_path = source/"observations.json", source/"planning_diagnostics.json"
    observations = json.loads(observation_path.read_text(encoding="utf-8"))
    details = json.loads(detail_path.read_text(encoding="utf-8"))
    if corruption == "not_delivered":
        observations[0]["delivered_time_s"] = .2
    elif corruption == "pending":
        observations[0]["delivered_time_s"] = None
    elif corruption == "position":
        details[0]["tracks"][0]["positions"][0][0] += .1
    elif corruption == "future_track":
        details[0]["tracks"][0]["times_s"][0] = .02
    elif corruption == "early_delivery":
        observations[0]["measurement_time_s"] = .02
        observations[0]["delivered_time_s"] = 0
    observation_path.write_text(json.dumps(observations), encoding="utf-8")
    detail_path.write_text(json.dumps(details), encoding="utf-8")
    out = tmp_path/"analysis"
    with pytest.raises(ValueError):
        analyze_sources([source], out, forecasters=providers(), plot=False)
    assert not out.exists()


def test_truth_regeneration_must_match_recorded_actual_snapshots(tmp_path):
    source = saved_actual(tmp_path)
    path = source/"obstacles.json"
    truth = json.loads(path.read_text(encoding="utf-8"))
    truth[0]["positions"][0][0] += .1
    path.write_text(json.dumps(truth), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        analyze_sources([source], tmp_path/"out", forecasters=providers(), plot=False)


def test_wrong_trained_validity_mask_is_rejected(tmp_path):
    class WrongMask(FakeTrained):
        def forecast(self, *args, **kwargs):
            result = super().forecast(*args, **kwargs)
            return replace(result, metadata=dict(result.metadata,
                trained_valid_mask=[True]*len(result.absolute_times)))
    with pytest.raises(ValueError, match="validity mask"):
        paired_case(saved_actual(tmp_path), forecasters={"wrong": WrongMask(tiny_config())})


def test_empty_obstacle_run_is_an_honest_zero_sample_comparison(tmp_path):
    rows, anchors, summary = paired_case(saved_actual(tmp_path, "empty3d"), forecasters=providers())
    assert rows == anchors == []
    assert summary["planning_anchor_count"] == 2
    assert summary["paired_row_count"] == 0
    assert error_statistics(rows) == []


@pytest.mark.parametrize("horizons", [[9, 3], [0, 9], [3, float("nan")], [3.001, 9]])
def test_horizon_times_must_be_ordered_finite_and_on_original_physics_grid(tmp_path, horizons):
    with pytest.raises(ValueError):
        paired_case(saved_actual(tmp_path), forecasters=providers(), horizons=horizons)


def test_predictor_cannot_supply_another_objects_forecast(tmp_path):
    class WrongID(FakeTrained):
        def forecast(self, *args, **kwargs):
            return replace(super().forecast(*args, **kwargs), object_id=9999)
    with pytest.raises(ValueError, match="identity"):
        paired_case(saved_actual(tmp_path), forecasters={"wrong": WrongID(tiny_config())})


def test_every_common_support_statistic_has_identical_ca_and_learned_anchor_counts(tmp_path):
    rows, _, _ = paired_case(saved_actual(tmp_path), forecasters=providers())
    for row in rows:
        assert row["model_scope"] == row["training_scope"]
    statistics = error_statistics(rows)
    domains = {(row["planning_horizon_s"], row["common_pplnm_support_scope"]) for row in statistics}
    for horizon, scope in domains:
        selected = [row for row in statistics if row["planning_horizon_s"] == horizon
                    and row["common_pplnm_support_scope"] == scope]
        assert {row["variant"] for row in selected} == {"ca_dt3", "fake_trained"}
        assert len({row["sample_count"] for row in selected}) == 1
