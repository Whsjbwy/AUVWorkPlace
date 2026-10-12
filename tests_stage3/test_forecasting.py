"""Shared forecast contract: causal clocks, baseline parity and safe injection."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from auv3d.prediction import CAPredictor
from auv3d.remus import RemusPlant
from auv3d.remus_planner import RemusDWA, _clearance_positions, _forecast_tracks
from auv3d.sensors import Track
from auv3d.stage2_config import Stage2Config
from auv3d.stage3_forecasting import (
    CAObstacleForecaster, ForecastContext, ObstacleForecast,
)
from auv3d.terrain import HeightField


def observed_track(count=3):
    times = [0.0, 0.7, 2.0][:count]
    positions = [np.array([70.0 + 1.7*t + .2*t*t, 80.0 - .3*t, 30.0]) for t in times]
    return Track(7, 4.0, times, positions)


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("age", [0.0, 0.8, 4.0])
def test_default_ca_positions_and_enclosing_formula_exactly_match_stage2(count, age):
    cfg, track = Stage2Config(), observed_track(count)
    now = track.times[-1] + age
    offsets = np.array([0.0, 0.02, 1.0, 9.0, 24.0])
    expected_positions = CAPredictor(cfg.observation_period_s).predict(track, now + offsets)
    speed = 0.0 if count == 1 else np.linalg.norm(track.positions[-1] - track.positions[-2]) / (track.times[-1] - track.times[-2])
    margin = cfg.prediction_margin_m + age * max(1.0, speed)
    expected_radius = track.radius_m + cfg.body_radius_m + margin
    forecast = CAObstacleForecaster(cfg).forecast(track, now + offsets, now)
    np.testing.assert_array_equal(forecast.absolute_times, now + offsets)
    np.testing.assert_array_equal(forecast.positions, expected_positions)
    np.testing.assert_array_equal(forecast.enclosing_radii_m,
                                  np.full(len(offsets), expected_radius))
    assert forecast.metadata["future_truth_used"] is False
    assert forecast.metadata["calibrated_radii_applied"] is False
    legacy_tuple = _forecast_tracks([track], offsets, now, cfg)[0]
    np.testing.assert_array_equal(legacy_tuple[0], expected_positions)
    assert legacy_tuple[1] == expected_radius


def test_ca_explicit_history_interval_is_separate_from_sensor_delivery_period():
    cfg = replace(Stage2Config(), prediction_history_dt_s=3.0, observation_period_s=0.5)
    track = observed_track()
    times = np.array([2.0, 5.0, 11.0])
    forecast = CAObstacleForecaster(cfg).forecast(track, times, 2.0)
    np.testing.assert_array_equal(forecast.positions, CAPredictor(3).predict(track, times))
    assert forecast.metadata["history_dt_s"] == 3.0


@pytest.mark.parametrize("times", [[], [1.0, 1.0], [2.0, 1.0], [1.999, 3.0],
                                   [2.0, float("nan")], [[2.0, 3.0]]])
def test_absolute_forecast_time_contract_rejects_bad_requests(times):
    with pytest.raises(ValueError):
        ForecastContext(Stage2Config()).forecast(observed_track(), times, 2.0)


def test_future_measurement_rejected_even_if_requested_forecast_is_later():
    with pytest.raises(ValueError, match="future"):
        ForecastContext(Stage2Config()).forecast(observed_track(), [3.0, 4.0], 1.9)


@pytest.mark.parametrize("extra", [-1.0, float("nan"), float("inf")])
def test_memory_uncertainty_cannot_shrink_or_poison_forecast(extra):
    context = ForecastContext(Stage2Config())
    context.begin({7: extra})
    with pytest.raises(ValueError, match="uncertainty"):
        context.forecast(observed_track(), [2.0, 3.0], 2.0)


class CountingCA(CAObstacleForecaster):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.calls = 0

    def forecast(self, *args, **kwargs):
        self.calls += 1
        return super().forecast(*args, **kwargs)


def test_one_planning_context_reuses_only_identical_history_clock_and_uncertainty():
    cfg, track = Stage2Config(), observed_track()
    predictor = CountingCA(cfg)
    context = ForecastContext(cfg, predictor)
    context.begin({7: 1.5})
    first = context.forecast(track, [2.0, 3.0], 2.0)
    cached = context.forecast(deepcopy(track), np.array([2.0, 3.0]), 2.0)
    assert first is cached
    assert predictor.calls == 1
    assert len(context.records) == 1
    context.extra_uncertainty[7] = 2.0
    second = context.forecast(track, [2.0, 3.0], 2.0)
    assert predictor.calls == 2
    np.testing.assert_allclose(second.enclosing_radii_m-first.enclosing_radii_m, .5)
    context.begin({7: 2.0})
    context.forecast(track, [2.0, 3.0], 2.0)
    assert predictor.calls == 3
    assert len(context.records) == 1


@pytest.mark.parametrize("changed", ["times", "positions", "radius", "request_times", "now"])
def test_cache_invalidates_when_causal_forecast_inputs_change(changed):
    cfg, track = Stage2Config(), observed_track()
    predictor = CountingCA(cfg)
    context = ForecastContext(cfg, predictor)
    context.forecast(track, [3.0, 4.0], 2.0)
    request_times, now = [3.0, 4.0], 2.0
    if changed == "times":
        track.times[-1] = 1.9
    elif changed == "positions":
        track.positions[-1][0] += .3
    elif changed == "radius":
        track.radius_m += .5
    elif changed == "request_times":
        request_times = [3.0, 4.1]
    elif changed == "now":
        now = 2.1
    context.forecast(track, request_times, now)
    assert predictor.calls == 2


def test_forecast_copies_request_times_and_does_not_mutate_measurement_history():
    track, times = observed_track(), np.array([2.0, 3.0, 4.0])
    before = deepcopy(track)
    forecast = ForecastContext(Stage2Config()).forecast(track, times, 2.0)
    assert not np.shares_memory(times, forecast.absolute_times)
    for supplied in track.positions:
        assert not np.shares_memory(forecast.positions, supplied)
    np.testing.assert_array_equal(track.positions, before.positions)
    assert track.times == before.times
    assert track.radius_m == before.radius_m
    np.testing.assert_array_equal(times, [2.0, 3.0, 4.0])


class VariableRadiusForecaster:
    name = "variable_radius_test_fixture"

    def __init__(self, radius_end=25.0):
        self.radius_end = radius_end
        self.calls = []

    def forecast(self, track, absolute_times, now, extra_uncertainty_m=0.0):
        times = np.array(absolute_times, dtype=float)
        self.calls.append((times.copy(), now, extra_uncertainty_m))
        return ObstacleForecast(track.object_id, times,
            np.repeat(track.positions[-1][None, :], len(times), axis=0),
            np.linspace(5.0, self.radius_end, len(times)) + extra_uncertainty_m,
            {"predictor": self.name, "future_truth_used": False})


def test_planner_accepts_injected_variable_radius_contract_and_preserves_real_plant():
    plant = RemusPlant(depth=30.0, heading_deg=0, rpm=1100,
                       eta0=[20.0, 100.0, 30.0, .03, -.02, .04],
                       initialize_references=True)
    before = deepcopy((plant.eta, plant.nu, plant.actual, plant.time_s, vars(plant.vehicle)))
    cfg = replace(Stage2Config(), horizon_s=.4, control_dt=.2)
    track = Track(7, 4.0, [0.0], [np.array([40.0, 100.0, 30.0])])
    track_before = deepcopy(track)
    predictor = VariableRadiusForecaster()
    planner = RemusDWA(cfg, forecaster=predictor)
    planner.set_memory_uncertainty({7: 2.0})
    plan = planner.plan(plant, [210.0, 100.0, 30.0],
                        HeightField.flat(260.0, 200.0, 90.0), [track], 0.0)
    assert plan.emergency and not plan.emergency_verified_safe
    assert plan.minimum_predicted_clearance_m < 0
    assert len(predictor.calls) == 2  # Ranking time grid and the shared physical grid.
    assert all(call[2] == 2.0 for call in predictor.calls)
    assert all(record["predictor"] == predictor.name for record in plan.forecast_records)
    eta, nu, actual, stamp, vehicle = before
    np.testing.assert_array_equal(plant.eta, eta)
    np.testing.assert_array_equal(plant.nu, nu)
    np.testing.assert_array_equal(plant.actual, actual)
    assert plant.time_s == stamp
    for name, value in vehicle.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(plant.vehicle, name), value)
        else:
            assert getattr(plant.vehicle, name) == value
    assert track.times == track_before.times
    np.testing.assert_array_equal(track.positions, track_before.positions)


def test_time_varying_radius_uses_larger_endpoint_for_entire_segment_and_batch():
    cfg = Stage2Config()
    terrain = HeightField.flat(260.0, 200.0, 90.0)
    points = np.array([[20.0, 100.0, 30.0]] * 3)
    obstacle = np.array([[30.0, 100.0, 30.0]] * 3)
    # Radius peaks at the middle: both adjacent segments require the radius 9.
    variable = _clearance_positions(points, terrain, [(obstacle, np.array([1.0, 9.0, 1.0]))], cfg)
    fixed = _clearance_positions(points, terrain, [(obstacle, 9.0)], cfg)
    assert variable == fixed == 1.0
    batch = np.stack([points, points + np.array([0.0, 2.0, 0.0])])
    np.testing.assert_allclose(_clearance_positions(batch, terrain,
        [(obstacle, np.array([1.0, 9.0, 1.0]))], cfg),
        [1.0, np.hypot(10, 2) - 9])


@pytest.mark.parametrize("radii", [[1.0, 2.0], [1.0, -1.0, 2.0],
                                   [1.0, float("nan"), 2.0]])
def test_variable_radius_shape_and_values_cannot_create_false_clearance(radii):
    points = np.array([[20.0, 100.0, 30.0]] * 3)
    obstacle = np.array([[30.0, 100.0, 30.0]] * 3)
    with pytest.raises(ValueError):
        _clearance_positions(points, HeightField.flat(260.0, 200.0, 90.0),
                             [(obstacle, np.array(radii))], Stage2Config())


@pytest.mark.parametrize("bad_field", ["absolute_times", "object_id"])
def test_injected_forecaster_must_preserve_requested_time_anchor_and_identity(bad_field):
    class WrongAnchor(VariableRadiusForecaster):
        def forecast(self, *args, **kwargs):
            result = super().forecast(*args, **kwargs)
            if bad_field == "absolute_times":
                return replace(result, absolute_times=result.absolute_times + 0.1)
            return replace(result, object_id=result.object_id + 1)
    with pytest.raises(ValueError):
        ForecastContext(Stage2Config(), WrongAnchor()).forecast(observed_track(), [2.0, 3.0], 2.0)
