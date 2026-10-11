"""Real frozen-model smoke plus independent explicit-tail contract tests."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from auv3d.prediction import PPLNMPredictor, SphericalCalibration, pplnm_contract
from auv3d.remus import RemusPlant
from auv3d.remus_planner import RemusDWA
from auv3d.sensors import Track
from auv3d.stage2_config import Stage2Config
from auv3d.stage3_forecasting import CAObstacleForecaster, ForecastContext, make_obstacle_forecaster
from auv3d.stage3_pplnm import PPLNMObstacleForecaster
from auv3d.terrain import HeightField


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def track():
    # Backward finite differences: current velocity 6 m/s, acceleration 4/3 m/s².
    return Track(5, 4., [0., 3., 6.],
                 [np.array([x, 100., 30.]) for x in (0., 6., 24.)])


class FakePredictor:
    """Known trained-grid fixture; it is not a trained experimental model."""
    name = "pplnm_reimplementation"
    dt, max_horizon_s = 3., 9.
    checkpoint_meta = pplnm_contract(3., 1, "unit_test_fixture")

    def __init__(self):
        self.calls = 0

    def predict(self, measured_track, times):
        self.calls += 1
        offsets = np.asarray(times)-measured_track.times[-1]
        dx = np.interp(offsets, [0., 3., 6., 9.], [0., 3., 9., 18.])
        return measured_track.positions[-1]+np.column_stack([dx, np.zeros((len(dx), 2))])


def fake_calibration():
    calibration = SphericalCalibration(bins=(0., float("inf")), min_count=1)
    calibration.global_q = np.array([1., 2., 3.])
    calibration.quantiles = np.array([[1., 2., 3.]])
    calibration.counts = [1]
    calibration.metadata = dict(FakePredictor.checkpoint_meta)
    return calibration


def fake_forecaster(mode="cv_after9", **changes):
    options = dict(obstacle_predictor="pplnm", prediction_history_dt_s=3.,
                   pplnm_extension_mode=mode)
    options.update(changes)
    cfg = replace(Stage2Config(), **options)
    return PPLNMObstacleForecaster(cfg, predictor=FakePredictor(), calibration=fake_calibration())


@pytest.fixture(scope="module")
def actual_predictor():
    cfg = Stage2Config()
    return PPLNMPredictor(SOURCE_ROOT / cfg.pplnm_checkpoint, device="cpu", threads=2)


def test_existing_weight_and_calibration_load_and_real_forecast_is_finite(actual_predictor):
    cfg = replace(Stage2Config(), obstacle_predictor="pplnm", prediction_history_dt_s=3.)
    calibration = SphericalCalibration.load(SOURCE_ROOT / cfg.pplnm_calibration)
    adapter = PPLNMObstacleForecaster(cfg, predictor=actual_predictor, calibration=calibration)
    measured = track()
    times = np.array([6., 9., 12., 15., 18., 30.])
    result = adapter.forecast(measured, times, 6.)
    assert result.positions.shape == (6, 3) and np.isfinite(result.positions).all()
    expected = actual_predictor.predict(measured, np.array([6., 9., 12., 15.]))
    np.testing.assert_array_equal(result.positions[:4], expected)
    assert result.metadata["checkpoint_parameters"]["epochs_completed"] == 150
    assert result.metadata["trained_horizon_s"] == 9
    assert result.metadata["extension_point_count"] == 2
    assert not result.metadata["future_truth_used"]
    assert not result.metadata["model_fitted_on_this_experiment"]


@pytest.mark.parametrize("mode", ["cv_after9", "ca_after9"])
def test_explicit_tail_formula_uses_known_grid_and_measured_finite_differences(mode):
    adapter = fake_forecaster(mode)
    result = adapter.forecast(track(), np.array([6., 9., 12., 15., 18.]), 6.)
    np.testing.assert_array_equal(result.positions[:4, 0], [24., 27., 33., 42.])
    # CV endpoint slope is (18-9)/3=3. CA-at-boundary velocity is 6+9*4/3=18.
    # Both tails start at the learned boundary position 42, not at CA's position.
    expected = 42.+3.*3. if mode == "cv_after9" else 42.+3.*18.+.5*3.**2*(4./3.)
    assert result.positions[-1, 0] == pytest.approx(expected)
    assert result.metadata["trained_valid_mask"] == [True, True, True, True, False]
    assert result.metadata["extension_mode"] == mode


@pytest.mark.parametrize("mode", ["cv_after9", "ca_after9"])
def test_tail_position_is_continuous_at_nine_second_boundary(mode):
    result = fake_forecaster(mode).forecast(track(), [15.-1e-6, 15., 15.+1e-6], 6.)
    assert result.positions[1, 0] == 42.
    assert np.max(np.abs(result.positions[:, 0]-42.)) < 2e-5


def test_trained_support_is_anchored_to_acquisition_not_planning_or_delivery():
    result = fake_forecaster().forecast(track(), [8., 15., 17.], 8.)
    assert result.metadata["latest_measurement_time_s"] == 6.
    assert result.metadata["planning_time_s"] == 8.
    assert result.metadata["trained_support_end_s"] == 15.
    assert result.metadata["observation_age_s"] == 2.
    assert result.metadata["trained_valid_mask"] == [True, True, False]
    assert result.metadata["calibration_valid_mask"] == [True, True, False]
    assert result.metadata["p25_calibration_radii_m"] == pytest.approx([2/3, 3., None], nan_ok=True)


@pytest.mark.parametrize("mode", ["cv_after9", "ca_after9"])
def test_history_stale_more_than_nine_seconds_has_only_explicit_uncalibrated_tail(mode):
    result = fake_forecaster(mode).forecast(track(), [16., 18., 30.], 16.)
    assert result.metadata["trained_valid_mask"] == [False, False, False]
    assert result.metadata["calibration_valid_mask"] == [False, False, False]
    assert result.metadata["p25_calibration_radii_m"] == [None, None, None]
    assert result.metadata["extension_point_count"] == 3
    assert not result.metadata["calibrated_radii_applied"]


def test_reject_mode_never_silently_extends_trained_range():
    adapter = fake_forecaster("reject")
    assert adapter.forecast(track(), [6., 9., 15.], 6.).metadata["extension_point_count"] == 0
    with pytest.raises(ValueError, match="nine seconds"):
        adapter.forecast(track(), [15., 15.001], 6.)
    with pytest.raises(ValueError, match="nine seconds"):
        adapter.forecast(track(), [16., 17.], 16.)


@pytest.mark.parametrize("dt,horizon", [(1., 9.), (3., 12.), (float("nan"), 9.)])
def test_adapter_does_not_relabel_a_different_trained_model(dt, horizon):
    predictor = FakePredictor()
    predictor.dt, predictor.max_horizon_s = dt, horizon
    with pytest.raises(ValueError, match="dt3"):
        PPLNMObstacleForecaster(Stage2Config(), predictor=predictor, calibration=fake_calibration())


def test_explicit_history_sampling_must_match_checkpoint():
    with pytest.raises(ValueError, match="physical dt"):
        fake_forecaster(prediction_history_dt_s=.3)


@pytest.mark.parametrize("corruption", ["dt", "name", "architecture", "horizon", "negative"])
def test_model_calibration_matching_uses_parameters_and_dimensions(corruption):
    calibration = fake_calibration()
    if corruption == "dt":
        calibration.metadata["prediction_dt_s"] = 1.
    elif corruption == "name":
        calibration.metadata["predictor_name"] = "constant_acceleration_debug_baseline"
    elif corruption == "architecture":
        calibration.metadata["architecture"] = "another_architecture"
    elif corruption == "horizon":
        calibration.global_q = np.zeros(4)
    elif corruption == "negative":
        calibration.quantiles[0, 0] = -1.
    with pytest.raises(ValueError):
        PPLNMObstacleForecaster(Stage2Config(), predictor=FakePredictor(), calibration=calibration)


@pytest.mark.parametrize("changed", ["history_time", "history_position", "object_id"])
def test_learned_grid_cache_invalidates_on_measurement_history_or_identity(changed):
    measured, adapter = track(), fake_forecaster()
    adapter.forecast(measured, [6., 9.], 6.)
    adapter.forecast(measured, [8., 15., 30.], 8.)
    assert adapter.predictor.calls == 1  # Planning age/tail queries reuse a frozen measured grid.
    if changed == "history_time":
        measured.times[1] = 2.9
    elif changed == "history_position":
        measured.positions[-1][0] += .1
    elif changed == "object_id":
        measured.object_id += 1
    adapter.forecast(measured, [8., 15.], 8.)
    assert adapter.predictor.calls == 2


@pytest.mark.parametrize("age,extra", [(0., 0.), (2., 0.), (10., 7.)])
def test_pplnm_and_ca_have_exactly_same_enclosing_radius_in_predictor_only_ablation(age, extra):
    cfg = replace(Stage2Config(), prediction_history_dt_s=3.)
    now = 6.+age
    times = now+np.array([0., 3., 9., 24.])
    learned = fake_forecaster().forecast(track(), times, now, extra)
    ca = CAObstacleForecaster(cfg).forecast(track(), times, now, extra)
    np.testing.assert_array_equal(learned.enclosing_radii_m, ca.enclosing_radii_m)
    assert learned.metadata["radius_components"] == ca.metadata["radius_components"]


def test_real_learned_forecast_does_not_modify_track_or_original_plant(actual_predictor):
    cfg = replace(Stage2Config(), obstacle_predictor="pplnm", prediction_history_dt_s=3.,
                  horizon_s=.4, control_dt=.2)
    adapter = PPLNMObstacleForecaster(cfg, predictor=actual_predictor)
    measured = Track(5, 4., [0.], [np.array([80., 100., 30.])])
    measured_before = deepcopy(measured)
    plant = RemusPlant(depth=30., heading_deg=0., rpm=1100.,
        eta0=[30., 100., 30., .03, -.02, .04], initialize_references=True)
    before = deepcopy((plant.eta, plant.nu, plant.actual, plant.time_s, vars(plant.vehicle)))
    plan = RemusDWA(cfg, forecaster=adapter).plan(plant, [210., 100., 30.],
        HeightField.flat(260., 200., 90.), [measured], 0.)
    assert np.isfinite(plan.states).all()
    assert measured.times == measured_before.times
    np.testing.assert_array_equal(measured.positions, measured_before.positions)
    for current, original in zip((plant.eta, plant.nu, plant.actual), before[:3]):
        np.testing.assert_array_equal(current, original)
    assert plant.time_s == before[3]
    for key, value in before[4].items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(plant.vehicle, key), value)
        else:
            assert getattr(plant.vehicle, key) == value


def test_factory_switches_using_configuration_without_planner_code_changes(actual_predictor):
    ca = make_obstacle_forecaster(Stage2Config())
    learned = make_obstacle_forecaster(replace(Stage2Config(), obstacle_predictor="pplnm", prediction_history_dt_s=3.))
    assert isinstance(ca, CAObstacleForecaster)
    assert isinstance(learned, PPLNMObstacleForecaster)
    result = ForecastContext(learned.cfg, learned).forecast(track(), [6., 15., 30.], 6.)
    assert result.metadata["predictor"] == "pplnm"
    assert result.metadata["trained_valid_mask"] == [True, True, False]


def test_calibration_speed_bin_matches_resampled_dataset_not_last_raw_turn_interval():
    measured = Track(5, 4., [0., 1., 2., 3.], [
        np.array([0., 0., 30.]), np.array([1., 0., 30.]),
        np.array([1., 1., 30.]), np.array([0., 1., 30.])])
    # Three one-second sides of a square: last raw speed=1, dt3 displacement
    # speed=1/3. Original data.py assigns calibration bins using the latter.
    cfg = replace(Stage2Config(), prediction_history_dt_s=3., obstacle_predictor="pplnm")
    calibration = fake_calibration()
    called = []
    original_radius = calibration.radius

    def record_radius(speed, offsets, dt):
        called.append((speed, np.asarray(offsets).copy(), dt))
        return original_radius(speed, offsets, dt)

    calibration.radius = record_radius
    adapter = PPLNMObstacleForecaster(cfg, predictor=FakePredictor(), calibration=calibration)
    result = adapter.forecast(measured, [4., 6., 12., 16.], 4.)
    assert len(called) == 1
    assert called[0][0] == pytest.approx(1/3)
    np.testing.assert_array_equal(called[0][1], [1., 3., 9.])
    assert called[0][2] == 3.
    assert result.metadata["observed_speed_mps"] == 1.
    assert result.metadata["calibration_speed_mps"] == pytest.approx(1/3)
    assert "resampled history velocity" in result.metadata["calibration_speed_definition"]
    assert "not a new coverage guarantee" in result.metadata["calibration_mask_scope"]
    assert result.metadata["p25_calibration_radii_m"][-1] is None
    assert not result.metadata["calibrated_radii_applied"]
    ca = CAObstacleForecaster(cfg).forecast(measured, [4., 6., 12., 16.], 4.)
    # Navigation still uses raw-speed Stage2 age inflation, unchanged by the
    # corrected calibration lookup. No P25 quantile is silently added.
    np.testing.assert_array_equal(result.enclosing_radii_m, ca.enclosing_radii_m)
    assert result.metadata["radius_components"] == ca.metadata["radius_components"]
