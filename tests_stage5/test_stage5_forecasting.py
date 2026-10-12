"""Causal clocks, unchanged legacy CA and the stated velocity-noise control."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from auv3d.sensors import Track
from auv3d.stage2_config import Stage2Config
from auv3d.stage3_forecasting import CAObstacleForecaster, ForecastContext
from auv3d.stage5_forecasting import forecaster_contract, make_forecaster


def track(times=(0., .7, 2., 3.1, 4., 5.6)):
    return Track(11, 2., list(times),
                 [np.array([80.+1.5*t, 40.-.3*t, 30.+.1*t]) for t in times])


@pytest.mark.parametrize('predictor', ['ca', 'cv', 'filtered_cv'])
def test_forecaster_keeps_absolute_clock_and_never_mutates_delivered_history(predictor):
    cfg, history = Stage2Config(), track()
    before = deepcopy(history)
    now, requested = 6., np.array([6., 6.02, 15., 30.])
    output = ForecastContext(cfg, make_forecaster(predictor, cfg)).forecast(history, requested, now)
    np.testing.assert_array_equal(output.absolute_times, requested)
    assert not np.shares_memory(output.absolute_times, requested)
    assert output.metadata['future_truth_used'] is False
    np.testing.assert_array_equal(history.positions, before.positions)
    assert history.times == before.times
    with pytest.raises(ValueError, match='future'):
        make_forecaster(predictor, cfg).forecast(history, requested, 5.5)


@pytest.mark.parametrize('predictor', ['cv', 'filtered_cv'])
@pytest.mark.parametrize('age', [0., 2.3])
def test_irregular_delayed_linear_history_is_extrapolated_from_acquisition_time(predictor, age):
    history, cfg = track(), Stage2Config()
    now = history.times[-1]+age
    requested = now+np.array([0., .02, 9., 24.])
    output = make_forecaster(predictor, cfg).forecast(history, requested, now)
    expected = np.column_stack([80.+1.5*requested, 40.-.3*requested, 30.+.1*requested])
    np.testing.assert_allclose(output.positions, expected, atol=1e-12)
    np.testing.assert_allclose(output.enclosing_radii_m,
        history.radius_m+cfg.body_radius_m+cfg.prediction_margin_m
        +age*np.linalg.norm([1.5, -.3, .1]))


@pytest.mark.parametrize('predictor', ['cv', 'filtered_cv'])
def test_one_measurement_means_hold_and_filtered_does_not_denoise_latest_position(predictor):
    history, cfg = track((3.,)), Stage2Config()
    output = make_forecaster(predictor, cfg).forecast(history, [4., 28.], 4.)
    np.testing.assert_array_equal(output.positions, np.repeat(history.positions, 2, axis=0))
    multi = track()
    output = make_forecaster(predictor, cfg).forecast(multi, [5.6, 29.6], 5.6)
    np.testing.assert_array_equal(output.positions[0], multi.positions[-1])


def test_ca_factory_is_original_type_and_exact_numerical_output():
    cfg = replace(Stage2Config(), prediction_history_dt_s=.5)
    history = track()
    history.positions[-1] += [.02, -.03, .01]
    actual = make_forecaster('ca', cfg)
    assert type(actual) is CAObstacleForecaster
    old = CAObstacleForecaster(cfg).forecast(history, [6., 15., 30.], 6., 2.)
    new = actual.forecast(history, [6., 15., 30.], 6., 2.)
    np.testing.assert_array_equal(old.positions, new.positions)
    np.testing.assert_array_equal(old.enclosing_radii_m, new.enclosing_radii_m)
    assert old.metadata == new.metadata


@pytest.mark.parametrize('noise', [.02, .2])
def test_fixed_filtered_velocity_fit_reduces_stationary_long_horizon_noise(noise):
    cfg, rng = Stage2Config(), np.random.default_rng(508)
    sums = {'cv': 0., 'filtered_cv': 0.}
    # Paired samples and fixed parameters, not selecting a test-set filter.
    for _ in range(120):
        history = Track(1, 2., list(np.arange(6.)),
                        [np.array([80., 50., 30.])+rng.normal(0, noise, 3) for _ in range(6)])
        for predictor in sums:
            point = make_forecaster(predictor, cfg).forecast(history, [29.], 5.).positions[0]
            sums[predictor] += float(np.sum((point-[80., 50., 30.])**2))
    assert sums['filtered_cv'] < .5*sums['cv']


def test_alias_has_one_canonical_calibration_contract_and_invalid_id_fails():
    cfg = Stage2Config()
    assert make_forecaster('filtered', cfg).name == 'filtered_cv'
    assert forecaster_contract('filtered', cfg) == forecaster_contract('filtered_cv', cfg)
    with pytest.raises(ValueError, match='unsupported'):
        make_forecaster('future_truth', cfg)


@pytest.mark.parametrize('bad_times', [[5., 5.], [5., 4.], [5., float('nan')]])
def test_new_predictors_enforce_time_contract(bad_times):
    with pytest.raises(ValueError):
        make_forecaster('filtered_cv', Stage2Config()).forecast(track(), bad_times, 5.6)
