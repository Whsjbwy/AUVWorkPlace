"""Explicit causal predictor alternatives; the stage3 CA implementation is intact.

The filtered alternatives fit a fixed small polynomial to delivered positions,
anchored at the latest measured position. They are engineering noise controls,
not learned models, and their parameters are part of each calibration contract.
"""
from __future__ import annotations

import numpy as np

from .sensors import validate_track
from .stage3_forecasting import (CAObstacleForecaster, ObstacleForecast,
                                 radius_components, validate_absolute_times)


PREDICTORS = ('ca', 'cv', 'filtered_cv')
FILTER_HISTORY_LENGTH = 6


def canonical_predictor(predictor):
    return 'filtered_cv' if predictor == 'filtered' else predictor


def forecaster_contract(predictor, cfg):
    """Serializable *actual parameters*, never a file-identity check."""
    predictor = canonical_predictor(predictor)
    if predictor not in PREDICTORS:
        raise ValueError(f'unsupported stage5 predictor: {predictor}')
    common = {'predictor': predictor, 'version': 1,
              'time_reference': 'absolute simulation time; latest acquisition anchor',
              'future_truth_used': False}
    if predictor == 'ca':
        return dict(common, algorithm='unchanged stage3 CAPredictor',
                    history_dt_s=float(cfg.prediction_history_dt_s or cfg.observation_period_s),
                    history_length=6)
    if predictor == 'cv':
        return dict(common, algorithm='last two actual delivered acquisition times',
                    history_length=2, position_anchor='latest measured position')
    return dict(common, algorithm='latest-position-anchored causal least squares',
                history_length=FILTER_HISTORY_LENGTH,
                polynomial_degree=1,
                position_anchor='latest measured position',
                insufficient_history='hold with one observation; linear with two',
                acceleration_clipping='none', velocity_clipping='none')


class CausalPolynomialForecaster:
    """Stateless delivered-history CV or filtered polynomial extrapolation."""
    def __init__(self, predictor, cfg):
        predictor = canonical_predictor(predictor)
        if predictor not in PREDICTORS[1:]:
            raise ValueError('polynomial forecaster requires cv/filtered_cv')
        self.name, self.cfg = predictor, cfg
        self.contract = forecaster_contract(predictor, cfg)

    def forecast(self, track, absolute_times, now, extra_uncertainty_m=0.):
        validate_track(track, now)
        times = validate_absolute_times(absolute_times, now)
        components, age, speed = radius_components(track, now, self.cfg, extra_uncertainty_m)
        count = 2 if self.name == 'cv' else FILTER_HISTORY_LENGTH
        observed_t = np.asarray(track.times[-count:], dtype=float)
        observed_p = np.asarray(track.positions[-count:], dtype=float)
        degree = min(1, len(observed_t)-1)
        velocity, acceleration = np.zeros(3), np.zeros(3)
        if degree:
            # Scale times for conditioning. The intercept is fixed at the latest
            # measured point; all nonzero design rows are strictly in the past.
            delta = observed_t[:-1] - observed_t[-1]
            scale = float(np.max(np.abs(delta)))
            design = np.column_stack([(delta/scale)**power
                                      for power in range(1, degree+1)])
            coeff, _, rank, _ = np.linalg.lstsq(design,
                observed_p[:-1]-observed_p[-1], rcond=None)
            if rank != degree:
                raise ValueError('delivered history is numerically rank deficient')
            velocity = coeff[0]/scale
            if degree == 2:
                acceleration = 2*coeff[1]/scale**2
        offsets = times-observed_t[-1]
        points = (observed_p[-1] + offsets[:, None]*velocity
                  + .5*offsets[:, None]**2*acceleration)
        radius = sum(components.values())
        result = ObstacleForecast(int(track.object_id), times.copy(), points,
            np.full(times.shape, radius),
            {'predictor': self.name, 'predictor_contract': dict(self.contract),
             'latest_measurement_time_s': float(observed_t[-1]),
             'planning_time_s': float(now), 'observation_age_s': float(age),
             'observed_speed_mps': speed, 'history_samples_used': len(observed_t),
             'fitted_degree': degree, 'estimated_velocity_mps': velocity.tolist(),
             'estimated_acceleration_mps2': acceleration.tolist(),
             'radius_components': components, 'calibrated_radii_applied': False,
             'trained_horizon_s': None, 'extension_mode': 'not_applicable',
             'future_truth_used': False})
        result.validate(now)
        return result


def make_forecaster(predictor, cfg):
    """Inject this object into ForecastContext/REMUS without changing old CA."""
    predictor = canonical_predictor(predictor)
    forecaster_contract(predictor, cfg)  # Validate IDs and actual clock settings.
    if predictor == 'ca':
        return CAObstacleForecaster(cfg)
    return CausalPolynomialForecaster(predictor, cfg)
