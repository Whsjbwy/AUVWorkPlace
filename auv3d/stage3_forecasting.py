"""Causal obstacle forecast contract shared by REMUS planners.

Forecasts use absolute simulation timestamps. Radius components remain explicit
so the existing P25 empirical ball can later be consumed without calling it an
ECP probability or a platform safety guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .prediction import CAPredictor
from .sensors import validate_track


@dataclass(frozen=True)
class ObstacleForecast:
    object_id: int
    absolute_times: np.ndarray
    positions: np.ndarray
    enclosing_radii_m: np.ndarray
    metadata: dict

    def validate(self, now):
        times, points, radii = self.absolute_times, self.positions, self.enclosing_radii_m
        if (times.ndim != 1 or len(times) < 1 or points.shape != (len(times), 3)
                or radii.shape != times.shape):
            raise ValueError('forecast requires matching [time], [time,3], [time] arrays')
        if (not np.isfinite(now) or not np.isfinite(times).all()
                or not np.isfinite(points).all() or not np.isfinite(radii).all()
                or np.any(radii <= 0) or np.any(np.diff(times) <= 0)
                or np.any(times < now-1e-9)):
            raise ValueError('forecast times/positions/positive radii must be finite and causal')


class ObstacleForecaster(Protocol):
    name: str

    def forecast(self, track, absolute_times, now, extra_uncertainty_m=0.) -> ObstacleForecast: ...


def validate_absolute_times(absolute_times, now):
    times = np.asarray(absolute_times, dtype=float)
    if (not np.isfinite(now) or times.ndim != 1 or len(times) < 1
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or np.any(times < now-1e-9)):
        raise ValueError('forecast absolute times must be finite, increasing and >= planning now')
    return times


def observation_speed(track):
    if len(track.times) < 2:
        return 0.
    return float(np.linalg.norm(np.asarray(track.positions[-1])-track.positions[-2])
                 / (track.times[-1]-track.times[-2]))


def radius_components(track, now, cfg, extra_uncertainty_m):
    validate_track(track, now)
    if not np.isfinite(extra_uncertainty_m) or extra_uncertainty_m < 0:
        raise ValueError('memory uncertainty must be finite and nonnegative')
    age, speed = max(0., now-track.times[-1]), observation_speed(track)
    components = {'obstacle_radius_m':float(track.radius_m),
                  'vehicle_radius_m':float(cfg.body_radius_m),
                  'fixed_prediction_margin_m':float(cfg.prediction_margin_m),
                  'stage2_age_margin_m':float(age*max(1., speed)),
                  'memory_extra_uncertainty_m':float(extra_uncertainty_m)}
    return components, age, speed


class CAObstacleForecaster:
    name = 'ca'

    def __init__(self, cfg):
        self.cfg = cfg
        self.history_dt = cfg.prediction_history_dt_s or cfg.observation_period_s
        self.predictor = CAPredictor(self.history_dt)

    def forecast(self, track, absolute_times, now, extra_uncertainty_m=0.):
        components, age, speed = radius_components(track, now, self.cfg, extra_uncertainty_m)
        times = validate_absolute_times(absolute_times, now)
        radius = (components['obstacle_radius_m'] + components['vehicle_radius_m']
                  + (components['fixed_prediction_margin_m'] + components['stage2_age_margin_m'])
                  + components['memory_extra_uncertainty_m'])
        result = ObstacleForecast(int(track.object_id), times.copy(),
                                  np.asarray(self.predictor.predict(track, times)),
                                  np.full(times.shape, radius),
                                  {'predictor':'ca','latest_measurement_time_s':float(track.times[-1]),
                                   'planning_time_s':float(now),'observation_age_s':float(age),
                                   'history_dt_s':float(self.history_dt),'observed_speed_mps':speed,
                                   'radius_components':components,'calibrated_radii_applied':False,
                                   'trained_horizon_s':None,'extension_mode':'not_applicable',
                                   'future_truth_used':False})
        result.validate(now)
        return result


def make_obstacle_forecaster(cfg) -> ObstacleForecaster:
    if cfg.obstacle_predictor == 'ca':
        return CAObstacleForecaster(cfg)
    raise ValueError(f'unsupported obstacle predictor: {cfg.obstacle_predictor}')


class ForecastContext:
    """One planning call's immutable-history cache, with no vehicle ownership."""
    def __init__(self, cfg, predictor=None):
        self.predictor = predictor or make_obstacle_forecaster(cfg)
        self.extra_uncertainty = {}
        self.records = []
        self._cache = []

    def begin(self, uncertainty_by_id=None):
        self.extra_uncertainty = dict(uncertainty_by_id or {})
        self.records, self._cache = [], []

    def forecast(self, track, absolute_times, now):
        validate_track(track, now)
        times = validate_absolute_times(absolute_times, now)
        extra = self.extra_uncertainty.get(track.object_id, 0.)
        history_times, history_positions = np.asarray(track.times), np.asarray(track.positions)
        for cached in self._cache:
            if (cached[0] == track.object_id and cached[1] == now and cached[2] == extra
                    and np.array_equal(cached[3], times)
                    and np.array_equal(cached[4], history_times)
                    and np.array_equal(cached[5], history_positions) and cached[6] == track.radius_m):
                return cached[7]
        result = self.predictor.forecast(track, times, now, extra)
        result.validate(now)
        if result.object_id != track.object_id or not np.array_equal(result.absolute_times, times):
            raise ValueError('forecast must preserve requested object identity and absolute timestamps')
        # Private arrays do not alias observation histories or planner times.
        self._cache.append((track.object_id, now, extra, times.copy(), history_times.copy(),
                            history_positions.copy(), track.radius_m, result))
        self.records.append(dict(result.metadata, object_id=int(track.object_id),
                                 forecast_start_s=float(times[0]),forecast_end_s=float(times[-1])))
        return result
