"""Frozen nine-second PPLNM plus explicitly uncalibrated tail extension.

The support starts at the last *measurement*, not the planning/delivery time.
P25 empirical balls are exposed for later risk interfaces, not silently added
to the retained baseline's deterministic enclosing radius.
"""
from pathlib import Path

import numpy as np

from .prediction import PPLNMPredictor, SphericalCalibration, make_history
from .stage3_forecasting import (ObstacleForecast, radius_components,
                                 validate_absolute_times)


def _project_file(value):
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


class PPLNMObstacleForecaster:
    name = 'pplnm'

    def __init__(self, cfg, *, predictor=None, calibration=None):
        self.cfg = cfg
        self.predictor = predictor or PPLNMPredictor(_project_file(cfg.pplnm_checkpoint),
                                                    device='cpu', threads=cfg.pplnm_threads)
        self.history_dt = float(self.predictor.dt)
        self.trained_horizon = float(self.predictor.max_horizon_s)
        if not np.isclose(self.history_dt, 3.) or not np.isclose(self.trained_horizon, 9.):
            raise ValueError('this adapter requires the independent trained dt3 / nine-second version')
        if cfg.prediction_history_dt_s and not np.isclose(cfg.prediction_history_dt_s, self.history_dt):
            raise ValueError('PPLNM history sampling must match the checkpoint physical dt')
        if cfg.pplnm_extension_mode not in ('cv_after9', 'ca_after9', 'reject'):
            raise ValueError('unsupported explicit PPLNM extension mode')
        self.calibration = calibration or SphericalCalibration.load(_project_file(cfg.pplnm_calibration))
        self.calibration.validate_for(self.predictor, self.history_dt, 3)
        self._grid_cache = None

    def _trained_grid(self, track):
        times, points = np.asarray(track.times), np.asarray(track.positions)
        cache = self._grid_cache
        if (cache is not None and cache[0] == track.object_id
                and np.array_equal(cache[1], times) and np.array_equal(cache[2], points)):
            return cache[3], cache[4]
        offsets = np.arange(4) * self.history_dt
        grid = np.asarray(self.predictor.predict(track, times[-1]+offsets), dtype=float)
        if grid.shape != (4, 3) or not np.isfinite(grid).all():
            raise FloatingPointError('trained PPLNM grid must be finite [4,3]')
        if not np.allclose(grid[0], points[-1], atol=1e-10, rtol=0):
            raise ValueError('PPLNM zero-offset grid must preserve the observed anchor')
        history = make_history(track.times, track.positions, self.history_dt)
        self._grid_cache = (track.object_id, times.copy(), points.copy(), grid.copy(), history)
        return grid, history

    def forecast(self, track, absolute_times, now, extra_uncertainty_m=0.):
        components, age, speed = radius_components(track, now, self.cfg, extra_uncertainty_m)
        times = validate_absolute_times(absolute_times, now)
        offsets = times-float(track.times[-1])
        valid = offsets <= self.trained_horizon+1e-9
        mode = self.cfg.pplnm_extension_mode
        if mode == 'reject' and not valid.all():
            raise ValueError('requested forecast exceeds trained nine seconds from last measurement')
        grid, history = self._trained_grid(track)
        positions = np.column_stack([np.interp(offsets, np.arange(4)*self.history_dt, grid[:,j])
                                     for j in range(3)])
        tail = ~valid
        if tail.any():
            delta = offsets[tail]-self.trained_horizon
            if mode == 'cv_after9':
                velocity = (grid[-1]-grid[-2])/self.history_dt
                positions[tail] = grid[-1]+delta[:,None]*velocity
            else:
                acceleration = history.features[-1,6:9]
                velocity = history.features[-1,3:6]+self.trained_horizon*acceleration
                positions[tail] = (grid[-1]+delta[:,None]*velocity
                                   + .5*delta[:,None]**2*acceleration)
        # Exactly the same heuristic enclosing radius as CA for the first
        # predictor-only ablation. Empirical calibration is an exposed output.
        radius = (components['obstacle_radius_m']+components['vehicle_radius_m']
                  +(components['fixed_prediction_margin_m']+components['stage2_age_margin_m'])
                  +components['memory_extra_uncertainty_m'])
        calibration_radii = [None]*len(times)
        if valid.any():
            values = self.calibration.radius(speed, np.clip(offsets[valid],0.,self.trained_horizon),
                                             self.history_dt)
            for i, value in zip(np.flatnonzero(valid), values):
                calibration_radii[int(i)] = float(value)
        metadata = {
            'predictor':'pplnm','planning_time_s':float(now),
            'latest_measurement_time_s':float(track.times[-1]),
            'boundary_measurement_time_s':float(track.times[-1]),
            'observation_age_s':float(age),'observed_speed_mps':speed,
            'history_dt_s':self.history_dt,'trained_horizon_s':self.trained_horizon,
            'trained_support_end_s':float(track.times[-1])+self.trained_horizon,
            'trained_valid_mask':valid.tolist(),'calibration_valid_mask':valid.tolist(),
            'p25_calibration_radii_m':calibration_radii,
            'calibrated_radii_applied':False,'radius_components':components,
            'extension_mode':mode,'extension_point_count':int(tail.sum()),
            'extension_claim':'explicit engineering continuation; neither trained nor calibrated beyond nine seconds',
            'real_history_sample_count':len(track.times),
            'resampled_real_mask':history.real_mask.tolist(),
            'checkpoint_parameters':dict(self.predictor.checkpoint_meta),
            'future_truth_used':False,'model_fitted_on_this_experiment':False,
        }
        result = ObstacleForecast(int(track.object_id), times.copy(), positions,
                                  np.full(times.shape, radius), metadata)
        result.validate(now)
        return result
