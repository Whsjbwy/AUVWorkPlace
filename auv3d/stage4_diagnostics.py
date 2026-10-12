"""Offline controller replay and selected-reference risk diagnostics.

Controller restoration uses original .02-second reference execution and checks
every archived state/control sample.  It never infers integral/reference state
from a pose CSV.  Future obstacle truth has a separate explicit offline API and
is not accepted by a planner or risk policy.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .adapters.remus_guidance import ReferenceAction, RemusGuidance, clone_plant
from .geometry import swept_sphere_clearance
from .scenarios import make_stage2_scenario
from .sensors import Track, validate_track
from .stage2_config import Stage2Config
from .stage2_simulation import CONTROL_COLUMNS, STATE_COLUMNS, initialize_plant
from .stage3_forecasting import ForecastContext
from .stage4_calibration import _verify_observed


TIME_ATOL = 1e-8
STATE_ATOL = 1e-7
DEFAULT_ANCHORS_S = (20., 40., 60.)
ERROR_HORIZONS_S = (3., 6., 9., 12., 18., 24.)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _csv_matrix(path, columns):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or any(column not in reader.fieldnames for column in columns):
            raise ValueError(f"{Path(path).name} requires the original named columns")
        rows = [[float(row[column]) for column in columns] for row in reader]
    values = np.asarray(rows, dtype=float).reshape(-1, len(columns))
    if not np.isfinite(values).all():
        raise ValueError(f"{Path(path).name} requires finite values")
    return values


@dataclass(frozen=True)
class ReplayAnchor:
    plant: object
    tracks: list[Track]
    goal: np.ndarray
    previous_action: ReferenceAction | None
    now: float
    memory_uncertainty: dict[int, float]
    original_action: ReferenceAction
    original_feasible: bool
    original_predicted_clearance_m: float
    terrain: object
    cfg: Stage2Config
    scenario: str


def replay_anchors(case_dir, cfg=None, scenario=None, anchors=DEFAULT_ANCHORS_S):
    """Yield isolated, causally checked anchors after exact full-episode replay.

    Only fixed requested planning times with delivered tracks are returned.
    Missing/empty anchors are not replaced by favorable later times.  The
    archived *previous selected* action restores the planner's smoothness term;
    applied controls restore the real controller and actuator dynamic state.
    """
    directory = Path(case_dir)
    saved_cfg = Stage2Config.load(directory / "config.json")
    cfg = cfg or saved_cfg
    cfg.validate()
    if cfg.to_dict() != saved_cfg.to_dict():
        raise ValueError("requested replay configuration differs from archived actual parameters")
    summary = _json(directory / "summary.json")
    scenario = scenario or summary.get("scenario")
    if scenario != summary.get("scenario") or summary.get("backend") != "remus6dof" or summary.get("seed") != cfg.seed:
        raise ValueError("replay scenario/backend/seed disagrees with archived run")
    requested = np.asarray(tuple(anchors), dtype=float)
    if requested.ndim != 1 or not np.isfinite(requested).all() or np.any(requested < 0) or np.any(np.diff(requested) <= 0):
        raise ValueError("anchors must be finite, nonnegative and strictly increasing")
    if np.any(np.abs(requested / cfg.physics_dt - np.round(requested / cfg.physics_dt)) > TIME_ATOL):
        raise ValueError("anchors must align with the physical clock")
    states = _csv_matrix(directory / "states.csv", STATE_COLUMNS)
    controls = _csv_matrix(directory / "controls.csv", CONTROL_COLUMNS)
    planning = _csv_matrix(directory / "planning.csv", (
        "time_s", "predicted_clearance_m"))
    if not len(states) or len(controls) != len(states)-1:
        raise ValueError("archived states and interval controls must cover the same complete execution")
    clock = np.arange(len(states), dtype=float)*cfg.physics_dt
    if not np.allclose(states[:, 0], clock, atol=TIME_ATOL, rtol=0):
        raise ValueError("actual state timestamps do not follow the original physical clock")
    if not np.allclose(controls[:, 0], clock[:-1], atol=TIME_ATOL, rtol=0):
        raise ValueError("actual control timestamps do not match interval starts")
    diagnostics = _json(directory / "planning_diagnostics.json")
    selected = _json(directory / "selected_plans.json")
    packets = _json(directory / "observations.json")
    if any(not isinstance(value, list) for value in (diagnostics, selected, packets)):
        raise ValueError("archived planning and observation logs must be lists")
    scene = make_stage2_scenario(scenario, cfg)
    plant = initialize_plant(scene, cfg)
    frames = []
    for tick, state in enumerate(states):
        now = float(state[0])
        actual = np.r_[plant.time_s, plant.eta, plant.nu, plant.actual]
        if not np.allclose(actual, state, atol=STATE_ATOL, rtol=0):
            raise ValueError(f"exact REMUS replay disagrees with actual state at {now:g} s")
        if np.any(np.isclose(requested, now, atol=TIME_ATOL, rtol=0)):
            plans = [row for row in diagnostics if abs(float(row["time_s"])-now) <= TIME_ATOL]
            choices = [row for row in selected if abs(float(row["time_s"])-now) <= TIME_ATOL]
            if len(plans) > 1 or len(choices) > 1:
                raise ValueError("duplicate archived planning anchor")
            if bool(plans) != bool(choices):
                raise ValueError("archived diagnostics and selected plan disagree at requested anchor")
            if plans and choices and plans[0].get("tracks"):
                plan, choice = plans[0], choices[0]
                tracks = []
                for item in plan["tracks"]:
                    track = Track(int(item["object_id"]), float(item["radius"]),
                                  list(np.asarray(item["times_s"], float)),
                                  [row.copy() for row in np.asarray(item["positions"], float)])
                    _verify_observed(track, now, packets)
                    tracks.append(track)
                if len({track.object_id for track in tracks}) != len(tracks):
                    raise ValueError("duplicate object identity in archived tracks")
                preceding = [row for row in selected if float(row["time_s"]) < now-TIME_ATOL]
                previous = (ReferenceAction(*max(preceding, key=lambda row: float(row["time_s"]))["action"])
                            if preceding else None)
                original_action = ReferenceAction(*choice["action"])
                original_action.validate(plant)
                if previous is not None:
                    previous.validate(plant)
                match = planning[np.isclose(planning[:, 0], now, atol=TIME_ATOL, rtol=0)]
                if len(match) != 1:
                    raise ValueError("planning anchor needs exactly one archived clearance")
                goal = np.asarray(plan["planning_goal_ned_m"], float)
                uncertainty = {int(key): float(value) for key, value in
                               plan.get("memory_uncertainty_by_id", {}).items()}
                if goal.shape != (3,) or not np.isfinite(goal).all() or any(not np.isfinite(value) or value < 0 for value in uncertainty.values()):
                    raise ValueError("archived goal/uncertainty is invalid")
                frames.append(ReplayAnchor(
                    clone_plant(plant), tracks, goal.copy(), previous, now, uncertainty,
                    original_action, bool(choice["feasible"]),
                    float(match[0, 1]), scene.terrain, cfg, scenario))
        if tick < len(controls):
            control = controls[tick]
            reference = ReferenceAction(*control[1:4])
            sample = RemusGuidance.step(plant, reference, cfg.physics_dt)
            replayed_control = np.r_[now, reference.as_array(), sample.raw, sample.applied,
                                    plant.actual, np.asarray(sample.saturated, dtype=int)]
            if not np.allclose(replayed_control, control, atol=STATE_ATOL, rtol=0):
                raise ValueError(f"exact REMUS replay disagrees with actual control at {now:g} s")
    yield from frames


class OfflineTruthReplay:
    """Owned original-scene replay for post-run diagnostics only.

    ``evaluate`` checks one selected reference held over its entire forecast
    horizon.  That trajectory is counterfactual after the next replanning call;
    these results are neither actual task collisions nor all-candidate rates.
    """
    def __init__(self, cfg, scenario, saved_snapshots=None):
        cfg.validate()
        self.cfg, self.scenario = cfg, scenario
        self._scene = make_stage2_scenario(scenario, cfg)
        snapshot = self._scene.obstacles.snapshot()
        self.ids = np.asarray(snapshot.ids, dtype=int)
        self.radii = np.asarray(snapshot.radii, dtype=float)
        if self.ids.ndim != 1 or len(set(self.ids.tolist())) != len(self.ids):
            raise ValueError("truth replay requires unique obstacle identities")
        self._positions = []
        self._saved = {}
        self.checked_saved_snapshots = 0
        self.maximum_saved_position_error_m = 0.
        for row in saved_snapshots or []:
            stamp = float(row["time_s"])
            tick = int(round(stamp / cfg.physics_dt))
            if not np.isfinite(stamp) or stamp < 0 or abs(tick*cfg.physics_dt-stamp) > TIME_ATOL or tick in self._saved:
                raise ValueError("saved truth snapshots must have unique aligned timestamps")
            self._saved[tick] = row

    def _ensure(self, end_tick):
        while len(self._positions) <= end_tick:
            tick = len(self._positions)
            snapshot = self._scene.obstacles.snapshot()
            ids = np.asarray(snapshot.ids, dtype=int)
            points = np.asarray(snapshot.positions, dtype=float).reshape(-1, 3)
            radii = np.asarray(snapshot.radii, dtype=float)
            if (not np.array_equal(ids, self.ids) or not np.array_equal(radii, self.radii)
                    or points.shape != (len(ids), 3) or not np.isfinite(points).all()):
                raise ValueError("original obstacle replay changed identity/size or produced invalid positions")
            if tick in self._saved:
                row = self._saved[tick]
                expected = np.asarray(row["positions"], dtype=float).reshape(-1, 3)
                if (not np.array_equal(self.ids, np.asarray(row["ids"], int))
                        or not np.array_equal(self.radii, np.asarray(row["radii"], float))
                        or expected.shape != points.shape or not np.isfinite(expected).all()):
                    raise ValueError("saved obstacle IDs/radii disagree with offline original-scene replay")
                error = float(np.max(np.abs(points-expected))) if len(points) else 0.
                if error > STATE_ATOL:
                    raise ValueError("saved obstacle positions disagree with offline original-scene replay")
                self.checked_saved_snapshots += 1
                self.maximum_saved_position_error_m = max(self.maximum_saved_position_error_m, error)
            self._positions.append(points.copy())
            self._scene.obstacles.step(self.cfg.physics_dt)

    def evaluate(self, states, tracks, now, *, quantile_m=None, feasible=False,
                 uncertainty_by_id=None, horizons=ERROR_HORIZONS_S):
        """Return physical-sweep truth gaps and observed-target score coverage.

        ECP scores use the per-interval difference of the *nearest* predicted
        and labelled physical sphere gaps, then the maximum positive difference
        over intervals.  Ball scores use max 3D position error.  Missing targets
        cannot be invented as prediction labels.  RMSE sums/counts permit
        correct weighted aggregation rather than averaging case RMSE values.
        """
        cfg = self.cfg
        rows = np.asarray(states, dtype=float)
        expected_times = np.arange(round(cfg.horizon_s/cfg.physics_dt)+1, dtype=float)*cfg.physics_dt
        if (rows.shape != (len(expected_times), 16) or not np.isfinite(rows).all()
                or not np.allclose(rows[:, 0], expected_times, atol=TIME_ATOL, rtol=0)):
            raise ValueError("offline selected prediction must retain every original physics tick over the full horizon")
        if not np.isfinite(now) or now < 0 or abs(now/cfg.physics_dt-round(now/cfg.physics_dt)) > TIME_ATOL:
            raise ValueError("offline anchor must lie on the original nonnegative physical clock")
        if quantile_m is not None and (not np.isfinite(quantile_m) or quantile_m < 0):
            raise ValueError("archived calibrated quantile must be finite and nonnegative")
        requested_horizons = np.asarray(tuple(horizons), dtype=float)
        if requested_horizons.ndim != 1 or not np.isfinite(requested_horizons).all() or np.any(requested_horizons <= 0) or np.any(np.diff(requested_horizons) <= 0):
            raise ValueError("error horizons must be positive, finite and strictly increasing")
        for track in tracks:
            validate_track(track, now)
        if len({track.object_id for track in tracks}) != len(tracks):
            raise ValueError("offline observed targets must have distinct identities")
        first = round(now/cfg.physics_dt)
        self._ensure(first+len(rows)-1)
        truth = np.asarray(self._positions[first:first+len(rows)])
        path = rows[:, 1:4]
        terrain_gap = min(
            min(self._scene.terrain.segment_clearance(a, b, cfg.body_radius_m)
                for a, b in zip(path[:-1], path[1:])),
            float(np.min(cfg.depth_limit_m-cfg.body_radius_m-path[:, 2])))
        actual_all = float(terrain_gap)
        truth_gaps = None
        if len(self.ids):
            truth_gaps = swept_sphere_clearance(
                path[:-1, None, :], path[1:, None, :], truth[:-1], truth[1:],
                self.radii[None, :]+cfg.body_radius_m)
            actual_all = min(actual_all, float(np.min(truth_gaps)))
        context = ForecastContext(cfg)
        context.begin(uncertainty_by_id)
        predictions, observed_labels, pred_gaps, observed_gaps = [], [], [], []
        for track in tracks:
            index = np.flatnonzero(self.ids == track.object_id)
            if len(index) != 1:
                raise ValueError("observed target identity is absent from original truth scene")
            slot = int(index[0])
            if abs(float(self.radii[slot])-track.radius_m) > STATE_ATOL:
                raise ValueError("observed and original physical obstacle sizes disagree")
            forecast = context.forecast(track, now+expected_times, now)
            predictions.append(forecast.positions)
            observed_labels.append(truth[:, slot, :])
            pred_gaps.append(swept_sphere_clearance(
                path[:-1], path[1:], forecast.positions[:-1], forecast.positions[1:],
                track.radius_m+cfg.body_radius_m))
            observed_gaps.append(truth_gaps[:, slot])
        score = ball_error = coverage_ecp = coverage_ball = None
        actual_observed = float(terrain_gap)
        if tracks:
            predicted_gaps = np.min(np.asarray(pred_gaps), axis=0)
            labelled_gaps = np.min(np.asarray(observed_gaps), axis=0)
            score = float(np.maximum(predicted_gaps-labelled_gaps, 0.).max())
            squared_errors = np.sum((np.asarray(predictions)-np.asarray(observed_labels))**2, axis=-1)
            ball_error = float(np.sqrt(np.max(squared_errors)))
            actual_observed = min(actual_observed, float(np.min(labelled_gaps)))
            if quantile_m is not None:
                coverage_ecp = bool(score <= quantile_m+TIME_ATOL)
                coverage_ball = bool(ball_error <= quantile_m+TIME_ATOL)
        statistics = {}
        for horizon in requested_horizons:
            key = f"{horizon:g}"
            count, total = 0, 0.
            if tracks and horizon <= cfg.horizon_s+TIME_ATOL:
                tick = int(round(horizon/cfg.physics_dt))
                if abs(tick*cfg.physics_dt-horizon) > TIME_ATOL:
                    raise ValueError("requested error horizon must align with the physical clock")
                count, total = len(tracks), float(np.sum(squared_errors[:, tick]))
            statistics[key] = {"sample_count": count, "squared_error_sum_m2": total,
                               "rmse_m": float(np.sqrt(total/count)) if count else None}
        return {
            "scope": "offline selected reference held for the full horizon; counterfactual after replanning, not all-candidate misclassification or actual task collision",
            "anchor_s": float(now), "horizon_s": float(cfg.horizon_s),
            "physics_dt_s": cfg.physics_dt, "forecast_feasible": bool(feasible),
            "selected_future_truth_min_clearance_m": float(actual_all),
            "selected_observed_future_truth_min_clearance_m": float(actual_observed),
            "selected_forecast_false_feasible": bool(feasible and actual_all < cfg.safety_margin_m-1e-9),
            "selected_truth_collision": bool(actual_all <= 0),
            "selected_ecp_gap_score_m": score, "ball_error_max_m": ball_error,
            "quantile_m": float(quantile_m) if quantile_m is not None else None,
            "coverage_ecp": coverage_ecp, "coverage_ball": coverage_ball,
            "coverage_unit": "one selected observed-target forecast anchor at physical ticks; not independent scene coverage or a closed-loop guarantee",
            "observed_target_count": len(tracks),
            "all_true_target_count": len(self.ids),
            "unobserved_true_target_count": len(self.ids)-len(tracks),
            "prediction_errors_by_horizon": statistics,
            "prediction_RMSE_by_horizon": {key: value["rmse_m"] for key, value in statistics.items()},
            "checked_saved_snapshots": self.checked_saved_snapshots,
            "maximum_saved_position_error_m": self.maximum_saved_position_error_m,
            "truth_used_online": False,
        }
