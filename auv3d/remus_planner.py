"""Fixed-budget reference-space DWA with original REMUS forward validation.

This retains the P25 baseline's sample / predict / collision-check / score
structure. REMUS actions are *references*, not the P25 [u,q,r] command. A cheap
surrogate ranks candidates only; every accepted trajectory is integrated by an
isolated six-DOF REMUS including autopilot memory and actuator dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from .adapters.remus_guidance import ReferenceAction, predict_reference, validate_plant_state
from .geometry import swept_sphere_clearance, wrap_pi
from .prediction import CAPredictor
from .sensors import validate_track


@dataclass(frozen=True)
class RemusPlanResult:
    action: ReferenceAction
    states: np.ndarray
    feasible: bool
    candidate_count: int
    verified_count: int
    valid_count: int
    minimum_predicted_clearance_m: float
    planning_ms: float
    prediction_fallback_count: int = 0
    emergency: bool = False
    reason: str = 'verified_reference'
    emergency_verified_safe: bool = False
    prediction_failure_count: int = 0
    fast_screen_model: str = 'reference_response_surrogate_ranking_only'
    verification_model: str = 'original_remus100_full_6dof_autopilot_actuators'
    candidate_evaluations: tuple[dict, ...] = ()
    shortlist_indices: dict[str, list[int]] = field(default_factory=dict)

    @property
    def command(self) -> np.ndarray:
        return self.action.as_array()


def candidate_references(plant, goal, cfg) -> list[ReferenceAction]:
    """Sample reachable actuator-reference envelopes, not instantaneous rates.

    Dynamic reachability is established only by the subsequent full rollout.
    Heading changes are targets for the retained rate-limited reference model.
    """
    goal = np.asarray(goal, dtype=float)
    if goal.shape != (3,) or not np.isfinite(goal).all():
        raise ValueError('goal must contain three finite NED coordinates')
    delta = goal[:2] - plant.eta[:2]
    bearing = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-9 else float(plant.eta[5])
    depth_low = cfg.body_radius_m + cfg.safety_margin_m
    depth_high = min(100., cfg.depth_limit_m) - depth_low
    actions = []
    for offset in cfg.heading_offsets_deg:
        heading = plant.eta[5] + float(wrap_pi(bearing + np.deg2rad(offset) - plant.eta[5]))
        for depth_offset in cfg.depth_offsets_m:
            depth = float(np.clip(goal[2] + depth_offset, depth_low, depth_high))
            for rpm in cfg.rpm_values:
                action = ReferenceAction(depth, heading, float(rpm))
                action.validate(plant)
                if not any(np.allclose(action.as_array(), other.as_array(), atol=1e-10, rtol=0) for other in actions):
                    actions.append(action)
    if not actions:
        raise ValueError('REMUS reference grid is empty')
    return actions


def _forecast_tracks(tracks, times, now, cfg):
    times = np.asarray(times, dtype=float)
    if not np.isfinite(now) or times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all():
        raise ValueError('obstacle forecast requires finite now and one-dimensional finite offsets')
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError('obstacle forecast offsets must be nonnegative and strictly increasing')
    predictor = CAPredictor(cfg.observation_period_s)
    forecasts = []
    for track in tracks:
        validate_track(track, now)
        observed_times = np.asarray(track.times, dtype=float)
        observed_positions = np.asarray(track.positions, dtype=float)
        positions = predictor.predict(track, now + times)
        if np.shape(positions) != (len(times), 3):
            raise ValueError('obstacle predictor must return [forecast_time,3] positions')
        if not np.isfinite(positions).all():
            raise FloatingPointError('nonfinite obstacle forecast')
        age = max(0., now - track.times[-1])
        speed = 0.
        if len(observed_times) >= 2:
            speed = np.linalg.norm(observed_positions[-1] - observed_positions[-2]) / (observed_times[-1] - observed_times[-2])
        # This disclosed deterministic margin is not a calibrated guarantee.
        margin = cfg.prediction_margin_m + age * max(1., speed)
        radius = track.radius_m + cfg.body_radius_m + margin
        if not np.isfinite([speed, margin, radius]).all():
            raise FloatingPointError('nonfinite obstacle speed or uncertainty radius')
        forecasts.append((positions, radius))
    return forecasts


def _clearance_positions(positions, terrain, forecasts, cfg):
    points = np.asarray(positions, dtype=float)
    if points.ndim not in (2, 3) or points.shape[-1] != 3 or points.shape[-2] < 2:
        raise ValueError('clearance requires at least two three-dimensional trajectory points')
    if (not np.isfinite(points).all() or not np.isfinite(terrain.slope_bound)
            or terrain.slope_bound < 0 or not np.isfinite(terrain.depths).all()):
        raise FloatingPointError('nonfinite trajectory or invalid terrain slope bound')
    # Supports either [time,3] or a candidate batch [candidate,time,3].
    terrain_clear = terrain.clearance(points, cfg.body_radius_m).min(axis=-1)
    segment_gap = np.linalg.norm(np.diff(points, axis=-2), axis=-1).max(axis=-1)
    terrain_clear -= .5 * segment_gap * np.sqrt(1. + terrain.slope_bound ** 2)
    depth_clear = np.min(cfg.depth_limit_m - cfg.body_radius_m - points[..., 2], axis=-1)
    clear = np.minimum(terrain_clear, depth_clear)
    for obstacle, radius in forecasts:
        swept = swept_sphere_clearance(points[..., :-1, :], points[..., 1:, :],
                                       obstacle[:-1], obstacle[1:], radius)
        clear = np.minimum(clear, swept.min(axis=-1))
    if not np.isfinite(clear).all():
        raise FloatingPointError('trajectory clearance must be finite')
    return clear


def trajectory_clearance(states, terrain, tracks, now, cfg) -> float:
    """Conservative terrain and swept moving-sphere clearance at physical ticks.

    The obstacle prediction is CA from causal observations. This checks a
    six-DOF vehicle trajectory against that forecast; unseen motion changes are
    handled by the real execution collision monitor, not claimed to be known.
    """
    rows = np.asarray(states, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 16 or len(rows) < 2:
        raise ValueError('REMUS trajectory requires at least two [offset,eta6,nu6,actuator3] rows')
    if not np.isfinite(rows).all():
        raise FloatingPointError('REMUS trajectory must contain finite states')
    if abs(rows[0, 0]) > 1e-12:
        raise ValueError('REMUS trajectory offsets must start at zero')
    forecasts = _forecast_tracks(tracks, rows[:, 0], now, cfg)
    return float(_clearance_positions(rows[:, 1:4], terrain, forecasts, cfg))


def _fast_rollouts(plant, actions, cfg):
    """Deliberately approximate reference-response model, used ONLY for ranking."""
    count = max(2, int(np.ceil(cfg.horizon_s / .5)))
    dt = cfg.horizon_s / count
    times = np.linspace(0., cfg.horizon_s, count + 1)
    refs = np.array([action.as_array() for action in actions])
    positions = np.repeat(plant.eta[None, :3], len(actions), axis=0)
    yaw = np.full(len(actions), plant.eta[5])
    depth_ref = np.full(len(actions), plant.vehicle.z_d)
    water_speed = np.full(len(actions), max(0., plant.relative_velocity()[0]))
    current = np.array([plant.vehicle.V_c * np.cos(plant.vehicle.beta_c),
                        plant.vehicle.V_c * np.sin(plant.vehicle.beta_c), 0.])
    paths = [positions.copy()]
    for _ in range(count):
        # Generic operating-point heuristic; not system identification or a
        # replacement for the actual reference model/actuator integration.
        yaw += np.clip(float(plant.vehicle.wn_d) * wrap_pi(refs[:, 1] - yaw),
                       -plant.vehicle.r_max, plant.vehicle.r_max) * dt
        depth_ref += (refs[:, 0] - depth_ref) * (1. - np.exp(-plant.vehicle.wn_d_z * dt))
        target_speed = 2.5 * refs[:, 2] / plant.vehicle.nMax
        water_speed += np.clip(target_speed - water_speed, -.25 * dt, .25 * dt)
        vertical = np.clip((depth_ref - positions[:, 2]) * .08, -.25 * water_speed, .25 * water_speed)
        horizontal = np.sqrt(np.maximum(0., water_speed ** 2 - vertical ** 2))
        velocity = np.column_stack([horizontal * np.cos(yaw), horizontal * np.sin(yaw), vertical]) + current
        positions = positions + dt * velocity
        paths.append(positions.copy())
    return times, np.stack(paths, axis=1)


def _diverse_shortlist(actions, fast_cost, budget):
    """Cover heading, depth and RPM groups before filling the remaining budget."""
    fast_cost = np.asarray(fast_cost, dtype=float)
    if not actions or fast_cost.shape != (len(actions),) or not np.isfinite(fast_cost).all():
        raise ValueError('shortlist requires nonempty references and matching finite costs')
    if not isinstance(budget, (int, np.integer)) or isinstance(budget, (bool, np.bool_)) or budget < 1:
        raise ValueError('shortlist budget must be a positive integer')
    order = list(np.argsort(fast_cost, kind='stable'))
    chosen = [order[0]]
    array = np.array([action.as_array() for action in actions])
    # Each heading gets a candidate; next cover shallow/deep and speed extrema.
    # If the budget is unusually small, farthest reference-space sampling below
    # preserves turn diversity instead of admitting the first few equal groups.
    groups = []
    for column in (1, 0, 2):
        values = np.unique(array[:, column])
        if column in (0, 2) and len(values) > 2:
            values = values[[0, -1]]
        for value in values:
            members = [i for i in order if abs(array[i, column] - value) < 1e-9]
            if members and members[0] not in groups:
                groups.append(members[0])
    if budget < len(np.unique(array[:, 1])):
        scale = np.array([max(np.ptp(array[:, 0]), 1.), max(np.ptp(array[:, 1]), .1), max(np.ptp(array[:, 2]), 1.)])
        normalized = array / scale
        while len(chosen) < min(budget, len(actions)):
            available = [i for i in order if i not in chosen]
            distance = np.array([np.linalg.norm(normalized[i] - normalized[chosen], axis=1).min() for i in available])
            chosen.append(available[int(np.argmax(distance))])
        return chosen
    for index in groups + order:
        if len(chosen) >= min(budget, len(actions)):
            break
        if index not in chosen:
            chosen.append(index)
    return chosen


def _rank_reference_grid(plant, goal, terrain, tracks, now, cfg, cost_function):
    validate_plant_state(plant)
    if not np.isfinite(now) or abs(plant.time_s - now) > 1e-6:
        raise ValueError('planner now must match actual REMUS simulation time')
    actions = candidate_references(plant, goal, cfg)
    fast_times, fast_paths = _fast_rollouts(plant, actions, cfg)
    fast_forecasts = _forecast_tracks(tracks, fast_times, now, cfg)
    fast_clear = _clearance_positions(fast_paths, terrain, fast_forecasts, cfg)
    fast_cost = np.array([cost_function(path[-1], action, float(clear), plant, goal)
                          for path, action, clear in zip(fast_paths, actions, fast_clear)])
    # Preserve the original baseline ranking and its disclosed heuristic penalty.
    # A fast prediction never rejects or accepts a six-DOF reference as safe.
    fast_cost += 100. * np.maximum(0., cfg.safety_margin_m - fast_clear)
    if not np.isfinite(fast_cost).all():
        raise FloatingPointError('reference ranking costs must be finite')
    budgets = dict.fromkeys((9, 18, 45, cfg.candidate_budget))
    shortlists = {str(budget): [int(index) for index in _diverse_shortlist(actions, fast_cost, budget)]
                  for budget in budgets}
    return actions, fast_clear, fast_cost, shortlists


def _evaluate_reference(index, action, plant, goal, terrain, tracks, now, cfg,
                        cost_function, fast_clear, fast_cost, shortlists):
    """One isolated physical rollout, shared by normal planning and diagnostics."""
    row = {
        'index': int(index), 'reference': action.as_array().tolist(),
        'fast_clearance_m': float(fast_clear[index]), 'fast_cost': float(fast_cost[index]),
        'feasible': False, 'minimum_predicted_clearance_m': None, 'cost': None,
        'prediction_failure': None, 'endpoint_ned_m': None,
        'in_shortlist': {budget: int(index) in members for budget, members in shortlists.items()},
    }
    try:
        prediction = predict_reference(plant, action, cfg.horizon_s, cfg.physics_dt, record_controls=False)
        clear = trajectory_clearance(prediction.states, terrain, tracks, now, cfg)
        cost = cost_function(prediction.positions[-1], action, clear, plant, goal)
        if not np.isfinite(cost):
            raise FloatingPointError('full six-DOF reference cost must be finite')
        row.update(feasible=bool(clear >= cfg.safety_margin_m),
                   minimum_predicted_clearance_m=float(clear), cost=float(cost),
                   endpoint_ned_m=prediction.positions[-1].tolist())
        return prediction, row
    except (FloatingPointError, OverflowError) as error:
        # Numeric failures reject this candidate; malformed causal inputs were
        # already rejected before candidate integration and are never hidden.
        row['prediction_failure'] = f'{type(error).__name__}: {error}'
        return None, row


def evaluate_reference_grid(plant, goal, terrain, tracks, now, cfg, previous_action=None):
    """Diagnose all reference combinations using only the supplied causal tracks.

    This offline API does not command a vehicle and does not read obstacle truth.
    Its 9/18/45 lists use exactly the default planner's fast ranking/diversity
    policy. Supply the action selected at the *previous planning decision* to
    reproduce its smoothness cost. With command delay this can differ from the
    reference currently applied to the real plant. Omitting it explicitly means
    zero history cost.
    """
    tic = time.perf_counter()
    planner = RemusDWA(cfg)
    if previous_action is not None:
        previous_action.validate(plant)
    planner.previous = previous_action
    goal = np.asarray(goal, dtype=float)
    actions, fast_clear, fast_cost, shortlists = _rank_reference_grid(
        plant, goal, terrain, tracks, now, cfg, planner._cost)
    candidates = []
    for index, action in enumerate(actions):
        _, row = _evaluate_reference(index, action, plant, goal, terrain, tracks, now, cfg,
                                    planner._cost, fast_clear, fast_cost, shortlists)
        candidates.append(row)
    feasible_counts, best_indices, best_costs, coverage = {}, {}, {}, {}
    for budget, members in shortlists.items():
        valid = [index for index in members if candidates[index]['feasible']]
        best = min(valid, key=lambda index: candidates[index]['cost']) if valid else None
        feasible_counts[budget] = len(valid)
        best_indices[budget] = best
        best_costs[budget] = candidates[best]['cost'] if best is not None else None
        references = np.array([actions[index].as_array() for index in members])
        coverage[budget] = {'reference_count': len(members),
                            'depth_value_count': len(np.unique(references[:, 0])),
                            'heading_value_count': len(np.unique(references[:, 1])),
                            'rpm_value_count': len(np.unique(references[:, 2]))}
    feasible_count = sum(row['feasible'] for row in candidates)
    if not feasible_count:
        reason = 'no_feasible_reference_in_full_grid'
    elif not feasible_counts['9']:
        reason = 'full_grid_feasible_but_budget9_missed'
    else:
        reason = 'budget9_contains_verified_feasible_reference'
    return {
        'now_s': float(now), 'horizon_s': cfg.horizon_s, 'physics_dt_s': cfg.physics_dt,
        'candidate_count': len(actions), 'feasible_count': feasible_count,
        'prediction_failure_count': sum(row['prediction_failure'] is not None for row in candidates),
        'evaluation_ms': (time.perf_counter() - tic) * 1e3,
        'previous_reference': previous_action.as_array().tolist() if previous_action is not None else None,
        'cost_history': 'supplied_previous_reference' if previous_action is not None else 'no_previous_reference',
        'shortlist_policy': 'baseline_diverse_reference_groups_then_fast_cost',
        'shortlist_indices': shortlists, 'shortlist_feasible_counts': feasible_counts,
        'shortlist_best_feasible_indices': best_indices, 'shortlist_best_feasible_cost': best_costs,
        'shortlist_coverage': coverage, 'best_feasible_index': best_indices['45'],
        'reason': reason, 'candidates': candidates,
        'validation_backend': 'original_remus100_full_6dof_autopilot_actuators',
        'obstacle_information': 'causal tracks only; no future truth',
    }


class RemusDWA:
    """Reference-space DWA baseline; bounded candidate budget plus one fallback.

    The CA obstacle baseline and deterministic margin are explicit limitations.
    No candidate is considered safe because its cheap rollout passed a test.
    """

    def __init__(self, cfg):
        cfg.validate()
        self.cfg = cfg
        self.previous: ReferenceAction | None = None

    def reset(self):
        self.previous = None

    def _cost(self, endpoint, action, clearance, plant, goal):
        cfg = self.cfg
        travel = max(5., np.linalg.norm(plant.ground_velocity()) * cfg.horizon_s)
        goal_cost = np.linalg.norm(endpoint - goal) / travel
        # A smooth bounded penalty starts responding before the hard margin.
        clearance_cost = np.exp(-max(clearance, 0.) / travel)
        smooth = 0.
        if self.previous is not None:
            difference = action.as_array() - self.previous.as_array()
            difference[1] = wrap_pi(difference[1])
            scale = np.array([max(1., np.ptp(cfg.depth_offsets_m)), np.deg2rad(90), plant.vehicle.nMax])
            smooth = np.sum((difference / scale) ** 2)
        return (cfg.goal_weight * goal_cost + cfg.clearance_weight * clearance_cost
                - cfg.speed_weight * action.rpm / plant.vehicle.nMax + cfg.smooth_weight * smooth)

    def plan(self, plant, goal, terrain, tracks, now) -> RemusPlanResult:
        tic = time.perf_counter()
        cfg = self.cfg
        goal = np.asarray(goal, dtype=float)
        actions, fast_clear, fast_cost, shortlists = _rank_reference_grid(
            plant, goal, terrain, tracks, now, cfg, self._cost)
        shortlist = shortlists[str(cfg.candidate_budget)]
        valid = []
        failures = 0
        evaluations = []
        for index in shortlist:
            action = actions[index]
            prediction, row = _evaluate_reference(
                index, action, plant, goal, terrain, tracks, now, cfg,
                self._cost, fast_clear, fast_cost, shortlists)
            evaluations.append(row)
            if row['prediction_failure'] is not None:
                failures += 1
                continue
            if row['feasible']:
                valid.append((row['cost'], action, prediction.states, row['minimum_predicted_clearance_m']))
        if valid:
            _, action, states, clear = min(valid, key=lambda row: row[0])
            feasible = True
            emergency = False
            emergency_safe = False
            reason = 'verified_reference'
        else:
            # Cut propulsion while requesting the current depth/heading. Real
            # speed and actuator memory remain intact: this is coasting/drift,
            # not an instantaneous stop and not a guaranteed safe backup.
            action = ReferenceAction(float(np.clip(plant.eta[2], 0., 100.)), float(plant.eta[5]), cfg.emergency_rpm)
            prediction = predict_reference(plant, action, cfg.horizon_s, cfg.physics_dt, record_controls=False)
            states = prediction.states
            clear = trajectory_clearance(states, terrain, tracks, now, cfg)
            feasible = False
            emergency = True
            emergency_safe = clear >= cfg.safety_margin_m
            reason = ('no_verified_feasible_reference_emergency_coast_forecast_clear'
                      if emergency_safe else 'no_verified_feasible_reference_emergency_not_safe')
        self.previous = action
        return RemusPlanResult(
            action=action, states=states, feasible=feasible, candidate_count=len(actions),
            verified_count=len(shortlist), valid_count=len(valid),
            minimum_predicted_clearance_m=float(clear),
            planning_ms=(time.perf_counter() - tic) * 1e3,
            emergency=emergency, reason=reason, emergency_verified_safe=bool(emergency_safe),
            prediction_failure_count=failures, candidate_evaluations=tuple(evaluations),
            shortlist_indices=shortlists,
        )
