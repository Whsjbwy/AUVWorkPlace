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
from .stage3_forecasting import ForecastContext
from .sensors import validate_track
from .stage3_candidates import (backup_references, coast_reference, coverage_shortlist,
                                 expanded_references, ranked_shortlist)


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
    candidate_scheme: str = 'baseline45'
    shortlist_policy: str = 'diverse'
    emergency_policy: str = 'coast'
    backup_evaluations: tuple[dict, ...] = ()
    backup_verified_count: int = 0
    backup_safe_count: int = 0
    selected_backup_index: int | None = None
    planning_goal: tuple[float, float, float] | None = None
    forecast_records: tuple[dict, ...] = ()
    risk_records: tuple[dict, ...] = ()

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
    scheme = getattr(cfg, 'candidate_scheme', 'baseline45')
    if scheme == 'expanded':
        actions = expanded_references(plant, goal, cfg, actions)
    elif scheme != 'baseline45':
        raise ValueError(f'unknown REMUS candidate_scheme: {scheme}')
    return actions


def _forecast_tracks(tracks, times, now, cfg, forecast_context=None):
    times = np.asarray(times, dtype=float)
    if not np.isfinite(now) or times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all():
        raise ValueError('obstacle forecast requires finite now and one-dimensional finite offsets')
    if np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError('obstacle forecast offsets must be nonnegative and strictly increasing')
    context = forecast_context or ForecastContext(cfg)
    forecasts = []
    for track in tracks:
        forecast = context.forecast(track, now + times, now)
        radii = forecast.enclosing_radii_m
        # Scalar compatibility preserves existing stage2 offline consumers.
        radius = float(radii[0]) if np.all(radii == radii[0]) else radii
        forecasts.append((forecast.positions, radius))
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
        radius = np.asarray(radius, dtype=float)
        if not np.isfinite(radius).all() or np.any(radius <= 0):
            raise ValueError('forecast radius must be finite and positive')
        if radius.ndim:
            if radius.shape != (points.shape[-2],):
                raise ValueError('time-varying obstacle radius must match trajectory samples')
            # The entire segment uses the larger endpoint radius.
            radius = np.maximum(radius[:-1], radius[1:])
        swept = swept_sphere_clearance(points[..., :-1, :], points[..., 1:, :],
                                       obstacle[:-1], obstacle[1:], radius)
        clear = np.minimum(clear, swept.min(axis=-1))
    if not np.isfinite(clear).all():
        raise FloatingPointError('trajectory clearance must be finite')
    return clear


def trajectory_clearance(states, terrain, tracks, now, cfg, forecast_context=None) -> float:
    """Conservative terrain and swept moving-sphere clearance at physical ticks.

    The configured obstacle predictor receives only causal observations. This
    checks a six-DOF vehicle trajectory against that forecast; unseen changes are
    handled by the real execution collision monitor, not claimed to be known.
    """
    rows = np.asarray(states, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 16 or len(rows) < 2:
        raise ValueError('REMUS trajectory requires at least two [offset,eta6,nu6,actuator3] rows')
    if not np.isfinite(rows).all():
        raise FloatingPointError('REMUS trajectory must contain finite states')
    if abs(rows[0, 0]) > 1e-12:
        raise ValueError('REMUS trajectory offsets must start at zero')
    forecasts = _forecast_tracks(tracks, rows[:, 0], now, cfg, forecast_context)
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


def _rank_reference_grid(plant, goal, terrain, tracks, now, cfg, cost_function, forecast_context=None):
    validate_plant_state(plant)
    if not np.isfinite(now) or abs(plant.time_s - now) > 1e-6:
        raise ValueError('planner now must match actual REMUS simulation time')
    actions = candidate_references(plant, goal, cfg)
    fast_times, fast_paths = _fast_rollouts(plant, actions, cfg)
    fast_forecasts = _forecast_tracks(tracks, fast_times, now, cfg, forecast_context)
    fast_clear = _clearance_positions(fast_paths, terrain, fast_forecasts, cfg)
    fast_cost = np.array([cost_function(path[-1], action, float(clear), plant, goal)
                          for path, action, clear in zip(fast_paths, actions, fast_clear)])
    # Preserve the original baseline ranking and its disclosed heuristic penalty.
    # A fast prediction never rejects or accepts a six-DOF reference as safe.
    fast_cost += 100. * np.maximum(0., cfg.safety_margin_m - fast_clear)
    if not np.isfinite(fast_cost).all():
        raise FloatingPointError('reference ranking costs must be finite')
    budgets = dict.fromkeys((9, 18, 45, cfg.candidate_budget))
    policy = getattr(cfg, 'shortlist_policy', 'diverse')
    selector = _shortlist_selector(policy)
    shortlists = {str(budget): [int(index) for index in selector(actions, fast_cost, budget)]
                  for budget in budgets}
    return actions, fast_clear, fast_cost, shortlists


def _shortlist_selector(policy):
    selectors = {'diverse': _diverse_shortlist, 'ranked': ranked_shortlist,
                 'coverage': coverage_shortlist}
    if policy not in selectors:
        raise ValueError(f'unknown REMUS shortlist_policy: {policy}')
    return selectors[policy]


def _evaluate_reference(index, action, plant, goal, terrain, tracks, now, cfg,
                        cost_function, fast_clear, fast_cost, shortlists, forecast_context=None):
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
        clear = trajectory_clearance(prediction.states, terrain, tracks, now, cfg, forecast_context)
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


def evaluate_reference_grid(plant, goal, terrain, tracks, now, cfg, previous_action=None, *,
                            forecaster=None, uncertainty_by_id=None):
    """Diagnose all reference combinations using only the supplied causal tracks.

    This offline API does not command a vehicle and does not read obstacle truth.
    Its 9/18/45 lists use exactly the configured production shortlist policy.
    Other policies are diagnosed at this same real state without additional
    physical rollouts. Supply the action selected at the *previous decision* to
    reproduce its smoothness cost. With command delay this can differ from the
    reference currently applied to the real plant. Omitting it explicitly means
    zero history cost.
    """
    tic = time.perf_counter()
    planner = RemusDWA(cfg, forecaster=forecaster)
    planner.forecasts.begin(uncertainty_by_id)
    if previous_action is not None:
        previous_action.validate(plant)
    planner.previous = previous_action
    goal = np.asarray(goal, dtype=float)
    actions, fast_clear, fast_cost, shortlists = _rank_reference_grid(
        plant, goal, terrain, tracks, now, cfg, planner._cost, planner.forecasts)
    candidates = []
    for index, action in enumerate(actions):
        _, row = _evaluate_reference(index, action, plant, goal, terrain, tracks, now, cfg,
                                    planner._cost, fast_clear, fast_cost, shortlists, planner.forecasts)
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
    policy_indices, policy_feasible_counts, policy_misses = {}, {}, {}
    for policy in ('diverse', 'ranked', 'coverage'):
        selector = _shortlist_selector(policy)
        policy_indices[policy], policy_feasible_counts[policy], policy_misses[policy] = {}, {}, {}
        for budget in shortlists:
            members = [int(index) for index in selector(actions, fast_cost, int(budget))]
            count = sum(candidates[index]['feasible'] for index in members)
            policy_indices[policy][budget] = members
            policy_feasible_counts[policy][budget] = count
            policy_misses[policy][budget] = bool(feasible_count > 0 and count == 0)
    all_valid = [row for row in candidates if row['feasible']]
    # For the original <=45 grid retain historical tie ordering. For an
    # expanded grid, budget45 is a shortlist and must not be called full-grid.
    full_best = (best_indices['45'] if len(actions) <= 45 else
                 (min(all_valid, key=lambda row: row['cost'])['index'] if all_valid else None))
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
        'candidate_scheme': getattr(cfg, 'candidate_scheme', 'baseline45'),
        'configured_shortlist_policy': getattr(cfg, 'shortlist_policy', 'diverse'),
        'shortlist_policy': ('baseline_diverse_reference_groups_then_fast_cost'
                             if getattr(cfg, 'shortlist_policy', 'diverse') == 'diverse'
                             else getattr(cfg, 'shortlist_policy')),
        'shortlist_indices': shortlists, 'shortlist_feasible_counts': feasible_counts,
        'shortlist_best_feasible_indices': best_indices, 'shortlist_best_feasible_cost': best_costs,
        'shortlist_coverage': coverage, 'best_feasible_index': full_best,
        'policy_shortlist_indices': policy_indices,
        'policy_shortlist_feasible_counts': policy_feasible_counts,
        'policy_shortlist_missed_safe_reference': policy_misses,
        'planning_goal': goal.tolist(),
        'reason': reason, 'candidates': candidates,
        'validation_backend': 'original_remus100_full_6dof_autopilot_actuators',
        'obstacle_information': 'causal tracks only; no future truth',
    }


class RemusDWA:
    """Reference-space DWA with a bounded full-plant validation budget.

    CA is the default pluggable obstacle forecast; deterministic margins remain
    an explicit limitation. Emergency validation uses one coast reference or
    at most eight configured backup references, without assuming instant stop.
    No candidate is considered safe because its cheap rollout passed a test.
    """

    def __init__(self, cfg, forecaster=None, risk_policy=None):
        cfg.validate()
        self.cfg = cfg
        self.forecasts = ForecastContext(cfg, forecaster)
        self.memory_uncertainty = {}
        self.previous: ReferenceAction | None = None
        self.risk_policy = risk_policy

    def _risk_clearance(self, states, terrain, tracks, now, base_clear):
        if self.risk_policy is None:
            return base_clear, None
        return self.risk_policy.adjust(states, terrain, tracks, now, self.cfg,
                                       self.forecasts, base_clear)

    def set_memory_uncertainty(self, uncertainty_by_id):
        self.memory_uncertainty = dict(uncertainty_by_id)

    def reset(self):
        self.previous = None
        self.forecasts.begin()
        self.memory_uncertainty = {}

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
        self.forecasts.begin(self.memory_uncertainty)
        goal = np.asarray(goal, dtype=float)
        actions, fast_clear, fast_cost, shortlists = _rank_reference_grid(
            plant, goal, terrain, tracks, now, cfg, self._cost, self.forecasts)
        shortlist = shortlists[str(cfg.candidate_budget)]
        valid = []
        failures = 0
        evaluations = []
        backup_evaluations = []
        selected_backup_index = None
        risk_records = []
        for index in shortlist:
            action = actions[index]
            prediction, row = _evaluate_reference(
                index, action, plant, goal, terrain, tracks, now, cfg,
                self._cost, fast_clear, fast_cost, shortlists, self.forecasts)
            evaluations.append(row)
            if row['prediction_failure'] is not None:
                failures += 1
                continue
            clear, risk = self._risk_clearance(prediction.states, terrain, tracks, now,
                                               row['minimum_predicted_clearance_m'])
            if risk is not None:
                risk_records.append(dict(risk, candidate_index=int(index), action_kind='candidate'))
                row['risk'] = risk
                row['minimum_predicted_clearance_m'] = float(clear)
                row['cost'] = float(self._cost(prediction.positions[-1], action, clear, plant, goal))
                row['feasible'] = bool(clear >= cfg.safety_margin_m)
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
            policy = getattr(cfg, 'emergency_policy', 'coast')
            if policy == 'coast':
                action = coast_reference(plant, cfg)
                prediction = predict_reference(plant, action, cfg.horizon_s, cfg.physics_dt, record_controls=False)
                states = prediction.states
                clear = trajectory_clearance(states, terrain, tracks, now, cfg, self.forecasts)
                clear, risk = self._risk_clearance(states, terrain, tracks, now, clear)
                if risk is not None:
                    risk_records.append(dict(risk, candidate_index=0, action_kind='emergency_coast'))
            elif policy == 'verified_backup':
                options = []
                for index, (label, backup) in enumerate(backup_references(plant, goal, cfg)):
                    row = {'index': index, 'label': label,
                           'reference': backup.as_array().tolist(), 'feasible': False,
                           'minimum_predicted_clearance_m': None, 'cost': None,
                           'prediction_failure': None}
                    try:
                        candidate = predict_reference(plant, backup, cfg.horizon_s, cfg.physics_dt,
                                                      record_controls=False)
                        gap = trajectory_clearance(candidate.states, terrain, tracks, now, cfg, self.forecasts)
                        gap, risk = self._risk_clearance(candidate.states, terrain, tracks, now, gap)
                        if risk is not None:
                            risk_records.append(dict(risk, candidate_index=index, action_kind='backup'))
                            row['risk'] = risk
                        cost = self._cost(candidate.positions[-1], backup, gap, plant, goal)
                        if not np.isfinite(cost):
                            raise FloatingPointError('six-DOF backup cost must be finite')
                        row.update(feasible=bool(gap >= cfg.safety_margin_m),
                                   minimum_predicted_clearance_m=float(gap), cost=float(cost))
                        options.append((index, backup, candidate.states, float(gap), float(cost)))
                    except (FloatingPointError, OverflowError) as error:
                        failures += 1
                        row['prediction_failure'] = f'{type(error).__name__}: {error}'
                    backup_evaluations.append(row)
                safe = [option for option in options if option[3] >= cfg.safety_margin_m]
                if safe:
                    selected = min(safe, key=lambda option: option[4])
                else:
                    # Prefer the unchanged coast when it has a finite rollout.
                    # A numeric coast failure cannot become a fabricated stop.
                    finite_coast = [option for option in options if option[0] == 0]
                    if finite_coast:
                        selected = finite_coast[0]
                    elif options:
                        selected = max(options, key=lambda option: option[3])
                    else:
                        raise FloatingPointError('all REMUS backup rollouts failed numerically; no executable forecast')
                selected_backup_index, action, states, clear, _ = selected
            else:
                raise ValueError(f'unknown REMUS emergency_policy: {policy}')
            feasible = False
            emergency = True
            emergency_safe = clear >= cfg.safety_margin_m
            reason = ('no_verified_feasible_reference_emergency_coast_forecast_clear'
                      if emergency_safe else 'no_verified_feasible_reference_emergency_not_safe')
            if policy == 'verified_backup':
                reason = ('no_verified_feasible_reference_verified_backup_forecast_clear'
                          if emergency_safe else 'no_verified_feasible_reference_no_safe_backup')
        self.previous = action
        return RemusPlanResult(
            action=action, states=states, feasible=feasible, candidate_count=len(actions),
            verified_count=len(shortlist), valid_count=len(valid),
            minimum_predicted_clearance_m=float(clear),
            planning_ms=(time.perf_counter() - tic) * 1e3,
            emergency=emergency, reason=reason, emergency_verified_safe=bool(emergency_safe),
            prediction_failure_count=failures, candidate_evaluations=tuple(evaluations),
            shortlist_indices=shortlists,
            candidate_scheme=getattr(cfg, 'candidate_scheme', 'baseline45'),
            shortlist_policy=getattr(cfg, 'shortlist_policy', 'diverse'),
            emergency_policy=getattr(cfg, 'emergency_policy', 'coast'),
            backup_evaluations=tuple(backup_evaluations),
            backup_verified_count=len(backup_evaluations),
            backup_safe_count=sum(row['feasible'] for row in backup_evaluations),
            selected_backup_index=selected_backup_index,
            planning_goal=tuple(float(value) for value in goal),
            forecast_records=tuple(self.forecasts.records),
            risk_records=tuple(risk_records),
        )
