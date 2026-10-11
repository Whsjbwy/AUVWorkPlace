"""Explicit reference-grid and emergency-action engineering ablations.

These ordinary sampling policies are not a new avoidance algorithm. They use
only the current vehicle, delivered tracks (in the planner) and the navigation
goal. A reference is not physically feasible until the original REMUS rollout
has checked it. No source-model coefficient or actuator limit is changed here.
"""
from __future__ import annotations

import numpy as np

from .adapters.remus_guidance import ReferenceAction
from .geometry import wrap_pi


CANDIDATE_SCHEMES = ('baseline45', 'expanded')
SHORTLIST_POLICIES = ('diverse', 'ranked', 'coverage')
EMERGENCY_POLICIES = ('coast', 'verified_backup')


def _append_unique(actions, action, plant):
    action.validate(plant)
    if not any(np.allclose(action.as_array(), other.as_array(), atol=1e-10, rtol=0.)
               for other in actions):
        actions.append(action)


def expanded_references(plant, goal, cfg, baseline):
    """Retain the entire baseline prefix, then add at most 24 references.

    Additional targets use present/intermediate/goal depth, current heading or
    goal bearing +/-90 degrees, and 600/nominal RPM. These are actuator-reference
    envelopes, not a claim of instantaneous rate or speed attainability.
    """
    actions = list(baseline)
    goal = np.asarray(goal, dtype=float)
    delta = goal[:2] - plant.eta[:2]
    bearing = (float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-9
               else float(plant.eta[5]))
    low = cfg.body_radius_m + cfg.safety_margin_m
    high = min(100., cfg.depth_limit_m) - low
    depths = (plant.eta[2], .5 * (plant.eta[2] + goal[2]), goal[2])
    headings = (float(plant.eta[5]), bearing, bearing - np.pi / 2., bearing + np.pi / 2.)
    rpms = (min(600., plant.vehicle.nMax), min(cfg.nominal_rpm, plant.vehicle.nMax))
    for heading in headings:
        unwrapped = plant.eta[5] + float(wrap_pi(heading - plant.eta[5]))
        for depth in depths:
            for rpm in rpms:
                _append_unique(actions, ReferenceAction(float(np.clip(depth, low, high)),
                               float(unwrapped), float(rpm)), plant)
    return actions


def ranked_shortlist(actions, fast_cost, budget):
    """Use surrogate rank alone, without implying physical safety."""
    costs = np.asarray(fast_cost, dtype=float)
    _validate_shortlist(actions, costs, budget)
    return [int(index) for index in np.argsort(costs, kind='stable')[:budget]]


def coverage_shortlist(actions, fast_cost, budget):
    """Greedy reference-space coverage, anchored at the best surrogate score.

    Heading distances are wrapped. Depth and RPM distances use grid ranges.
    Fast-score order breaks exact coverage ties reproducibly.
    """
    costs = np.asarray(fast_cost, dtype=float)
    _validate_shortlist(actions, costs, budget)
    order = [int(index) for index in np.argsort(costs, kind='stable')]
    refs = np.array([action.as_array() for action in actions])
    scales = np.array([max(np.ptp(refs[:, 0]), 1.), np.pi,
                       max(np.ptp(refs[:, 2]), 1.)])
    chosen = [order[0]]
    while len(chosen) < min(budget, len(actions)):
        available = [index for index in order if index not in chosen]
        differences = refs[available, None, :] - refs[None, chosen, :]
        differences[..., 1] = wrap_pi(differences[..., 1])
        distances = np.linalg.norm(differences / scales, axis=-1).min(axis=1)
        chosen.append(available[int(np.argmax(distances))])
    return chosen


def _validate_shortlist(actions, costs, budget):
    if not actions or costs.shape != (len(actions),) or not np.isfinite(costs).all():
        raise ValueError('shortlist requires nonempty references and matching finite costs')
    if not isinstance(budget, (int, np.integer)) or isinstance(budget, (bool, np.bool_)) or budget < 1:
        raise ValueError('shortlist budget must be a positive integer')


def coast_reference(plant, cfg):
    """The unchanged stage2 RPM0/default emergency request; no stop guarantee."""
    return ReferenceAction(float(np.clip(plant.eta[2], 0., 100.)),
                           float(plant.eta[5]), cfg.emergency_rpm)


def backup_references(plant, goal, cfg):
    """At most eight disclosed backup references, beginning with original coast.

    No backup is accepted by this generator. The production planner integrates
    each using all original controller, reference-model and actuator states and
    applies the exact same obstacle/terrain clearance criterion as candidates.
    If all backups are unsafe, finite original coast is retained and disclosed.
    """
    goal = np.asarray(goal, dtype=float)
    delta = goal[:2] - plant.eta[:2]
    bearing = (float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-9
               else float(plant.eta[5]))
    yaw = float(plant.eta[5])
    low = cfg.body_radius_m + cfg.safety_margin_m
    high = min(100., cfg.depth_limit_m) - low
    depth = float(np.clip(plant.eta[2], low, high))
    rpm = float(min(600., plant.vehicle.nMax))
    definitions = [
        ('original_coast', coast_reference(plant, cfg)),
        ('low_rpm_current_heading', ReferenceAction(depth, yaw, rpm)),
        ('low_rpm_goal_bearing', ReferenceAction(depth, yaw + float(wrap_pi(bearing - yaw)), rpm)),
        ('low_rpm_turn_left60', ReferenceAction(depth, yaw - np.pi / 3., rpm)),
        ('low_rpm_turn_right60', ReferenceAction(depth, yaw + np.pi / 3., rpm)),
        ('coast_turn_left60', ReferenceAction(depth, yaw - np.pi / 3., cfg.emergency_rpm)),
        ('coast_turn_right60', ReferenceAction(depth, yaw + np.pi / 3., cfg.emergency_rpm)),
        ('low_rpm_shallower4', ReferenceAction(float(np.clip(depth - 4., low, high)), yaw, rpm)),
    ]
    result = []
    for label, action in definitions:
        action.validate(plant)
        if not any(np.allclose(action.as_array(), old.as_array(), atol=1e-10, rtol=0.)
                   for _, old in result):
            result.append((label, action))
    return result
