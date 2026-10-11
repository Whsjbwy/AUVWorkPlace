"""Causal perception-memory ablations for the shared six-DOF experiment.

Only already delivered :class:`Observation` objects enter this module.  A lost
track remains a history of measurements; retention never appends synthetic
positions.  The extra radius is an explicitly configured heuristic, not a
calibrated probability or a certified reachable-set bound.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral, Real

import numpy as np

from .sensors import Observation, Track, validate_track


POLICIES = ("hard_expiry", "retain_uncertain", "conditional")
_TIME_TOL = 1e-9


def _finite_number(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _position(value, name: str) -> np.ndarray:
    try:
        position = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite position of shape (3,)") from exc
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"{name} must be a finite position of shape (3,)")
    return position.copy()


class CausalTrackMemory:
    """Explicit hard expiry, uncertain retention, or conditional removal.

    ``conditional`` removes only lost objects for which disjoint conservative
    reachable balls exclude interaction over ``horizon_s + control_dt +
    command_delay_s``.  Their centers are the last delivered measurement and
    the supplied *actual* current ego position, without planned/true future
    positions.  The rule is conditional on the configured obstacle and ego
    speed bounds holding; those bounds are declared assumptions, not vehicle
    certification.  Missing ego information therefore prevents removal.

    For valid input, ``hard_expiry`` uses the same age comparison, 32-sample
    history and insertion ordering as the Stage2 :class:`TrackMemory`.
    """

    def __init__(self, max_age_s: float = 4.0, *, policy: str = "hard_expiry",
                 expected_period_s: float = 1.0, observation_delay_s: float = 0.0,
                 uncertainty_growth_mps: float = 2.5,
                 obstacle_speed_bound_mps: float = 4.0,
                 ego_speed_bound_mps: float = 4.0,
                 horizon_s: float = 24.0, control_dt: float = 2.0,
                 command_delay_s: float = 0.0,
                 interaction_distance_m: float = 30.0,
                 body_radius_m: float = float(np.hypot(0.8, 0.095)),
                 safety_margin_m: float = 1.0,
                 conditional_measurements_bounded: bool = True):
        if policy not in POLICIES:
            raise ValueError(f"memory_policy must be one of {POLICIES}")
        self.policy = policy
        self.max_age_s = _finite_number("max_age_s", max_age_s, positive=True)
        self.expected_period_s = _finite_number("expected_period_s", expected_period_s, positive=True)
        self.observation_delay_s = _finite_number("observation_delay_s", observation_delay_s)
        self.uncertainty_growth_mps = _finite_number("uncertainty_growth_mps", uncertainty_growth_mps)
        self.obstacle_speed_bound_mps = _finite_number("obstacle_speed_bound_mps", obstacle_speed_bound_mps, positive=True)
        self.ego_speed_bound_mps = _finite_number("ego_speed_bound_mps", ego_speed_bound_mps, positive=True)
        self.horizon_s = _finite_number("horizon_s", horizon_s, positive=True)
        self.control_dt = _finite_number("control_dt", control_dt, positive=True)
        self.command_delay_s = _finite_number("command_delay_s", command_delay_s)
        self.interaction_distance_m = _finite_number("interaction_distance_m", interaction_distance_m)
        self.body_radius_m = _finite_number("body_radius_m", body_radius_m, positive=True)
        self.safety_margin_m = _finite_number("safety_margin_m", safety_margin_m)
        if not isinstance(conditional_measurements_bounded, bool):
            raise ValueError("conditional_measurements_bounded must be bool")
        self.conditional_measurements_bounded = conditional_measurements_bounded
        self._tracks: dict[int, Track] = {}
        self._known_ids: set[int] = set()
        self._last_observed_times: dict[int, float] = {}
        self._ever_lost_ids: set[int] = set()
        self._lost_state: dict[int, bool] = {}
        self._last_update_time: float | None = None
        self._events: list[dict] = []
        self._counts = dict(added=0, lost=0, reacquired=0, hard_removed=0,
                            conditional_removed=0, delivered_observations=0)

    @property
    def freshness_age_s(self) -> float:
        return self.expected_period_s + self.observation_delay_s

    @property
    def conditional_horizon_s(self) -> float:
        return self.horizon_s + self.control_dt + self.command_delay_s

    def _time(self, now: float) -> float:
        now = _finite_number("now", now)
        if self._last_update_time is not None and now < self._last_update_time:
            raise ValueError("memory time must not go backwards")
        return now

    def _event(self, kind: str, object_id: int, now: float, track: Track,
               **details) -> None:
        self._events.append(dict(event=kind, object_id=int(object_id), time_s=now,
                                 last_observed_time_s=float(track.times[-1]),
                                 observation_age_s=now - float(track.times[-1]),
                                 **details))

    def _lost(self, track: Track, now: float) -> bool:
        return now - track.times[-1] > self.freshness_age_s + _TIME_TOL

    def _conditional_certificate(self, track: Track, now: float,
                                 ego_position: np.ndarray) -> dict:
        age = now - track.times[-1]
        future = self.conditional_horizon_s
        observed_distance = float(np.linalg.norm(track.positions[-1] - ego_position))
        obstacle_reach = self.obstacle_speed_bound_mps * (age + future)
        ego_reach = self.ego_speed_bound_mps * future
        required_separation = (self.interaction_distance_m + self.body_radius_m
                               + track.radius_m + self.safety_margin_m)
        excluded_gap = observed_distance - obstacle_reach - ego_reach - required_separation
        return dict(last_observed_distance_m=observed_distance,
                    obstacle_reachable_radius_m=obstacle_reach,
                    ego_reachable_radius_m=ego_reach,
                    interaction_envelope_m=required_separation,
                    excluded_interaction_gap_m=excluded_gap,
                    conditional_horizon_s=future,
                    obstacle_speed_bound_mps=self.obstacle_speed_bound_mps,
                    ego_speed_bound_mps=self.ego_speed_bound_mps,
                    bound_status="declared_assumption_not_certified")

    def update(self, delivered: list[Observation], now: float,
               ego_position=None) -> None:
        """Atomically validate and apply already delivered measurements.

        Acquisition times remain unchanged.  Invalid/future/duplicate or
        reversed observations reject the entire batch without a partial write.
        Empty updates advance loss/expiry bookkeeping and use actual ego pose
        for the optional conditional rule.
        """
        now = self._time(now)
        ego = None if ego_position is None else _position(ego_position, "ego_position")
        staged = dict(self._tracks)
        last_observed_times = dict(self._last_observed_times)
        observed_ids = set()
        count = 0
        for observation in delivered:
            if not isinstance(observation, Observation):
                raise ValueError("delivered entries must be Observation objects")
            if not isinstance(observation.object_id, Integral) or isinstance(observation.object_id, bool):
                raise ValueError("observation object_id must be an integer")
            object_id = int(observation.object_id)
            stamp = _finite_number("observation time_s", observation.time_s)
            if stamp > now:
                raise ValueError("future observation leak")
            radius = _finite_number("observation radius_m", observation.radius_m, positive=True)
            position = _position(observation.position, "observation position")
            if object_id in last_observed_times and stamp <= last_observed_times[object_id]:
                raise ValueError("observations must be strictly time ordered, including removed tracks")
            if object_id not in staged:
                staged[object_id] = Track(object_id, radius)
            elif object_id not in observed_ids:
                staged[object_id] = deepcopy(staged[object_id])
            staged[object_id].append(Observation(object_id, stamp, position, radius))
            last_observed_times[object_id] = stamp
            observed_ids.add(object_id)
            count += 1
        for track in staged.values():
            validate_track(track, now)
        self._tracks = staged
        self._last_observed_times = last_observed_times
        self._last_update_time = now
        self._counts["delivered_observations"] += count
        for object_id in observed_ids:
            track = self._tracks[object_id]
            if object_id not in self._known_ids:
                self._known_ids.add(object_id)
                self._counts["added"] += 1
                self._event("added", object_id, now, track)
            elif self._lost_state.get(object_id, False) and not self._lost(track, now):
                self._counts["reacquired"] += 1
                self._event("reacquired", object_id, now, track)
                self._lost_state[object_id] = False
        removed = []
        for object_id, track in self._tracks.items():
            lost = self._lost(track, now)
            if lost and not self._lost_state.get(object_id, False):
                self._counts["lost"] += 1
                self._ever_lost_ids.add(object_id)
                self._event("lost", object_id, now, track)
            self._lost_state[object_id] = lost
            if self.policy == "hard_expiry" and now - track.times[-1] > self.max_age_s + _TIME_TOL:
                self._counts["hard_removed"] += 1
                self._event("removed", object_id, now, track, reason="hard_expiry")
                removed.append(object_id)
            elif (self.policy == "conditional" and lost and ego is not None
                  and self.conditional_measurements_bounded):
                certificate = self._conditional_certificate(track, now, ego)
                if certificate["excluded_interaction_gap_m"] > _TIME_TOL:
                    self._counts["conditional_removed"] += 1
                    self._event("removed", object_id, now, track,
                                reason="reachable_balls_disjoint", **certificate)
                    removed.append(object_id)
        for object_id in removed:
            del self._tracks[object_id]

    def current(self, now: float) -> list[Track]:
        """Return isolated causal histories; never fabricate a lost position."""
        now = self._time(now)
        tracks = self._tracks.values()
        if self.policy == "hard_expiry":
            tracks = (t for t in tracks if now - t.times[-1] <= self.max_age_s + _TIME_TOL)
        return deepcopy(list(tracks))

    def uncertainty_for(self, track: Track, now: float) -> float:
        """Extra collision-envelope radius around the chosen nominal forecast.

        This is a policy ablation, not an alteration of the obstacle's physical
        radius or the common one-metre safety threshold.  Nominal CA/PPLNM
        motion and calibrated forecast balls are handled by the predictor.
        """
        now = self._time(now)
        validate_track(track, now)
        if self.policy == "hard_expiry":
            return 0.0
        stale_s = max(0.0, now - track.times[-1] - self.freshness_age_s)
        radius = self.uncertainty_growth_mps * stale_s
        if not np.isfinite(radius):
            raise ValueError("memory uncertainty overflow")
        return float(radius)

    def diagnostics(self, now: float) -> dict:
        now = self._time(now)
        tracks = self.current(now)
        radii = {int(t.object_id): self.uncertainty_for(t, now) for t in tracks}
        ages = {int(t.object_id): float(now - t.times[-1]) for t in tracks}
        return dict(policy=self.policy, time_s=now, max_age_s=self.max_age_s,
                    freshness_age_s=self.freshness_age_s,
                    active_track_count=len(tracks),
                    lost_ids=[int(t.object_id) for t in tracks if self._lost(t, now)],
                    ever_lost_ids=sorted(self._ever_lost_ids),
                    lost_object_count=len(self._ever_lost_ids),
                    retained_beyond_expiry_ids=[i for i, a in ages.items() if a > self.max_age_s + _TIME_TOL],
                    uncertainty_by_id=radii, age_by_id=ages,
                    maximum_uncertainty_m=max(radii.values(), default=0.0),
                    mean_uncertainty_m=float(np.mean(list(radii.values()))) if radii else 0.0,
                    uncertainty_status="configured_heuristic_not_calibrated_coverage",
                    conditional_bound_status="declared_assumption_not_certified",
                    conditional_measurements_bounded=self.conditional_measurements_bounded,
                    conditional_horizon_s=self.conditional_horizon_s,
                    **self._counts, events=deepcopy(self._events))


def make_track_memory(cfg) -> CausalTrackMemory:
    """Build from shared config; omitted Stage3 fields preserve Stage2 expiry."""
    return CausalTrackMemory(
        cfg.track_max_age_s, policy=getattr(cfg, "memory_policy", "hard_expiry"),
        expected_period_s=cfg.observation_period_s,
        observation_delay_s=cfg.observation_delay_s,
        uncertainty_growth_mps=getattr(cfg, "memory_uncertainty_growth_mps", 2.5),
        obstacle_speed_bound_mps=getattr(cfg, "memory_obstacle_speed_bound_mps", 4.0),
        ego_speed_bound_mps=getattr(cfg, "memory_ego_speed_bound_mps", 4.0),
        horizon_s=cfg.horizon_s, control_dt=cfg.control_dt,
        command_delay_s=cfg.command_delay_s,
        interaction_distance_m=cfg.interaction_distance_m,
        body_radius_m=cfg.body_radius_m, safety_margin_m=cfg.safety_margin_m,
        conditional_measurements_bounded=cfg.sensor_noise_std_m == 0)
