"""Reference guidance around the unmodified REMUS autopilot and dynamics.

The planner samples depth [m, down positive], heading [rad] and propeller RPM.
Body pitch/yaw rates are never passed to a rudder or stern plane. Each physics
step calls the original controller exactly once, clips its actuator request to
the original limits, and lets the original actuator dynamics integrate it.
Controller integral and reference-model states are intentionally retained.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math

import numpy as np

from ..geometry import wrap_pi
from ..remus import RemusPlant


@dataclass(frozen=True)
class ReferenceAction:
    depth_m: float
    heading_rad: float
    rpm: float

    def as_array(self) -> np.ndarray:
        return np.array([self.depth_m, self.heading_rad, self.rpm], dtype=float)

    def validate(self, plant: RemusPlant) -> None:
        if not np.isfinite(self.as_array()).all():
            raise ValueError('REMUS reference must contain finite depth, heading, RPM')
        if not 0 <= self.depth_m <= 100:
            raise ValueError('REMUS desired depth must remain in [0,100] m')
        if not 0 <= self.rpm <= plant.vehicle.nMax:
            raise ValueError('REMUS desired RPM must remain within original forward limit')


@dataclass(frozen=True)
class ControlSample:
    raw: np.ndarray
    applied: np.ndarray
    saturated: np.ndarray


@dataclass(frozen=True)
class ReferencePrediction:
    # [offset seconds, eta (6), nu (6), actual actuators (3)]. Every physics point
    # is retained for collision checks. Sparse visualization is a later concern.
    states: np.ndarray
    # One row per executed integration interval:
    # [interval-start offset, raw(3), applied(3), saturated(3)].
    controls: np.ndarray

    @property
    def positions(self) -> np.ndarray:
        return self.states[:, 1:4]


def validate_plant_state(plant: RemusPlant) -> None:
    """Reject malformed/nonnumeric input before a controller mutates its memory."""
    for name, shape in (('eta', (6,)), ('nu', (6,)), ('actual', (3,))):
        state = np.asarray(getattr(plant, name), dtype=float)
        if state.shape != shape:
            raise ValueError(f'REMUS {name} must have shape {shape}')
        if not np.isfinite(state).all():
            raise FloatingPointError(f'REMUS {name} must contain finite values')
    if not math.isfinite(plant.time_s):
        raise FloatingPointError('REMUS simulation time must be finite')
    controller_states = ('z_d', 'psi_d', 'r_d', 'a_d', 'z_int', 'theta_int', 'e_psi_int')
    if not np.isfinite([getattr(plant.vehicle, name) for name in controller_states]).all():
        raise FloatingPointError('REMUS controller memory must contain finite values')


class RemusGuidance:
    """Stateless adapter; all dynamic state lives in the supplied RemusPlant."""

    @staticmethod
    def step(plant: RemusPlant, action: ReferenceAction, dt: float = .02) -> ControlSample:
        if not math.isfinite(dt) or not 0 < dt <= .02:
            raise ValueError('REMUS physics integration requires 0 < dt <= 0.02 s')
        action.validate(plant)
        validate_plant_state(plant)
        # Unwrap at the actual yaw. The source reference model uses radians
        # internally but accepts its heading target in degrees.
        target = plant.eta[5] + float(wrap_pi(action.heading_rad - plant.eta[5]))
        plant.set_reference(action.depth_m, np.rad2deg(target), action.rpm)
        raw = np.asarray(plant.vehicle.depthHeadingAutopilot(plant.eta, plant.nu, dt), dtype=float)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            raise FloatingPointError('original autopilot produced an invalid actuator request')
        limits = np.array([plant.vehicle.deltaMax_r, plant.vehicle.deltaMax_s, plant.vehicle.nMax])
        applied = np.clip(raw, -limits, limits)
        saturated = np.abs(raw - applied) > 1e-12
        # Supplying command prevents RemusPlant.step from invoking the controller
        # a second time. The original dynamics retain fin/propeller time constants.
        plant.step(dt, command=applied)
        return ControlSample(raw.copy(), applied.copy(), saturated.copy())


def clone_plant(plant: RemusPlant) -> RemusPlant:
    """Isolate vehicle, integrators, reference filters, nu and actuator memory.

    Python modules are immutable shared implementation dependencies; deepcopy
    cannot pickle them. The memo shares only the GNC module, not vehicle state.
    This also preserves future ordinary state fields added to RemusPlant.
    """
    validate_plant_state(plant)
    clone = copy.deepcopy(plant, {id(plant.gnc): plant.gnc})
    # nu can alias vehicle.nu in the original source after dynamics(). deepcopy
    # preserves that alias *inside* the clone, while isolating it from reality.
    return clone


def predict_reference(
    plant: RemusPlant,
    action: ReferenceAction,
    horizon_s: float,
    physics_dt: float = .02,
    record_dt: float | None = None,
    record_controls: bool = True,
) -> ReferencePrediction:
    """Forward integrate an isolated original six-DOF model.

    All integration points are returned even if a display record_dt is supplied.
    This prevents sparse plot sampling from weakening collision verification.
    """
    if not math.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError('prediction horizon must be finite and positive')
    if not math.isfinite(physics_dt) or not 0 < physics_dt <= .02:
        raise ValueError('prediction physics_dt must be in (0,0.02]')
    if record_dt is not None and (not math.isfinite(record_dt) or record_dt < physics_dt):
        raise ValueError('record_dt must be finite and at least the physics step')
    simulated = clone_plant(plant)
    action.validate(simulated)
    # A positive horizon shorter than the rounding tolerance still requires
    # one physical integration interval, rather than returning the initial pose.
    count = max(1, int(math.ceil(horizon_s / physics_dt - 1e-10)))
    rows = np.empty((count + 1, 16))
    controls = np.empty((count if record_controls else 0, 10))
    elapsed = 0.
    rows[0] = np.r_[0., simulated.eta, simulated.nu, simulated.actual]
    for i in range(count):
        step = min(physics_dt, horizon_s - elapsed)
        sample = RemusGuidance.step(simulated, action, step)
        if record_controls:
            controls[i] = np.r_[elapsed, sample.raw, sample.applied, sample.saturated.astype(float)]
        elapsed = min(horizon_s, elapsed + step)
        rows[i + 1] = np.r_[elapsed, simulated.eta, simulated.nu, simulated.actual]
    return ReferencePrediction(rows, controls)


def velocity_frames(plant: RemusPlant) -> dict[str, np.ndarray]:
    """Full six-DOF coordinate transforms and source-model flow convention.

    nu is absolute body velocity, so R@nu already gives ground velocity. Do not
    add current again. The retained source dynamics approximate body current
    using yaw only; that convention is reported separately from the exact 3-D
    coordinate transform when roll/pitch is nonzero.
    """
    rotation = plant.gnc.Rzyx(*plant.eta[3:6])
    current = np.array([
        plant.vehicle.V_c * np.cos(plant.vehicle.beta_c),
        plant.vehicle.V_c * np.sin(plant.vehicle.beta_c), 0.,
    ])
    relative_body = plant.nu[:3] - rotation.T @ current
    return {
        'ground_ned_mps': rotation @ plant.nu[:3],
        'current_ned_mps': current,
        'relative_body_geometric_mps': relative_body,
        'relative_ned_geometric_mps': rotation @ relative_body,
        'source_relative_body_mps': plant.relative_velocity().copy(),
        'euler_rates_rad_s': plant.gnc.Tzyx(*plant.eta[3:5]) @ plant.nu[3:6],
    }
