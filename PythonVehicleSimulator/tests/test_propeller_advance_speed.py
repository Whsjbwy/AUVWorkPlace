#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The pytest for 'test_propeller_advance_speed.py' can be run in the terminal using:
1) cd <path of the Python Vehicle Simulator installation>
2) pytest -k propeller -v
"""

import numpy as np
from python_vehicle_simulator.vehicles import remus100


def nu_dot(vehicle, eta, nu, u_actual, u_control, dt):
    """Recover nu_dot from one forward-Euler step of dynamics()."""
    nu_out, _ = vehicle.dynamics(eta.copy(), nu.copy(), u_actual.copy(), u_control, dt)
    return (nu_out - nu) / dt


def test_propulsion_depends_on_relative_flow_only():
    """Hydrodynamic and propulsion forces arise from the flow past the hull
    and propeller, so they can depend only on the relative velocity nu_r,
    never on the ground-relative velocity. Two cases with identical nu_r --
    still water at u = 1.5 m/s, and a 0.5 m/s head current with u = 1.0 m/s
    -- must therefore produce identical accelerations. The yaw rate is zero
    so the rotating-frame current derivative Dnu_c vanishes and the full
    nu_dot is directly comparable. The current is 0.5 (exact in binary
    floating point) so nu - nu_c reproduces nu_r bitwise."""
    dt = 0.02
    eta = np.zeros(6)
    u_actual = np.array([0.05, -0.03, 1200.0])
    u_control = np.array([0.05, -0.03, 1200.0])
    nu_r = np.array([1.5, 0.1, 0.2, 0.05, 0.1, 0.0])

    still_water = remus100()
    a = nu_dot(still_water, eta, nu_r, u_actual, u_control, dt)

    current = 0.5
    head_current = remus100("stepInput", 0, 0, 0, current, 180)
    nu = nu_r + np.array([-current, 0, 0, 0, 0, 0])
    b = nu_dot(head_current, eta, nu, u_actual, u_control, dt)

    assert np.allclose(a, b, atol=1e-10)
