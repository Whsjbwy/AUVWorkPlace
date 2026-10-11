"""Physical-interface tests. None modify the retained REMUS coefficients."""
from dataclasses import replace

import numpy as np
import pytest

from auv3d.adapters.remus_guidance import (
    ReferenceAction, RemusGuidance, clone_plant, predict_reference, velocity_frames,
)
from auv3d.geometry import rotation_zyx, wrap_pi
from auv3d.remus import RemusPlant
from auv3d.remus_planner import (
    RemusDWA, _diverse_shortlist, candidate_references, trajectory_clearance,
)
from auv3d.sensors import Track
from auv3d.stage2_config import Stage2Config
from auv3d.terrain import HeightField


def plant(**kwargs):
    options = dict(depth=30., heading_deg=0., rpm=1100., current_speed=0.,
                   eta0=np.array([20., 100., 30., 0., 0., 0.]), initialize_references=True)
    options.update(kwargs)
    return RemusPlant(**options)


def execute(p, action, duration):
    for _ in range(round(duration / .02)):
        RemusGuidance.step(p, action, .02)
    return p


@pytest.mark.parametrize('rpm', [800., 1400.])
def test_speed_and_original_rpm_actuator_response(rpm):
    p = plant(rpm=rpm)
    first = RemusGuidance.step(p, ReferenceAction(30., 0., rpm), .02)
    assert first.applied[2] == rpm
    assert p.actual[2] == pytest.approx(.02 * rpm / p.vehicle.T_n)
    assert p.actual[2] < rpm  # Propeller lag, not an instantaneous speed command.
    execute(p, ReferenceAction(30., 0., rpm), 30.)
    assert p.nu[0] > 1.
    assert p.actual[2] == pytest.approx(rpm, abs=.01)
    assert np.isfinite(np.r_[p.eta, p.nu, p.actual]).all()


def test_higher_rpm_gives_higher_sustained_speed():
    low = execute(plant(), ReferenceAction(30., 0., 800.), 30.)
    high = execute(plant(), ReferenceAction(30., 0., 1400.), 30.)
    assert high.nu[0] > low.nu[0] + .5


@pytest.mark.parametrize('heading,depth', [(25., 35.), (-25., 25.)])
def test_original_heading_and_filtered_depth_reference_tracking(heading, depth):
    p = execute(plant(), ReferenceAction(depth, np.deg2rad(heading), 1200.), 100.)
    assert abs(float(wrap_pi(p.eta[5] - np.deg2rad(heading)))) < np.deg2rad(2.)
    assert abs(p.eta[2] - depth) < 1.5
    assert abs(p.vehicle.z_d - depth) > .1  # Original slow reference filter retained.
    assert abs(p.vehicle.r_d) <= p.vehicle.r_max + 1e-12


def test_full_ned_transforms_nonzero_roll_pitch_and_no_current_added_twice():
    p = plant(current_speed=.4, current_direction_deg=70.,
              eta0=np.array([20., 100., 30., .21, -.17, .63]))
    p.nu[:] = [1.4, .23, -.11, .03, -.04, .05]
    frames = velocity_frames(p)
    rotation = rotation_zyx(*p.eta[3:6])
    np.testing.assert_allclose(frames['ground_ned_mps'], rotation @ p.nu[:3], atol=1e-13)
    np.testing.assert_allclose(frames['relative_ned_geometric_mps'] + frames['current_ned_mps'],
                               frames['ground_ned_mps'], atol=1e-13)
    np.testing.assert_allclose(frames['euler_rates_rad_s'], p.gnc.Tzyx(*p.eta[3:5]) @ p.nu[3:6], atol=1e-13)
    # Explicitly expose the original yaw-only flow approximation rather than
    # silently changing the hydrodynamic model at nonzero roll and pitch.
    assert not np.allclose(frames['source_relative_body_mps'], frames['relative_body_geometric_mps'])
    old_eta = p.eta.copy()
    RemusGuidance.step(p, ReferenceAction(30., .63, 1100.), .02)
    np.testing.assert_allclose(p.eta[:3], old_eta[:3] + .02 * rotation @ p.nu[:3], atol=1e-12)


def test_cross_current_changes_executed_ground_track():
    calm = execute(plant(), ReferenceAction(30., 0., 1100.), 25.)
    flow = execute(plant(current_speed=.3, current_direction_deg=90.), ReferenceAction(30., 0., 1100.), 25.)
    assert abs(flow.eta[1] - calm.eta[1]) > 1.
    np.testing.assert_allclose(flow.ground_velocity(), flow.gnc.Rzyx(*flow.eta[3:6]) @ flow.nu[:3])


def test_autopilot_requests_clip_original_limits_and_keep_actuator_lag():
    p = plant(eta0=np.array([20., 100., 30., .1, .5, .8]))
    sample = RemusGuidance.step(p, ReferenceAction(80., -1., 1525.), .02)
    limits = np.array([p.vehicle.deltaMax_r, p.vehicle.deltaMax_s, p.vehicle.nMax])
    assert sample.saturated[:2].any()
    assert np.all(np.abs(sample.applied) <= limits)
    np.testing.assert_allclose(p.actual[:2], sample.applied[:2] * .02 / p.vehicle.T_delta)
    assert p.actual[2] < sample.applied[2]
    assert p.vehicle.z_int != 0. and p.vehicle.theta_int != 0.


def test_reference_heading_uses_nearest_equivalent_angle():
    p = plant(heading_deg=179., eta0=np.array([20., 100., 30., 0., 0., np.deg2rad(179.)]))
    RemusGuidance.step(p, ReferenceAction(30., np.deg2rad(-179.), 1000.), .02)
    assert p.vehicle.ref_psi == pytest.approx(181.)


@pytest.mark.parametrize('action', [ReferenceAction(-1., 0., 800.), ReferenceAction(101., 0., 800.),
                                    ReferenceAction(30., 0., 1526.), ReferenceAction(30., float('nan'), 800.)])
def test_reference_envelope_is_not_relaxed(action):
    p = plant()
    with pytest.raises(ValueError):
        RemusGuidance.step(p, action, .02)
    assert p.time_s == 0.


def test_clone_isolates_controller_reference_velocity_and_actuator_memory():
    p = execute(plant(), ReferenceAction(35., .4, 1100.), 2.)
    before = {key: np.copy(value) if isinstance(value, np.ndarray) else value
              for key, value in vars(p.vehicle).items()}
    eta, nu, actual = p.eta.copy(), p.nu.copy(), p.actual.copy()
    cloned = clone_plant(p)
    assert cloned.gnc is p.gnc
    assert cloned.vehicle is not p.vehicle
    for original, duplicate in [(p.eta, cloned.eta), (p.nu, cloned.nu),
                                (p.actual, cloned.actual), (p.vehicle.M, cloned.vehicle.M)]:
        assert not np.shares_memory(original, duplicate)
    execute(cloned, ReferenceAction(20., -.5, 800.), 1.)
    np.testing.assert_array_equal(p.eta, eta)
    np.testing.assert_array_equal(p.nu, nu)
    np.testing.assert_array_equal(p.actual, actual)
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(p.vehicle, key), value)
        else:
            assert getattr(p.vehicle, key) == value


def test_prediction_matches_physical_execution_and_all_physics_points_are_retained():
    p = execute(plant(current_speed=.2, current_direction_deg=65.), ReferenceAction(30., 0., 900.), 1.)
    p.eta[3:5] = [.04, -.03]
    action = ReferenceAction(34., .35, 1200.)
    prediction = predict_reference(p, action, 3., .02, record_dt=.2)
    assert prediction.states.shape == (151, 16)
    assert prediction.controls.shape == (150, 10)
    rows = [np.r_[0., p.eta, p.nu, p.actual]]
    for i in range(150):
        sample = RemusGuidance.step(p, action, .02)
        rows.append(np.r_[(i + 1) * .02, p.eta, p.nu, p.actual])
        np.testing.assert_allclose(sample.applied, prediction.controls[i, 4:7], atol=2e-12)
    np.testing.assert_allclose(rows, prediction.states, atol=2e-12, rtol=0)


def test_shortlist_covers_heading_depth_and_rpm_in_fixed_budget():
    cfg = Stage2Config()
    p = plant()
    actions = candidate_references(p, np.array([200., 100., 30.]), cfg)
    chosen = _diverse_shortlist(actions, np.arange(len(actions), dtype=float), cfg.candidate_budget)
    assert len(chosen) == cfg.candidate_budget
    assert len({actions[i].heading_rad for i in chosen}) == len(cfg.heading_offsets_deg)
    assert {actions[i].depth_m for i in chosen} >= {22., 38.}
    assert {actions[i].rpm for i in chosen} >= {800., 1400.}


def test_planner_uses_isolated_full_predictions_and_respects_budget():
    cfg = replace(Stage2Config(), horizon_s=2., control_dt=1., candidate_budget=3)
    p = plant()
    before = np.r_[p.eta, p.nu, p.actual]
    result = RemusDWA(cfg).plan(p, np.array([200., 100., 30.]), HeightField.flat(260., 200., 90.), [], 0.)
    assert result.feasible
    assert result.verified_count == 3
    assert result.valid_count <= result.verified_count
    assert result.states.shape == (101, 16)
    assert result.verification_model == 'original_remus100_full_6dof_autopilot_actuators'
    assert result.planning_ms > 0.
    np.testing.assert_array_equal(np.r_[p.eta, p.nu, p.actual], before)
    assert p.time_s == 0.
    assert p.vehicle.z_int == p.vehicle.theta_int == p.vehicle.e_psi_int == 0.


def test_dynamic_forecast_uses_absolute_observation_and_physics_times():
    cfg = Stage2Config(prediction_margin_m=0.)
    # At t=2 the obstacle is N=52 and moves toward the AUV at 1 m/s.
    track = Track(3, .5, [0., 1., 2.],
                  [np.array([54., 100., 30.]), np.array([53., 100., 30.]), np.array([52., 100., 30.])])
    rows = np.zeros((3, 16))
    rows[:, 0] = [0., 1., 2.]
    rows[:, 1:4] = [[48., 100., 30.], [49., 100., 30.], [50., 100., 30.]]
    clear = trajectory_clearance(rows, HeightField.flat(260., 200., 90.), [track], 2., cfg)
    assert clear == pytest.approx(-.5 - cfg.body_radius_m)


def test_no_feasible_candidate_emergency_coasts_without_instant_stop():
    cfg = replace(Stage2Config(), horizon_s=2., control_dt=1., candidate_budget=2)
    p = plant()
    p.nu[:] = [1.5, .1, 0., 0., 0., 0.]
    p.actual[:] = [0., 0., 1000.]
    track = Track(1, 10., [0.], [p.eta[:3].copy()])
    result = RemusDWA(cfg).plan(p, np.array([200., 100., 30.]), HeightField.flat(260., 200., 90.), [track], 0.)
    assert not result.feasible and result.emergency
    assert not result.emergency_verified_safe
    assert 'not_safe' in result.reason
    assert result.action.rpm == 0.
    assert result.states[0, 7] == 1.5
    assert result.states[1, 7] > 1.
    assert result.states[1, 15] > 0.  # Actual propeller decays with T_n.
    assert result.states[-1, 1] > result.states[0, 1] + 1.
    assert p.nu[0] == 1.5 and p.actual[2] == 1000.


def test_planner_rejects_time_desynchronization():
    with pytest.raises(ValueError, match='simulation time'):
        RemusDWA(Stage2Config()).plan(plant(), [200., 100., 30.], HeightField.flat(), [], 1.)
