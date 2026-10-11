"""Regressions for demonstrated P25 false-safe inputs and fallback reports."""
from dataclasses import replace
import importlib

import numpy as np
import pytest

from auv3d.sensors import Track
from auv3d.stage2_config import Stage2Config
from auv3d.stage2_p25 import p25_config
from auv3d.terrain import HeightField


@pytest.fixture(params=('stage1', 'stage2'))
def p25(request):
    module = importlib.import_module('auv3d.planner' if request.param == 'stage1'
                                     else 'auv3d.stage2_p25')
    cfg = p25_config(replace(Stage2Config(), horizon_s=2., warmup_s=0.))
    return module, module.DWA3D(cfg), np.array([30., 100., 35., 1., 0., 0.]), \
        np.array([210., 100., 35.]), HeightField.flat(depth=90.)


def track(radius=4., times=(0.,)):
    return Track(1, radius, list(times), [np.array([32., 100., 35.]) for _ in times])


@pytest.mark.parametrize('radius', [-100., 0., float('nan'), float('inf')])
def test_invalid_radius_cannot_turn_a_colliding_sphere_into_a_feasible_path(p25, radius):
    _, planner, state, goal, terrain = p25
    previous = planner.previous.copy()
    with pytest.raises(ValueError, match='radius'):
        planner.plan(state, goal, terrain, [track(radius)], 0.)
    np.testing.assert_array_equal(planner.previous, previous)


@pytest.mark.parametrize('times', [(0.1,), (0., 0.), (0., -.1)])
def test_future_or_repeated_measurement_times_are_rejected_before_forecasting(p25, times):
    _, planner, state, goal, terrain = p25
    with np.errstate(all='raise'), pytest.raises(ValueError):
        planner.plan(state, goal, terrain, [track(times=times)], 0.)


@pytest.mark.parametrize('field,value', [('depths', float('nan')), ('depths', float('inf')),
                                        ('slope_bound', float('nan')),
                                        ('slope_bound', float('inf')), ('slope_bound', -1.)])
def test_invalid_terrain_cannot_be_hidden_by_a_finite_horizontal_domain_gap(p25, field, value):
    _, planner, state, goal, terrain = p25
    if field == 'depths':
        terrain.depths[:] = value
    else:
        terrain.slope_bound = value
    with pytest.raises(FloatingPointError, match='terrain'):
        planner.plan(state, goal, terrain, [], 0.)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_a_nonfinite_rollout_state_cannot_be_accepted_from_finite_positions(p25, monkeypatch, value):
    module, planner, state, goal, terrain = p25
    original = module.rollout

    def broken(*args):
        paths = original(*args)
        paths[0, 1, 3] = value  # Speed is not part of the position-only gap test.
        return paths

    monkeypatch.setattr(module, 'rollout', broken)
    with pytest.raises(FloatingPointError, match='trajectory'):
        planner.plan(state, goal, terrain, [], 0.)


@pytest.mark.parametrize('field', ['state', 'goal', 'previous', 'now'])
def test_invalid_planning_input_is_rejected_before_sampling_commands(p25, monkeypatch, field):
    module, planner, state, goal, terrain = p25
    now = 0.
    if field == 'state':
        state[3] = np.nan
    elif field == 'goal':
        goal[0] = np.inf
    elif field == 'previous':
        planner.previous[0] = np.nan
    else:
        now = np.nan

    def should_not_sample(*_):
        pytest.fail('invalid planning input reached command sampling')

    monkeypatch.setattr(module, 'candidate_commands', should_not_sample)
    with pytest.raises(ValueError):
        planner.plan(state, goal, terrain, [], now)


@pytest.mark.parametrize('failure', ['nonfinite', 'wrong_shape'])
def test_invalid_obstacle_forecast_cannot_pass_collision_validation(p25, monkeypatch, failure):
    _, planner, state, goal, terrain = p25

    def broken_forecast(_track, times):
        return np.full((len(times), 3), np.nan) if failure == 'nonfinite' else np.zeros((len(times), 2))

    monkeypatch.setattr(planner.predictor, 'predict', broken_forecast)
    with pytest.raises(FloatingPointError, match='obstacle prediction'):
        planner.plan(state, goal, terrain, [track()], 0.)


def test_finite_causal_input_still_produces_a_verified_kinematic_candidate(p25):
    _, planner, state, goal, terrain = p25
    clear_track = Track(2, 4., [-1., 0.],
                        [np.array([65., 106., 35.]), np.array([65., 107., 35.])])
    original_times = clear_track.times.copy()
    original_positions = np.array(clear_track.positions)
    result = planner.plan(state, goal, terrain, [clear_track], 0.)
    assert result.feasible and result.valid_count > 0
    assert result.minimum_predicted_clearance_m >= planner.cfg.planner.safety_margin_m
    assert np.isfinite(result.states).all()
    assert clear_track.times == original_times
    np.testing.assert_array_equal(clear_track.positions, original_positions)


def test_real_initial_overlap_remains_infeasible_with_unchanged_sphere_size(p25):
    _, planner, state, goal, terrain = p25
    result = planner.plan(state, goal, terrain, [track()], 0.)
    assert not result.feasible and result.valid_count == 0
    assert result.minimum_predicted_clearance_m < 0.


def test_operating_depth_boundary_gets_the_same_one_metre_clearance(p25):
    _, planner, state, goal, _ = p25
    state[2] = goal[2] = 99.1
    result = planner.plan(state, goal, HeightField.flat(depth=200.), [], 0.)
    expected_gap = planner.cfg.vehicle.depth_limit_m-planner.cfg.vehicle.body_radius_m-state[2]
    assert 0. < expected_gap < planner.cfg.planner.safety_margin_m
    assert not result.feasible and result.valid_count == 0
    assert result.minimum_predicted_clearance_m == pytest.approx(expected_gap)


def test_reported_emergency_clearance_belongs_to_the_executed_fallback(p25):
    _, planner, state, goal, terrain = p25
    obstacle = track(radius=2.)
    result = planner.plan(state, goal, terrain, [obstacle], 0.)
    assert not result.feasible
    np.testing.assert_allclose(result.command, [.4, 0., 0.], atol=1e-14)
    # Constant heading/depth makes the actual fallback a straight 0.8 m segment.
    # Its closest point is the endpoint; this independent geometry exposes the
    # former stage-1 report that copied another rejected candidate's clearance.
    expected_gap = (32.-result.states[-1, 0]-obstacle.radius_m
                    -planner.cfg.vehicle.body_radius_m-planner.cfg.planner.base_prediction_margin_m)
    assert result.minimum_predicted_clearance_m == pytest.approx(expected_gap, abs=1e-12)


@pytest.mark.parametrize('unsafe', [False, True])
def test_arrival_preserves_the_emergency_forecast_safety_result(monkeypatch, unsafe):
    import auv3d.stage2_p25 as module
    cfg = replace(Stage2Config(), max_time_s=6., control_dt=1., horizon_s=2.,
                  warmup_s=0., goal_tolerance_m=.2)
    original_scene = module.make_stage2_scenario
    original_plan = module.DWA3D.plan

    def short_scene(name, settings):
        scene = original_scene(name, settings)
        scene.goal = scene.start_eta[:3]+np.array([3., 0., 0.])
        return scene

    def mark_emergency(self, *args):
        result = original_plan(self, *args)
        return replace(result, feasible=False,
                       minimum_predicted_clearance_m=cfg.safety_margin_m+(-.01 if unsafe else 1.))

    monkeypatch.setattr(module, 'make_stage2_scenario', short_scene)
    monkeypatch.setattr(module.DWA3D, 'plan', mark_emergency)
    result = module.run_p25_episode(cfg, 'empty3d')
    summary = result['summary']
    assert summary['status'] == 'success' and summary['reached_goal']
    assert summary['actual_safety_margin_met']
    assert summary['emergency_count'] > 0
    assert summary['avoidance_acceptance_passed'] is (not unsafe)
    assert bool(summary['unsafe_emergency_forecast_count']) is unsafe
    assert all(event['forecast_clear'] is (not unsafe) for event in result['events'])
    expected_status = ('failed_unsafe_emergency_forecast' if unsafe
                       else 'passed_finite_horizon_kinematic_checks')
    assert summary['safety_validation_status'] == expected_status
    assert not summary['remus6dof_avoidance_claim']
