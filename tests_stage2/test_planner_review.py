"""Regression tests for causal planning inputs and full-grid budget diagnostics."""
from dataclasses import replace
import copy
import json

import numpy as np
import pytest

from auv3d.adapters.remus_guidance import ReferenceAction, RemusGuidance, predict_reference
from auv3d.remus import RemusPlant
import auv3d.remus_planner as planning
from auv3d.sensors import Track
from auv3d.stage2_config import Stage2Config
from auv3d.terrain import HeightField


def make_plant():
    return RemusPlant(depth=30., heading_deg=0., rpm=1100., current_speed=.15,
                      current_direction_deg=70., eta0=[30., 100., 30., .03, -.02, .04],
                      initialize_references=True)


def short_config(budget=9):
    return replace(Stage2Config(), horizon_s=.4, control_dt=.2, candidate_budget=budget)


def state_snapshot(plant):
    return (plant.eta.copy(), plant.nu.copy(), plant.actual.copy(), plant.time_s,
            copy.deepcopy(vars(plant.vehicle)))


def assert_unchanged(plant, snapshot):
    eta, nu, actual, time_s, vehicle = snapshot
    np.testing.assert_array_equal(plant.eta, eta)
    np.testing.assert_array_equal(plant.nu, nu)
    np.testing.assert_array_equal(plant.actual, actual)
    assert plant.time_s == time_s
    for name, value in vehicle.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(plant.vehicle, name), value)
        else:
            assert getattr(plant.vehicle, name) == value


@pytest.mark.parametrize('budget', [9, 18, 45])
def test_shortlist_budget_covers_reference_extrema_without_duplicate_candidates(budget):
    plant = make_plant()
    cfg = short_config(budget)
    actions = planning.candidate_references(plant, np.array([210., 100., 30.]), cfg)
    # Several adversarial surrogate orderings exercise coverage beyond the
    # default easy straight-line ranking. Coverage is marginal, not all pairs.
    rng = np.random.default_rng(19)
    for costs in (np.arange(45.), np.arange(45.)[::-1], rng.permutation(45).astype(float)):
        indices = planning._diverse_shortlist(actions, costs, budget)
        assert len(indices) == len(set(indices)) == budget
        selected = np.array([actions[index].as_array() for index in indices])
        assert len(np.unique(selected[:, 1])) == 5
        assert {22., 38.} <= set(selected[:, 0])
        assert {800., 1400.} <= set(selected[:, 2])
        if budget == 45:
            assert set(indices) == set(range(45))


def test_full_grid_diagnostic_integrates_all_references_and_isolates_entire_plant():
    plant = make_plant()
    before = state_snapshot(plant)
    previous = ReferenceAction(32., .2, 900.)
    report = planning.evaluate_reference_grid(plant, [210., 100., 30.],
               HeightField.flat(260., 200., 90.), [], 0., short_config(), previous)
    assert report['candidate_count'] == report['feasible_count'] == 45
    assert report['prediction_failure_count'] == 0
    assert report['cost_history'] == 'supplied_previous_reference'
    assert report['previous_reference'] == previous.as_array().tolist()
    assert report['shortlist_feasible_counts'] == {'9': 9, '18': 18, '45': 45}
    assert report['obstacle_information'] == 'causal tracks only; no future truth'
    assert report['evaluation_ms'] > 0.
    assert len(report['candidates']) == 45
    assert all(row['minimum_predicted_clearance_m'] >= 1. for row in report['candidates'])
    json.dumps(report, allow_nan=False)  # Machine-readable finite diagnostic data.
    assert_unchanged(plant, before)


@pytest.mark.parametrize('budget', [9, 18, 45])
def test_diagnostic_reproduces_actual_budget_selection_and_candidate_evidence(budget):
    plant = make_plant()
    cfg = short_config(budget)
    terrain = HeightField.flat(260., 200., 90.)
    previous = ReferenceAction(35., .15, 1100.)
    planner = planning.RemusDWA(cfg)
    planner.previous = previous
    report = planning.evaluate_reference_grid(plant, [210., 100., 30.], terrain, [], 0., cfg, previous)
    plan = planner.plan(plant, [210., 100., 30.], terrain, [], 0.)
    best = report['shortlist_best_feasible_indices'][str(budget)]
    np.testing.assert_array_equal(plan.command, report['candidates'][best]['reference'])
    assert plan.feasible
    assert plan.shortlist_indices == report['shortlist_indices']
    assert len(plan.candidate_evaluations) == budget
    assert [row['index'] for row in plan.candidate_evaluations] == report['shortlist_indices'][str(budget)]
    for row in plan.candidate_evaluations:
        full = report['candidates'][row['index']]
        assert row['feasible'] == full['feasible']
        assert row['minimum_predicted_clearance_m'] == pytest.approx(full['minimum_predicted_clearance_m'])
        assert row['cost'] == pytest.approx(full['cost'])


def test_full_grid_reports_no_feasible_reference_without_inventing_safe_backup():
    plant = make_plant()
    track = Track(4, 10., [0.], [plant.eta[:3].copy()])
    report = planning.evaluate_reference_grid(plant, [210., 100., 30.],
               HeightField.flat(260., 200., 90.), [track], 0., short_config())
    assert report['reason'] == 'no_feasible_reference_in_full_grid'
    assert report['feasible_count'] == 0
    assert report['best_feasible_index'] is None
    assert report['shortlist_feasible_counts'] == {'9': 0, '18': 0, '45': 0}
    assert all(row['minimum_predicted_clearance_m'] < 0. for row in report['candidates'])


def test_full_grid_rejects_numeric_candidate_failures_without_poisoning_real_state(monkeypatch):
    plant = make_plant()
    before = state_snapshot(plant)
    actual_predict = planning.predict_reference

    def failed_low_rpm(real_plant, action, *args, **kwargs):
        if action.rpm == 800.:
            raise FloatingPointError('deliberate numeric rollout failure')
        return actual_predict(real_plant, action, *args, **kwargs)

    monkeypatch.setattr(planning, 'predict_reference', failed_low_rpm)
    report = planning.evaluate_reference_grid(plant, [210., 100., 30.],
               HeightField.flat(260., 200., 90.), [], 0., short_config())
    assert report['prediction_failure_count'] == 15
    assert report['feasible_count'] == 30
    failed = [row for row in report['candidates'] if row['prediction_failure']]
    assert all(not row['feasible'] and row['cost'] is None for row in failed)
    assert all('FloatingPointError' in row['prediction_failure'] for row in failed)
    json.dumps(report, allow_nan=False)
    assert_unchanged(plant, before)


@pytest.mark.parametrize('radius', [-1., 0., float('nan'), float('inf')])
def test_invalid_obstacle_radius_cannot_create_false_clearance(radius):
    plant = make_plant()
    before = state_snapshot(plant)
    track = Track(1, radius, [0.], [np.array([60., 100., 30.])])
    with pytest.raises(ValueError, match='radius'):
        planning.RemusDWA(short_config()).plan(plant, [210., 100., 30.],
                          HeightField.flat(260., 200., 90.), [track], 0.)
    assert_unchanged(plant, before)


@pytest.mark.parametrize('times', [[0., 0.], [0., -.1], [0., float('nan')], [0., .1]])
def test_bad_or_future_track_times_rejected_before_any_division(times):
    track = Track(1, 4., times, [np.array([60., 100., 30.]), np.array([59., 100., 30.])])
    # Duplicate or reversed times must give a useful input error, rather than
    # RuntimeWarning / division by zero or a falsely safe forecast.
    with np.errstate(all='raise'), pytest.raises(ValueError):
        planning.RemusDWA(short_config()).plan(make_plant(), [210., 100., 30.],
                          HeightField.flat(260., 200., 90.), [track], 0.)


@pytest.mark.parametrize('positions', [[[60., 100.]], [[60., 100., float('nan')]], []])
def test_malformed_obstacle_positions_rejected(positions):
    track = Track(1, 4., [0.], positions)
    with pytest.raises(ValueError, match='positions'):
        planning._forecast_tracks([track], [0., .2], 0., short_config())


def test_observation_delay_is_extrapolated_on_absolute_time_without_future_truth():
    track = Track(1, 4., [0., 1., 2.], [np.array([60., 100., 30.]),
                  np.array([59., 100., 30.]), np.array([58., 100., 30.])])
    forecasts = planning._forecast_tracks([track], [0., .2, .4], 3., short_config())
    np.testing.assert_allclose(forecasts[0][0][:, 0], [57., 56.8, 56.6])
    assert forecasts[0][1] == pytest.approx(4. + short_config().body_radius_m + .75 + 1.)


@pytest.mark.parametrize('offsets', [[0., 0.], [0., -.2], [0., float('nan')], [.1, .3]])
def test_trajectory_time_contract_rejects_duplicate_negative_nonfinite_or_missing_start(offsets):
    rows = np.zeros((2, 16))
    rows[:, 0] = offsets
    rows[:, 1:4] = [30., 100., 30.]
    with pytest.raises((ValueError, FloatingPointError)):
        planning.trajectory_clearance(rows, HeightField.flat(260., 200., 90.), [], 0., short_config())


@pytest.mark.parametrize('name,index', [('eta', 0), ('nu', 0), ('actual', 0)])
def test_bad_physical_state_rejected_before_reference_or_controller_mutation(name, index):
    plant = make_plant()
    getattr(plant, name)[index] = np.nan
    references = [plant.vehicle.ref_z, plant.vehicle.ref_psi, plant.vehicle.ref_n]
    integrals = [plant.vehicle.z_int, plant.vehicle.theta_int, plant.vehicle.e_psi_int]
    with pytest.raises(FloatingPointError, match='finite'):
        RemusGuidance.step(plant, ReferenceAction(40., .6, 900.))
    assert references == [plant.vehicle.ref_z, plant.vehicle.ref_psi, plant.vehicle.ref_n]
    assert integrals == [plant.vehicle.z_int, plant.vehicle.theta_int, plant.vehicle.e_psi_int]
    assert plant.time_s == 0.


def test_bad_controller_memory_and_clock_rejected_before_physics():
    plant = make_plant()
    plant.vehicle.z_int = np.nan
    with pytest.raises(FloatingPointError, match='controller memory'):
        RemusGuidance.step(plant, ReferenceAction(40., .6, 900.))
    plant.vehicle.z_int = 0.
    plant.time_s = np.nan
    with pytest.raises(FloatingPointError, match='time'):
        planning.RemusDWA(short_config()).plan(plant, [210., 100., 30.], HeightField.flat(), [], 0.)


def test_nonfinite_terrain_cannot_be_masked_by_finite_domain_clearance():
    terrain = HeightField.flat(260., 200., 90.)
    terrain.depths[:] = np.inf
    with pytest.raises(FloatingPointError, match='terrain'):
        planning.RemusDWA(short_config()).plan(make_plant(), [210., 100., 30.], terrain, [], 0.)


def test_tiny_positive_horizon_still_advances_one_physical_interval():
    plant = make_plant()
    prediction = predict_reference(plant, ReferenceAction(30., .04, 1100.), 1e-12)
    assert prediction.states.shape == (2, 16)
    assert prediction.states[-1, 0] == pytest.approx(1e-12, rel=0., abs=1e-24)
    assert prediction.controls.shape == (1, 10)
    assert plant.time_s == 0.
