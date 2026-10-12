"""Physical/reference-policy tests; passing these is not scene safety success."""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest

from auv3d.adapters.remus_guidance import ReferenceAction, predict_reference
from auv3d.geometry import wrap_pi
from auv3d.remus import RemusPlant
import auv3d.remus_planner as planning
from auv3d.sensors import Track
from auv3d.stage2_config import Stage2Config
from auv3d.stage3_candidates import (backup_references, coverage_shortlist,
                                     ranked_shortlist)
from auv3d.terrain import HeightField


def plant():
    instance = RemusPlant(depth=30., heading_deg=0., rpm=1100., current_speed=.15,
                          current_direction_deg=70., eta0=[30., 100., 30., .03, -.02, .04],
                          initialize_references=True)
    instance.nu[:] = [1.5, .05, -.02, .01, -.01, .02]
    instance.actual[:] = [.01, -.02, 1000.]
    return instance


def short_config(**changes):
    return replace(Stage2Config(), horizon_s=.4, control_dt=.2, **changes)


def snapshot(instance):
    return (instance.eta.copy(), instance.nu.copy(), instance.actual.copy(),
            instance.time_s, copy.deepcopy(vars(instance.vehicle)))


def unchanged(instance, before):
    for actual, expected in zip((instance.eta, instance.nu, instance.actual), before[:3]):
        np.testing.assert_array_equal(actual, expected)
    assert instance.time_s == before[3]
    for name, expected in before[4].items():
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(getattr(instance.vehicle, name), expected)
        else:
            assert getattr(instance.vehicle, name) == expected


def test_default_grid_retains_exact_original_45_order_and_references():
    instance, cfg = plant(), short_config()
    goal = np.array([210., 100., 30.])
    actual = planning.candidate_references(instance, goal, cfg)
    expected = [ReferenceAction(30. + depth,
                               float(instance.eta[5] + wrap_pi(heading - instance.eta[5])), rpm)
                for heading in np.deg2rad(cfg.heading_offsets_deg)
                for depth in cfg.depth_offsets_m for rpm in cfg.rpm_values]
    assert len(actual) == len(expected) == 45
    np.testing.assert_array_equal([action.as_array() for action in actual],
                                  [action.as_array() for action in expected])


def test_expanded_grid_has_unchanged_prefix_bounded_legal_unique_extra_actions():
    instance = plant()
    goal = np.array([210., 105., 46.])
    original = planning.candidate_references(instance, goal, short_config())
    actions = planning.candidate_references(instance, goal, short_config(candidate_scheme='expanded'))
    np.testing.assert_array_equal([action.as_array() for action in actions[:45]],
                                   [action.as_array() for action in original])
    assert 45 < len(actions) <= 69
    arrays = np.array([action.as_array() for action in actions])
    assert len({tuple(array) for array in arrays}) == len(actions)
    assert {30., 38., 46.} <= set(arrays[45:, 0])
    assert 600. in arrays[45:, 2]
    for action in actions:
        action.validate(instance)
    bearing = np.arctan2(5., 180.)
    assert any(abs(action.heading_rad - (bearing + np.pi / 2.)) < 1e-12 for action in actions[45:])


@pytest.mark.parametrize('policy', ['diverse', 'ranked', 'coverage'])
@pytest.mark.parametrize('budget', [9, 18, 45])
def test_configured_policy_respects_full_rollout_budget_and_isolates_plant(policy, budget, monkeypatch):
    instance, terrain = plant(), HeightField.flat(260., 200., 90.)
    before = snapshot(instance)
    calls = []
    actual_predict = planning.predict_reference

    def recording_predict(real_plant, action, *args, **kwargs):
        calls.append(action.as_array().tolist())
        return actual_predict(real_plant, action, *args, **kwargs)

    monkeypatch.setattr(planning, 'predict_reference', recording_predict)
    result = planning.RemusDWA(short_config(shortlist_policy=policy, candidate_budget=budget)).plan(
        instance, [210., 100., 30.], terrain, [], 0.)
    assert len(calls) == result.verified_count == budget
    assert len(result.candidate_evaluations) == budget
    assert result.candidate_scheme == 'baseline45' and result.shortlist_policy == policy
    assert result.planning_goal == (210., 100., 30.)
    assert result.feasible and not result.emergency
    assert result.backup_verified_count == 0
    assert result.states.shape == (21, 16)
    unchanged(instance, before)


def test_ranked_policy_stable_ties_and_coverage_handles_heading_wrap():
    actions = [ReferenceAction(30., np.deg2rad(179.), 1100.),
               ReferenceAction(30., np.deg2rad(-179.), 1100.),
               ReferenceAction(30., 0., 1100.)]
    assert ranked_shortlist(actions, [0., 0., 1.], 2) == [0, 1]
    assert coverage_shortlist(actions, [0., 1., 2.], 2) == [0, 2]
    assert len(set(coverage_shortlist(actions, [0., 1., 2.], 99))) == 3


@pytest.mark.parametrize('selector', [ranked_shortlist, coverage_shortlist])
@pytest.mark.parametrize('costs,budget', [([0.], 0), ([float('nan')], 1), ([], 1), ([0.], True)])
def test_new_shortlists_reject_invalid_costs_and_budgets(selector, costs, budget):
    with pytest.raises(ValueError):
        selector([ReferenceAction(30., 0., 1100.)], costs, budget)


def test_full_grid_shares_evidence_for_same_state_all_shortlist_policies():
    instance = plant()
    before = snapshot(instance)
    report = planning.evaluate_reference_grid(instance, [210., 105., 46.],
        HeightField.flat(260., 200., 90.), [], 0., short_config(candidate_scheme='expanded'))
    assert 45 < report['candidate_count'] <= 69
    assert report['feasible_count'] == report['candidate_count']
    for policy in ('diverse', 'ranked', 'coverage'):
        for budget in ('9', '18', '45'):
            indices = report['policy_shortlist_indices'][policy][budget]
            assert len(set(indices)) == int(budget)
            assert report['policy_shortlist_feasible_counts'][policy][budget] == int(budget)
            assert not report['policy_shortlist_missed_safe_reference'][policy][budget]
    best = min(report['candidates'], key=lambda row: row['cost'])
    assert report['best_feasible_index'] == best['index']
    assert report['planning_goal'] == [210., 105., 46.]
    json.dumps(report, allow_nan=False)
    unchanged(instance, before)


def test_full_grid_identifies_real_feasible_candidates_omitted_by_ranked_screen(monkeypatch):
    instance = plant()
    actual_evaluate = planning._evaluate_reference

    def marked_feasible(index, *args, **kwargs):
        prediction, row = actual_evaluate(index, *args, **kwargs)
        # Unit-level selection fixture: the physical integrations still run.
        # This deliberately defined set is not an experimental safety result.
        row['feasible'] = index == 44
        return prediction, row

    monkeypatch.setattr(planning, '_evaluate_reference', marked_feasible)
    report = planning.evaluate_reference_grid(instance, [210., 100., 30.],
        HeightField.flat(260., 200., 90.), [], 0., short_config())
    for policy in ('diverse', 'ranked', 'coverage'):
        for budget in ('9', '18', '45'):
            count = int(44 in report['policy_shortlist_indices'][policy][budget])
            assert report['policy_shortlist_feasible_counts'][policy][budget] == count
            assert report['policy_shortlist_missed_safe_reference'][policy][budget] == (count == 0)
    assert report['best_feasible_index'] == 44


def test_verified_backup_integrates_every_backup_and_never_claims_regular_feasibility(monkeypatch):
    instance, cfg = plant(), short_config(emergency_policy='verified_backup', candidate_budget=2)
    before = snapshot(instance)
    actual_evaluate = planning._evaluate_reference

    def main_grid_rejected(*args, **kwargs):
        prediction, row = actual_evaluate(*args, **kwargs)
        row['feasible'] = False  # Explicit unit fixture to enter emergency path.
        return prediction, row

    monkeypatch.setattr(planning, '_evaluate_reference', main_grid_rejected)
    goal, terrain = [210., 100., 30.], HeightField.flat(260., 200., 90.)
    result = planning.RemusDWA(cfg).plan(instance, goal, terrain, [], 0.)
    backups = backup_references(instance, goal, cfg)
    assert not result.feasible and result.emergency and result.emergency_verified_safe
    assert result.backup_verified_count == len(backups) <= 8
    assert result.backup_safe_count == len(backups)
    assert result.selected_backup_index is not None
    for row, (label, action) in zip(result.backup_evaluations, backups):
        physical = predict_reference(instance, action, cfg.horizon_s, cfg.physics_dt)
        gap = planning.trajectory_clearance(physical.states, terrain, [], 0., cfg)
        assert row['label'] == label
        assert row['reference'] == action.as_array().tolist()
        assert row['minimum_predicted_clearance_m'] == pytest.approx(gap)
        assert row['feasible'] == (gap >= cfg.safety_margin_m)
    chosen = result.backup_evaluations[result.selected_backup_index]
    np.testing.assert_array_equal(result.command, chosen['reference'])
    assert chosen['cost'] == min(row['cost'] for row in result.backup_evaluations if row['feasible'])
    unchanged(instance, before)


def test_no_safe_backup_preserves_unsafe_coast_physical_inertia_and_actuator_lag():
    instance = plant()
    cfg = short_config(emergency_policy='verified_backup', candidate_budget=2)
    before = snapshot(instance)
    track = Track(5, 10., [0.], [instance.eta[:3].copy()])
    result = planning.RemusDWA(cfg).plan(instance, [210., 100., 30.],
        HeightField.flat(260., 200., 90.), [track], 0.)
    assert not result.feasible and result.emergency and not result.emergency_verified_safe
    assert result.backup_safe_count == 0 and result.backup_verified_count > 1
    assert result.selected_backup_index == 0
    assert result.reason == 'no_verified_feasible_reference_no_safe_backup'
    assert result.action.rpm == 0.
    assert result.states[1, 7] > 1. and result.states[1, 15] > 0.
    assert result.states[-1, 1] > result.states[0, 1]
    unchanged(instance, before)


def test_numeric_backup_failures_are_rejected_recorded_and_do_not_corrupt_plant(monkeypatch):
    instance, cfg = plant(), short_config(emergency_policy='verified_backup', candidate_budget=2)
    before = snapshot(instance)
    actual_predict = planning.predict_reference

    def fail_low_rpm(real_plant, action, *args, **kwargs):
        if action.rpm == 600.:
            raise FloatingPointError('deliberate backup integration failure')
        return actual_predict(real_plant, action, *args, **kwargs)

    monkeypatch.setattr(planning, 'predict_reference', fail_low_rpm)
    track = Track(5, 10., [0.], [instance.eta[:3].copy()])
    result = planning.RemusDWA(cfg).plan(instance, [210., 100., 30.],
        HeightField.flat(260., 200., 90.), [track], 0.)
    failures = [row for row in result.backup_evaluations if row['prediction_failure']]
    assert failures and result.prediction_failure_count == len(failures)
    assert all(not row['feasible'] and row['cost'] is None for row in failures)
    assert result.action.rpm == 0. and result.selected_backup_index == 0
    json.dumps(result.backup_evaluations, allow_nan=False)
    unchanged(instance, before)


def test_default_coast_result_stays_separate_from_optional_backup_search():
    instance = plant()
    track = Track(5, 10., [0.], [instance.eta[:3].copy()])
    result = planning.RemusDWA(short_config(candidate_budget=2)).plan(instance,
        [210., 100., 30.], HeightField.flat(260., 200., 90.), [track], 0.)
    assert result.reason == 'no_verified_feasible_reference_emergency_not_safe'
    assert result.emergency_policy == 'coast'
    assert result.backup_evaluations == () and result.backup_verified_count == 0
    assert result.selected_backup_index is None
