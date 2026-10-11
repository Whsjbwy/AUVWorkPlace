"""Experiment registration and failure reporting, independent of safety scores."""
from dataclasses import replace
import json

import pytest

from auv3d.stage2_config import Stage2Config
from scripts_stage3 import run_experiments as experiment


def test_fixed_matrices_keep_each_setting_in_separate_group():
    assert len(experiment.build_cases('smoke')) == 3
    assert len(experiment.build_cases('memory')) == 15
    assert len(experiment.build_cases('candidate')) == 16
    cases = experiment.build_cases('all', (7, 17))
    assert len({case.case_id for case in cases}) == len(cases)
    groups = {}
    for case in cases:
        cfg = case.cfg.to_dict()
        cfg.pop('seed')
        if case.group_id in groups:
            assert cfg == groups[case.group_id]
        groups[case.group_id] = cfg
        assert case.cfg.physics_dt == .02
        assert case.cfg.safety_margin_m == 1.
        assert case.cfg.goal_tolerance_m == 5.


def test_memory_matrix_crosses_all_five_policies_and_all_three_sensor_conditions():
    cases = experiment.build_cases('memory')
    for sensor_name, hfov, vfov, interval in experiment.SENSOR_VARIANTS:
        matches = [case for case in cases if sensor_name in case.group_id]
        assert len(matches) == 5
        assert {(case.cfg.memory_policy, case.cfg.track_max_age_s) for case in matches} == {
            ('hard_expiry', 4.), ('hard_expiry', 8.), ('hard_expiry', 12.),
            ('retain_uncertain', 4.), ('conditional', 4.)}
        for case in matches:
            assert case.scenario == 'multi_crossing3d'
            assert case.cfg.obstacle_speed_scale == 1.5
            assert (case.cfg.sensor_hfov_deg, case.cfg.sensor_vfov_deg,
                    case.cfg.observation_period_s) == (hfov, vfov, interval)


def test_candidate_matrix_preserves_reference_grid_budget_and_backup_controls():
    for scenario in ('static3d', 'multi_crossing3d'):
        cases = [case for case in experiment.build_cases('candidate') if case.scenario == scenario]
        assert len(cases) == 8
        assert {(case.cfg.candidate_scheme, case.cfg.shortlist_policy, case.cfg.candidate_budget,
                 case.cfg.emergency_policy) for case in cases} == {
            ('baseline45', 'diverse', 9, 'coast'), ('baseline45', 'diverse', 18, 'coast'),
            ('baseline45', 'diverse', 45, 'coast'), ('baseline45', 'ranked', 9, 'coast'),
            ('baseline45', 'coverage', 9, 'coast'), ('expanded', 'coverage', 9, 'coast'),
            ('expanded', 'coverage', 45, 'coast'), ('baseline45', 'diverse', 9, 'verified_backup')}
        for case in cases:
            assert case.cfg.heading_offsets_deg == Stage2Config().heading_offsets_deg
            assert case.cfg.depth_offsets_m == Stage2Config().depth_offsets_m
            assert case.cfg.rpm_values == Stage2Config().rpm_values


def test_predictor_pairs_share_all_parameters_except_predictor_name():
    cases = experiment.build_cases('predictor')
    assert len(cases) == 8
    for scenario in ('static3d', 'crossing3d', 'multi_crossing3d', 'maneuvering3d'):
        pair = [case for case in cases if case.scenario == scenario]
        assert {case.cfg.obstacle_predictor for case in pair} == {'ca', 'pplnm'}
        first, second = [case.cfg.to_dict() for case in pair]
        first.pop('obstacle_predictor')
        second.pop('obstacle_predictor')
        assert first == second
        for case in pair:
            assert case.cfg.prediction_history_dt_s == 3.
            assert case.cfg.pplnm_extension_mode == 'cv_after9'
            assert case.cfg.pplnm_threads == 2
            assert case.cfg.obstacle_speed_scale == 1.
            assert case.cfg.max_time_s == 150.


@pytest.mark.parametrize('seeds', [(), (7, 7), (-1,), (True,)])
def test_invalid_seed_registration_rejected(seeds):
    with pytest.raises(ValueError, match='seeds'):
        experiment.build_cases('smoke', seeds)


def test_existing_experiment_directory_is_preserved(tmp_path, monkeypatch):
    marker = tmp_path/'raw_failure.txt'
    marker.write_text('keep this failure', encoding='utf-8')

    def must_not_execute(*args, **kwargs):
        pytest.fail('existing output must be rejected before simulation')

    monkeypatch.setattr(experiment, '_execute_case', must_not_execute)
    with pytest.raises(FileExistsError, match='preserved'):
        experiment.run_study(tmp_path, limit=1)
    assert marker.read_text(encoding='utf-8') == 'keep this failure'


def test_full_matrix_registration_precedes_limited_execution_and_arrival_is_not_safety(tmp_path, monkeypatch):
    out = tmp_path/'new_study'
    case_ids = []

    def run_case(case, output, *args):
        registration = json.loads((output/'EXPERIMENT_CONFIG.json').read_text(encoding='utf-8'))
        assert registration['registered_case_count'] == 3
        assert registration['selected_case_count'] == 1
        assert registration['common_physics_dt_s'] == .02
        assert registration['physics_dt_values_s'] == [.02]
        assert len(list((output/'registered_configs').glob('*.json'))) == 3
        case_ids.append(case.case_id)
        return {'case_id': case.case_id, 'program_error': False, 'planning_cycle_count': 3,
                'summary': {'status': 'success', 'reached_goal': True,
                            'actual_safety_margin_met': True, 'unsafe_emergency_forecast_count': 1,
                            'avoidance_acceptance_passed': False,
                            'planning_exceeds_control_period_count': 2}}

    monkeypatch.setattr(experiment, '_execute_case', run_case)
    result = experiment.run_study(out, limit=1)
    assert len(case_ids) == 1 and result['program_error_count'] == 0
    assert result['mixed_overall_navigation_success_rate'] is None
    groups = json.loads((out/'GROUPED_SUMMARY.json').read_text(encoding='utf-8'))
    assert groups[0]['navigation_success_rate'] == 1.
    assert groups[0]['finite_horizon_acceptance_rate'] == 0.
    row = experiment.metric_row(experiment.build_cases()[0],
                               json.loads((out/'EXPERIMENT_RECORDS.json').read_text(encoding='utf-8'))[0], 1)
    assert row['navigation_success'] is True and row['unsafe_emergency_forecast'] is True
    assert row['planning_deadline_miss'] == 2 and row['planning_deadline_miss_fraction'] == 2/3
    assert row['prediction_RMSE_by_horizon'] == 'NaN'
    assert 'pending_offline' in row['prediction_RMSE_status']


def test_program_exception_keeps_traceback_and_is_not_navigation_failure(tmp_path, monkeypatch):
    import auv3d.stage2_simulation as simulation

    def failed_program(*args, **kwargs):
        raise FloatingPointError('deliberate nonfinite state fixture')

    monkeypatch.setattr(simulation, 'run_remus_episode', failed_program)
    case = experiment.build_cases()[0]
    result = experiment.run_study(tmp_path/'error_study', cases=[case], plot=False)
    assert result['program_error_count'] == 1
    saved = json.loads((tmp_path/'error_study'/'cases'/case.case_id/'program_error.json').read_text(encoding='utf-8'))
    assert saved['error_type'] == 'FloatingPointError'
    assert 'deliberate nonfinite state fixture' in saved['traceback']
    metric = experiment.metric_row(case, saved, 1)
    assert metric['program_error'] and metric['status'] == 'program_error'
    assert metric['navigation_success'] is None and metric['collision'] is None


def test_grouped_statistics_do_not_mix_candidate_budgets():
    cases = experiment.build_cases('candidate')[:3]
    rows = [experiment.metric_row(case, {'summary': {'status': 'success', 'reached_goal': True,
             'actual_safety_margin_met': True, 'avoidance_acceptance_passed': False}}, 2) for case in cases]
    result = experiment.grouped_summary(rows)
    assert len(result) == 3
    assert all(group['case_count'] == 1 for group in result)
    assert all(group['navigation_success_rate'] == 1. and group['finite_horizon_acceptance_rate'] == 0.
               for group in result)


def test_custom_registration_cannot_mix_different_budgets_in_one_group(tmp_path):
    case = experiment.build_cases()[0]
    changed = replace(case, case_id=case.case_id+'_different_budget',
                      cfg=replace(case.cfg, candidate_budget=18))
    with pytest.raises(ValueError, match='mixes configurations'):
        experiment.run_study(tmp_path/'invalid_study', cases=[case, changed])
    assert not (tmp_path/'invalid_study').exists()


def test_common_study_cannot_silently_change_original_safety_or_arrival_criterion(tmp_path):
    case = experiment.build_cases()[0]
    for name, value in [('safety_margin_m', .5), ('body_radius_m', .4), ('goal_tolerance_m', 8.)]:
        changed = replace(case, cfg=replace(case.cfg, **{name: value}))
        with pytest.raises(ValueError, match=f'retain original {name}'):
            experiment.run_study(tmp_path/name, cases=[changed])
        assert not (tmp_path/name).exists()


@pytest.mark.parametrize('steps,expected_common', [((.01,), .01), ((.01, .02), None)])
def test_registration_physics_metadata_comes_from_all_actual_configs(tmp_path, monkeypatch,
                                                                     steps, expected_common):
    baseline = experiment.build_cases()[0]
    cases = [replace(baseline, case_id=f'custom_physics_{step}', group_id=f'custom_physics_{step}',
                     cfg=replace(baseline.cfg, physics_dt=step)) for step in steps]

    def run_case(case, output, *args):
        return {'case_id': case.case_id, 'program_error': False, 'planning_cycle_count': 1,
                'summary': {'status': 'timeout', 'reached_goal': False,
                            'actual_safety_margin_met': True, 'avoidance_acceptance_passed': False}}

    monkeypatch.setattr(experiment, '_execute_case', run_case)
    out = tmp_path/'physics_metadata'
    experiment.run_study(out, cases=cases, limit=1)
    registration = json.loads((out/'EXPERIMENT_CONFIG.json').read_text(encoding='utf-8'))
    # Even a limited execution must describe the complete predeclared matrix.
    assert registration['common_physics_dt_s'] == expected_common
    assert registration['physics_dt_values_s'] == sorted(set(steps))
    assert registration['registered_case_count'] == len(steps)
    assert registration['selected_case_count'] == 1
