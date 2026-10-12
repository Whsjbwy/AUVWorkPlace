"""Stage 4 registration preserves failures and separates research outcomes."""
from dataclasses import replace
import json

import numpy as np
import pytest

from auv3d.stage2_config import Stage2Config
from auv3d import stage4_experiments as experiment


def case(case_id='crossing_001', split='test', **extra):
    return dict(case_id=case_id, split=split, family='crossing', scenario='crossing3d',
                seed=107, config_overrides={}, **extra)


@pytest.fixture(autouse=True)
def fixed_configuration(monkeypatch):
    from auv3d import stage4_protocol
    monkeypatch.setattr(stage4_protocol, 'config_for_case',
                        lambda row: replace(Stage2Config(), seed=row['seed']))


def test_arrival_is_not_safety_and_missing_diagnostics_stay_missing():
    job = experiment.ExperimentJob(case())
    row = experiment.episode_metrics(job, {'planning_cycle_count': 4, 'summary': {
        'status': 'success', 'reached_goal': True, 'actual_safety_margin_met': False,
        'unsafe_emergency_forecast_count': 2, 'avoidance_acceptance_passed': False,
        'planning_exceeds_control_period_count': 1}}, 2)
    assert row['navigation_success'] is True
    assert row['actual_clearance_margin_violation'] is True
    assert row['unsafe_emergency_forecast'] is True
    assert row['avoidance_acceptance_passed'] is False
    assert row['planning_deadline_miss_fraction'] == .25
    assert row['candidate_false_accept_count'] is None
    assert row['calibration_coverage'] is None
    assert row['prediction_RMSE_by_horizon'] is None
    assert 'not execution delay' in row['timing_interpretation']


def test_program_error_is_not_a_navigation_success_or_inferred_collision():
    row = experiment.episode_metrics(experiment.ExperimentJob(case()),
                                     {'program_error': True}, 1)
    assert row['status'] == 'program_error' and row['program_error']
    assert row['navigation_success'] is None and row['collision'] is None


@pytest.mark.parametrize('case_id', ['../raw', 'a/b', 'a\\b', '.', '..', ''])
def test_case_identifiers_cannot_escape_output_directory(case_id):
    with pytest.raises(ValueError, match='case_id'):
        experiment.select_cases([case(case_id)])


def test_unknown_case_is_not_silently_discarded_and_split_filter_is_explicit():
    rows = [case('a', 'train'), case('b', 'test')]
    assert experiment.select_cases(rows, splits=['test'])[0]['case_id'] == 'b'
    with pytest.raises(ValueError, match='unknown case'):
        experiment.select_cases(rows, case_ids=['typo'])
    with pytest.raises(ValueError, match='no cases'):
        experiment.select_cases(rows, splits=['calibration'])


@pytest.mark.parametrize('method,alpha,path', [('B3', None, None), ('B0', .1, None),
                                            ('B1', None, 'x'), ('B1', 0., 'x'),
                                            ('B2', 1., 'x'), ('B2', .1, None)])
def test_risk_jobs_require_explicit_valid_method_and_artifact(method, alpha, path):
    with pytest.raises(ValueError):
        experiment.ExperimentJob(case(), method, alpha, path).validate()


def test_grouped_rates_preserve_failures_and_do_not_pool_splits_or_risk_levels():
    rows = []
    for method, alpha, split, status in [('B0', None, 'test', 'success'),
                                        ('B1', .1, 'test', 'timeout'),
                                        ('B1', .2, 'test', 'success'),
                                        ('B1', .1, 'tune', 'success')]:
        job = experiment.ExperimentJob(case(split=split), method, alpha, 'fixture' if alpha else None)
        rows.append(experiment.episode_metrics(job, {'summary': {
            'status': status, 'reached_goal': status == 'success'}}, 1))
    groups = experiment.grouped_results(rows)
    assert len(groups) == 4
    failed = next(row for row in groups if row['split'] == 'test'
                  and row['risk_method'] == 'B1' and row['risk_alpha'] == .1)
    assert failed['navigation_success_rate'] == 0.
    assert failed['timeout_rate'] == 1.
    assert failed['measured_runs'] == 1


def test_existing_raw_failure_is_never_overwritten(tmp_path):
    marker = tmp_path/'failure.txt'
    marker.write_text('keep original failed run', encoding='utf-8')
    with pytest.raises(FileExistsError, match='preserved'):
        experiment.run_study(tmp_path, cases=[case()])
    assert marker.read_text(encoding='utf-8') == 'keep original failed run'


def test_registration_precedes_execution_and_all_registered_configs_are_kept(tmp_path, monkeypatch):
    selected = []

    def fake_execute(job, out, plot, animate):
        registered = json.loads((out/'EXPERIMENT_CONFIG.json').read_text(encoding='utf-8'))
        assert registered['registered_case_count'] == 2
        assert registered['selected_job_count'] == 1
        assert len(list((out/'registered_configs').glob('*.json'))) == 2
        selected.append(job.case['case_id'])
        return {'program_error': False, 'summary': {'status': 'timeout', 'reached_goal': False}}

    monkeypatch.setattr(experiment, '_execute_job', fake_execute)
    out = tmp_path/'new'
    result = experiment.run_study(out, cases=[case('a'), case('b')], limit=1, plot=False)
    assert selected == ['a'] and result['program_error_count'] == 0
    assert result['mixed_overall_success_rate'] is None
    assert (out/'EXPERIMENT_SUMMARY.csv').is_file()
    assert (out/'SAFETY_EFFICIENCY_CURVES.csv').is_file()


def test_json_records_reject_nonfinite_measurements(tmp_path):
    with pytest.raises(ValueError, match='nonfinite'):
        experiment.write_json(tmp_path/'bad.json', {'planning_ms': np.inf})


def test_runtime_exception_keeps_traceback_and_does_not_claim_success(tmp_path, monkeypatch):
    from auv3d import stage2_simulation

    def failed(*args, **kwargs):
        raise FloatingPointError('synthetic numerical failure')

    monkeypatch.setattr(stage2_simulation, 'run_remus_episode', failed)
    result = experiment.run_study(tmp_path/'new', cases=[case()], plot=False)
    assert result['program_error_count'] == 1
    saved = json.loads((tmp_path/'new'/'cases'/'crossing_001'/'program_error.json')
                       .read_text(encoding='utf-8'))
    assert saved['error_type'] == 'FloatingPointError'
    assert 'synthetic numerical failure' in saved['traceback']


def saved_baseline(root, row):
    from auv3d.stage4_protocol import config_for_case
    folder = root/'cases'/row['case_id']
    folder.mkdir(parents=True)
    values = {'config': config_for_case(row).to_dict(),
              'summary': {'backend': 'remus6dof', 'risk_method': 'B0', 'seed': row['seed'],
                          'scenario': row['scenario'], 'status': 'timeout',
                          'reached_goal': False},
              'planning_diagnostics': [{'feasible': True}, {'feasible': False}],
              'observations': [], 'obstacles': []}
    for name, value in values.items():
        experiment.write_json(folder/f'{name}.json', value)
    for name in ('planning.csv', 'states.csv'):
        (folder/name).write_text('time_s\n0.0\n', encoding='utf-8')
    experiment.write_json(root/'EXPERIMENT_CONFIG.json', {'workers': 2})
    return folder


def test_completed_baseline_reuse_compares_parameters_and_preserves_failure(tmp_path):
    row = case()
    source = saved_baseline(tmp_path, row)
    record = experiment.load_baseline_record(row, tmp_path)
    assert record['reused_read_only_B0']
    assert record['summary']['status'] == 'timeout'
    assert record['no_feasible_candidate_count'] == 1
    cfg = json.loads((source/'config.json').read_text(encoding='utf-8'))
    cfg['candidate_budget'] = 18
    (source/'config.json').write_text(json.dumps(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='candidate_budget'):
        experiment.load_baseline_record(row, tmp_path)


def test_paired_reuse_does_not_run_baseline_again(tmp_path, monkeypatch):
    row = case()
    root = tmp_path/'baseline'
    saved_baseline(root, row)
    artifact = tmp_path/'library.npz'
    artifact.write_bytes(b'fixture; worker does not load it in this unit test')
    executed = []

    def fake_execute(job, *args):
        executed.append(job.method)
        return {'program_error': False, 'summary': {'status': 'timeout', 'reached_goal': False}}

    monkeypatch.setattr(experiment, '_execute_job', fake_execute)
    result = experiment.run_study(tmp_path/'paired', cases=[row], methods=['B1', 'B2'],
                                  alphas=[.1, .2], calibration_path=artifact, baseline_root=root)
    assert executed == ['B1', 'B1', 'B2', 'B2']
    assert result['executed_job_count'] == 4
    assert result['read_only_reused_B0_count'] == 1
    groups = json.loads((tmp_path/'paired'/'GROUPED_SUMMARY.json').read_text(encoding='utf-8'))
    assert len(groups) == 5
    original = next(group for group in groups if group['risk_method'] == 'B0')
    assert original['timeout_rate'] == 1.


def test_existing_summary_does_not_hide_partial_save_or_render_error(tmp_path):
    row = case()
    source = saved_baseline(tmp_path, row)
    experiment.write_json(source/'program_error.json', {'program_error': True,
                                                       'error': 'render failed after raw save'})
    with pytest.raises(ValueError, match='recorded program error'):
        experiment.load_baseline_record(row, tmp_path)
