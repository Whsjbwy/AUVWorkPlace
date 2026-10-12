"""Explicit stage 4 commands keep model fitting and independent testing apart."""
import json

import pytest

import run_stage4


def test_manifest_command_prints_all_split_assignments(capsys):
    assert run_stage4.main(['manifest']) == 0
    record = json.loads(capsys.readouterr().out)
    assert len(record['cases']) == 48
    assert {row['split'] for row in record['cases']} == {'train', 'tune', 'calibration', 'test'}


def test_build_calibration_has_no_test_split_override():
    with pytest.raises(SystemExit) as error:
        run_stage4.main(['build-calibration', '--baseline-root', 'baseline',
                         '--out', 'library.npz', '--split', 'test'])
    assert error.value.code == 2


def test_paired_cli_uses_one_shared_alpha_and_reuses_baseline(tmp_path, monkeypatch):
    from auv3d import stage4_experiments
    called = {}

    def run(output, **kwargs):
        called.update(output=output, **kwargs)
        return {'program_error_count': 0}

    monkeypatch.setattr(stage4_experiments, 'run_study', run)
    assert run_stage4.main(['run-paired', '--split', 'test', '--alpha', '.1',
                            '--library', str(tmp_path/'library.npz'),
                            '--baseline-root', str(tmp_path/'baseline'),
                            '--out', str(tmp_path/'paired')]) == 0
    assert called['methods'] == ['B1', 'B2']
    assert called['alphas'] == [.1]
    assert called['splits'] == ['test']
    assert called['baseline_root'] == str(tmp_path/'baseline')


def test_missing_offline_library_is_an_error_before_any_analysis(tmp_path):
    with pytest.raises(FileNotFoundError, match='library'):
        run_stage4.main(['analyze', '--root', str(tmp_path/'raw'),
                         '--library', str(tmp_path/'missing.npz'), '--out', str(tmp_path/'analysis')])
    assert not (tmp_path/'analysis').exists()
