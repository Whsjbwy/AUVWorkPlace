"""Historical numerical baseline stays explicit and separate from main A0-A5."""
import pytest

from scripts_stage5 import historical_frozen as historical
from auv3d.stage5_protocol import config_for_case


def test_exact_two_registered_scenes_four_original_methods_without_role_change():
    jobs = historical.registered_jobs()
    assert len(jobs) == 4 and len({j['job_id'] for j in jobs}) == 4
    assert {j['case']['scene_id'] for j in jobs} == set(historical.SCENE_IDS)
    assert {j['historical_method'] for j in jobs} == {'H4_B1_original','H4_B2_original'}
    for job in jobs:
        assert job['case']['split'] == 'test' and job['case']['phase'] == 'formal'
        cfg = config_for_case(job['case'])
        assert cfg.sensor_noise_std_m == .02 and cfg.safety_margin_m == 1.
        assert cfg.candidate_budget == 9 and cfg.prediction_history_dt_s == 1.


@pytest.mark.parametrize('method',['B1','B2'])
def test_original_library_retains_original_24_scene_protocol_and_numerical_ball(method):
    policy = historical.make_original_policy(method)
    assert policy.method == method and policy.alpha == .1
    assert policy.metadata['calibration_scene_count'] == 24
    assert policy.metadata['empty_calibration_scene_count'] == 8
    assert policy.metadata['original_engineering_envelope_retained']
    assert policy.ball_q == pytest.approx(63.0408,abs=.0001)


def test_historical_metrics_do_not_turn_arrival_into_actual_safety():
    job = historical.registered_jobs()[0]
    row = historical.metrics(job,dict(program_error=False,summary=dict(
        status='success',reached_goal=True,actual_safety_margin_met=False)),1)
    assert row['navigation_success'] and not row['safe_arrival']
    assert row['actual_clearance_margin_violation'] and row['combination']=='original_add'
    assert row['historical_method']=='H4_B1_original'
    unknown = historical.metrics(job,dict(program_error=True),1)
    assert unknown['safe_arrival'] is None and unknown['collision'] is None


def test_existing_results_preserved_and_worker_count_bounded(tmp_path):
    (tmp_path/'kept.txt').write_text('kept',encoding='utf8')
    with pytest.raises(FileExistsError):
        historical.run(tmp_path)
    assert (tmp_path/'kept.txt').read_text(encoding='utf8')=='kept'
    with pytest.raises(ValueError,match='workers'):
        historical.run(tmp_path/'new',workers=3)
    with pytest.raises(ValueError):
        historical.make_original_policy('A4')
