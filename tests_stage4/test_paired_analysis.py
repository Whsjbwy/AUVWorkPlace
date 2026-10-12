"""Paired episode comparisons label state divergence explicitly."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from auv3d.stage4_analysis import compare_actions, load_saved_plans, summarize_actions, summarize_labels


def episode(folder, method, actions, starts):
    folder.mkdir()
    summary = {'backend': 'remus6dof', 'scenario': 'crossing3d', 'seed': 7,
               'risk_method': method, 'risk_metadata': {'alpha': .1}, 'status': 'timeout'}
    (folder/'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    plans = []
    for index, (action, start) in enumerate(zip(actions, starts)):
        states = np.zeros((2, 16))
        states[:, 0] = [0., .2]
        states[:, 1] = start
        plans.append({'time_s': float(index*2), 'action': action,
                      'states': states.tolist(), 'feasible': True})
    (folder/'selected_plans.json').write_text(json.dumps(plans), encoding='utf-8')


def test_common_time_with_different_state_is_not_a_causal_action_comparison(tmp_path):
    episode(tmp_path/'B0', 'B0', [[40, 0, 1100], [40, .1, 1100]], [0., 3.])
    episode(tmp_path/'B2', 'B2', [[40, 0, 800], [40, -.1, 800]], [0., 2.])
    rows = compare_actions('fixed_case', tmp_path/'B0', tmp_path/'B2')
    assert len(rows) == 2 and all(row['action_changed'] for row in rows)
    assert rows[0]['same_recorded_start_state']
    assert not rows[1]['same_recorded_start_state']
    assert 'not a same-controller-state' in rows[1]['interpretation']
    summary = summarize_actions(rows)[0]
    assert summary['changed_action_count'] == 2
    assert summary['changed_action_at_same_recorded_start_state_count'] == 1
    assert summary['first_changed_action_time_s'] == 0.


def test_pairing_rejects_different_seed(tmp_path):
    episode(tmp_path/'B0', 'B0', [[40, 0, 1100]], [0.])
    episode(tmp_path/'B1', 'B1', [[40, 0, 1100]], [0.])
    path = tmp_path/'B1'/'summary.json'
    summary = json.loads(path.read_text(encoding='utf-8'))
    summary['seed'] = 8
    path.write_text(json.dumps(summary), encoding='utf-8')
    with pytest.raises(ValueError, match='same scene'):
        compare_actions('fixed_case', tmp_path/'B0', tmp_path/'B1')


def test_saved_plan_must_begin_at_real_decision_state(tmp_path):
    episode(tmp_path/'B0', 'B0', [[40, 0, 1100]], [0.])
    path = tmp_path/'B0'/'selected_plans.json'
    plans = json.loads(path.read_text(encoding='utf-8'))
    plans[0]['states'][0][0] = .2
    path.write_text(json.dumps(plans), encoding='utf-8')
    with pytest.raises(ValueError, match='offset zero'):
        load_saved_plans(tmp_path/'B0')


def test_offline_rmse_uses_sample_counts_and_does_not_invent_false_rejects():
    frames = [dict(selected_forecast_false_feasible=True, coverage_ball=True,
                   prediction_errors_by_horizon={'3': {'sample_count': 1, 'squared_error_sum_m2': 4.}}),
              dict(selected_forecast_false_feasible=False, coverage_ball=False,
                   prediction_errors_by_horizon={'3': {'sample_count': 3, 'squared_error_sum_m2': 0.}})]
    summary = summarize_labels(frames, 'B1')
    assert summary['prediction_RMSE_by_horizon']['3'] == 1.
    assert summary['calibration_coverage'] == .5
    assert summary['candidate_false_accept_count'] == 1
    assert summary['candidate_false_reject_count'] is None
    assert 'Not all candidates' in summary['candidate_misclassification_scope']


def test_unobserved_sets_and_early_failures_remain_explicit_missing_labels():
    summary = summarize_labels([], 'B2')
    assert summary['evaluated_selected_reference_anchor_count'] == 0
    assert summary['calibration_coverage'] is None
    assert summary['calibration_label_count'] == 0
    assert summarize_labels([dict(selected_forecast_false_feasible=False,
                                  coverage_ecp=True)], 'B0')['calibration_coverage'] is None


@pytest.mark.parametrize('tampered_field', ['action', 'feasible', 'clearance'])
def test_shadow_rejects_tampered_archived_baseline_decision(monkeypatch, tampered_field):
    from auv3d.stage4_analysis import _shadow_frame
    from auv3d import remus_planner
    from auv3d.adapters.remus_guidance import ReferenceAction
    action = ReferenceAction(40., 0., 1100.)
    plan = SimpleNamespace(action=action, feasible=True, minimum_predicted_clearance_m=3.)
    frame = SimpleNamespace(
        plant=SimpleNamespace(eta=np.zeros(6), nu=np.zeros(6), actual=np.zeros(3), time_s=20.),
        cfg=None, previous_action=None, memory_uncertainty={}, goal=np.zeros(3),
        terrain=None, tracks=[], now=20., original_action=action,
        original_feasible=True, original_predicted_clearance_m=3.)
    if tampered_field == 'action':
        frame.original_action = ReferenceAction(40., .01, 1100.)
    elif tampered_field == 'feasible':
        frame.original_feasible = False
    else:
        frame.original_predicted_clearance_m += .01

    class Planner:
        def __init__(self, *args, **kwargs):
            pass
        def set_memory_uncertainty(self, value):
            pass
        def plan(self, *args):
            return plan

    monkeypatch.setattr(remus_planner, 'RemusDWA', Planner)
    with pytest.raises(RuntimeError, match='does not match archived'):
        _shadow_frame(frame, {}, ())


def test_b0_analysis_never_assigns_zero_quantile_as_engineering_coverage(tmp_path, monkeypatch):
    from auv3d import stage4_diagnostics, stage4_risk, remus_planner
    from auv3d import adapters
    from auv3d.stage4_analysis import analyze_studies
    from auv3d.stage4_protocol import manifest, config_for_case
    from auv3d.adapters.remus_guidance import ReferenceAction
    row = manifest()[0]
    cfg = config_for_case(row)
    root = tmp_path/'study'
    source = root/'cases'/row['case_id']
    source.mkdir(parents=True)
    values = {'summary': {'backend': 'remus6dof', 'risk_method': 'B0',
                          'scenario': row['scenario'], 'seed': row['seed'],
                          'status': 'timeout', 'reached_goal': False},
              'config': cfg.to_dict(), 'obstacles': [],
              'planning_diagnostics': [{'feasible': True}]}
    for name, value in values.items():
        (source/f'{name}.json').write_text(json.dumps(value), encoding='utf-8')
    (root/'EXPERIMENT_CONFIG.json').write_text(json.dumps({
        'registered_cases': [row], 'workers': 1,
        'jobs': [{'case_id': row['case_id'], 'method': 'B0', 'alpha': None}]}), encoding='utf-8')
    frame = SimpleNamespace(plant=None, original_action=ReferenceAction(40., 0., 1100.),
                            memory_uncertainty={}, terrain=None, tracks=[], now=20.,
                            original_feasible=True)
    monkeypatch.setattr(stage4_diagnostics, 'replay_anchors', lambda *a, **k: iter([frame]))
    monkeypatch.setattr(stage4_risk, 'RiskPolicy', lambda *a, **k: object())
    monkeypatch.setattr(adapters.remus_guidance, 'predict_reference',
                        lambda *a, **k: SimpleNamespace(states=np.zeros((2, 16))))
    monkeypatch.setattr(remus_planner, 'trajectory_clearance', lambda *a, **k: 3.)
    observed = []

    class Truth:
        def __init__(self, *args, **kwargs):
            pass
        def evaluate(self, *args, **kwargs):
            observed.append(kwargs['quantile_m'])
            return {'selected_forecast_false_feasible': False,
                    'coverage_ecp': None, 'coverage_ball': None}

    monkeypatch.setattr(stage4_diagnostics, 'OfflineTruthReplay', Truth)
    library = tmp_path/'library.npz'
    library.write_bytes(b'library loading is isolated by this wiring test')
    output = tmp_path/'analysis'
    analyze_studies([root], library, output)
    assert observed == [None]
    result = json.loads((output/'OFFLINE_EPISODE_METRICS.json').read_text(encoding='utf-8'))[0]
    assert result['calibration_coverage'] is None
    assert 'not represented by a zero' in result['calibration_coverage_scope']
