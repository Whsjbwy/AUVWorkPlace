"""Serial planning latency consumes physical time under the previous reference."""
from dataclasses import replace
import json

import numpy as np
import pytest

from auv3d.stage2_config import Stage2Config
from auv3d.stage2_simulation import run_remus_episode, save_remus_episode


def config(**overrides):
    return replace(Stage2Config(), warmup_s=0., max_time_s=3., control_dt=1.,
                   horizon_s=2., candidate_budget=1, **overrides)


def test_fixed_computation_holds_old_reference_and_keeps_obstacles_and_sensing_alive():
    result = run_remus_episode(config(), 'crossing3d', planning_delay_mode='fixed',
                               planning_delay_s=.5)
    np.testing.assert_array_equal(result['controls'][:25, 3], 1100.)
    assert result['controls'][25, 3] != 1100.
    applications = [e for e in result['events'] if e['kind'] == 'delayed_command_applied']
    np.testing.assert_allclose([e['time_s'] for e in applications], [.5, 1.5, 2.5])
    np.testing.assert_allclose([e['actual_total_delay_s'] for e in applications], .5)
    assert [p['measurement_time_s'] for p in result['observations']] == [0., 1., 2.]
    assert result['obstacles'][-1]['time_s'] == result['states'][-1, 0] == 3.
    assert result['obstacles'][-1]['positions'] != result['obstacles'][0]['positions']
    assert result['summary']['planning_computation_delay_max_s'] == .5
    assert result['summary']['prediction_position_max_error_m'] > .001
    assert 'latency under the previous reference' in result['summary']['prediction_error_scope']


def test_measured_latency_rounds_up_and_serial_worker_cannot_overlap(monkeypatch, tmp_path):
    import auv3d.stage2_simulation as module
    original = module.RemusDWA.plan
    def measured(self, *args):
        return replace(original(self, *args), planning_ms=1251.)
    monkeypatch.setattr(module.RemusDWA, 'plan', measured)
    cfg = config(observation_period_s=.2)
    result = run_remus_episode(cfg, 'crossing3d', planning_delay_mode='measured')
    np.testing.assert_allclose(result['planning'][:, 0], [0., 1.26, 2.52])
    np.testing.assert_array_equal(result['controls'][:63, 3], 1100.)
    applications = [e for e in result['events'] if e['kind'] == 'delayed_command_applied']
    np.testing.assert_allclose([e['time_s'] for e in applications], [1.26, 2.52])
    assert result['planning_diagnostics'][1]['tracks'][0]['times_s'][-1] == pytest.approx(1.2)
    assert result['planning_diagnostics'][0]['tracks'][0]['times_s'] == [0.]
    assert result['summary']['planning_exceeds_control_period_count'] == 3
    save_remus_episode(result, tmp_path/'measured', plot=False)
    saved = json.loads((tmp_path/'measured/selected_plans.json').read_text())
    assert saved[0]['command_application_time_s'] == 1.26
    assert saved[2]['command_application_time_s'] > result['states'][-1, 0]


def test_computation_delay_adds_transport_delay_without_early_application():
    result = run_remus_episode(config(command_delay_s=.2), 'empty3d',
                               planning_delay_mode='fixed', planning_delay_s=.201)
    # .201 rounds up to .22; .20 transport delay is additional.
    np.testing.assert_array_equal(result['controls'][:21, 3], 1100.)
    assert result['controls'][21, 3] != 1100.
    assert result['selected_plans'][0]['command_application_time_s'] == pytest.approx(.42)


@pytest.mark.parametrize('mode,seconds', [('unknown', 0), ('fixed', -1),
                                        ('fixed', float('nan')), ('none', .1),
                                        ('measured', .1)])
def test_invalid_latency_configuration_is_not_silently_ignored(mode, seconds):
    with pytest.raises(ValueError):
        run_remus_episode(config(), planning_delay_mode=mode, planning_delay_s=seconds)


def test_none_mode_retains_legacy_trajectory_exactly():
    plain = run_remus_episode(config(), 'static3d')
    explicit = run_remus_episode(config(), 'static3d', planning_delay_mode='none')
    for field in ('states', 'controls', 'clearance', 'prediction_errors'):
        np.testing.assert_array_equal(plain[field], explicit[field])
    assert 'matching deterministic original plant' in explicit['summary']['prediction_error_scope']
