"""Review regressions: diagnostics isolation, causal logs and physical delay."""
from dataclasses import replace
import json

import numpy as np
import pytest

from auv3d.stage2_config import Stage2Config
from auv3d.stage2_simulation import run_remus_episode, save_remus_episode


def config(**kwargs):
    return replace(Stage2Config(), warmup_s=0., max_time_s=3.,
                   control_dt=1., horizon_s=2., candidate_budget=1, **kwargs)


def test_diagnostic_hook_cannot_modify_live_vehicle_tracks_or_scene():
    expected = run_remus_episode(config(), 'static3d')
    calls = []
    def corrupt(plant, tracks, now, plan, scene):
        calls.append(now)
        plant.eta[:] = -1000
        plant.vehicle.e_int = 999
        if tracks:
            tracks[0].positions[-1][:] = 999
        plan.states[:] = -999
        scene.goal[:] = -999
        scene.obstacles.step(200.)
    actual = run_remus_episode(config(), 'static3d', on_plan=corrupt)
    assert calls == [0., 1., 2.]
    for name in ('states', 'controls', 'prediction_errors', 'clearance'):
        np.testing.assert_array_equal(actual[name], expected[name])
    np.testing.assert_array_equal(actual['goal'], expected['goal'])


def test_sensor_packets_and_tracks_keep_causal_measurement_times(tmp_path):
    result = run_remus_episode(config(observation_delay_s=1.), 'static3d')
    packets = result['observations']
    assert packets[0]['measurement_time_s'] == 0.
    assert packets[0]['delivered_time_s'] == 1.
    assert packets[-1]['delivered_time_s'] is None
    for row in result['planning_diagnostics']:
        assert all(max(t['times_s']) <= row['time_s']-1.+1e-9 for t in row['tracks'])
    save_remus_episode(result, tmp_path/'run', plot=False)
    assert json.loads((tmp_path/'run/observations.json').read_text()) == packets
    clear = np.loadtxt(tmp_path/'run/clearance.csv', delimiter=',', skiprows=1)
    np.testing.assert_allclose(clear[:, 2], clear[:, 1]-config().safety_margin_m)
    np.testing.assert_allclose(clear[:, 0], result['states'][1:, 0])


def test_fixed_command_delay_advances_actual_plant_with_previous_reference():
    immediate = run_remus_episode(config(), 'empty3d')
    delayed = run_remus_episode(config(command_delay_s=.5), 'empty3d')
    # Before the first application the original nominal RPM is still active.
    np.testing.assert_array_equal(delayed['controls'][:25, 3], 1100.)
    assert immediate['controls'][0, 3] != delayed['controls'][0, 3]
    applications = [e for e in delayed['events'] if e['kind']=='delayed_command_applied']
    np.testing.assert_allclose([e['time_s'] for e in applications], [.5, 1.5, 2.5])
    assert delayed['summary']['command_delay_s'] == .5
    assert delayed['summary']['prediction_position_max_error_m'] > .001
    np.testing.assert_allclose(delayed['states'][:, 0], immediate['states'][:, 0])
    assert delayed['prediction_errors'][0, 0] == pytest.approx(.52)
    assert delayed['prediction_errors'][0, 1] == 0.
    assert delayed['selected_plans'][0]['command_application_time_s'] == .5


@pytest.mark.parametrize('changes', [{'max_time_s':.03}, {'command_delay_s':.03},
                                    {'command_delay_s':-.02}, {'command_delay_s':float('nan')},
                                    {'command_delay_s':24.}])
def test_integration_time_settings_reject_silent_rounding(changes):
    with pytest.raises(ValueError):
        replace(Stage2Config(), **changes).validate()


def test_p25_does_not_silently_ignore_remus_only_delay_setting():
    from auv3d.stage2_p25 import run_p25_episode
    with pytest.raises(ValueError, match='only for the REMUS'):
        run_p25_episode(config(command_delay_s=.5), 'empty3d')
