"""Shared stage 2 P25 backend remains causal and explicitly kinematic."""
from dataclasses import replace
import json
import numpy as np
import pytest
from matplotlib.figure import Figure

from auv3d.scenarios import Scene, make_stage2_scenario
from auv3d.stage2_config import Stage2Config
from auv3d.stage2_p25 import DWA3D, pitch_legal_commands, p25_config, run_p25_episode
from auv3d.simulation import save_episode
from auv3d.obstacles import LinearObstacles
from auv3d.terrain import HeightField
from auv3d.planner import DWA3D as OriginalDWA3D
from auv3d.sensors import Track


def tiny(**kwargs):
    return replace(Stage2Config(), max_time_s=1., control_dt=.5, horizon_s=2.,
                   observation_period_s=.5, warmup_s=0., candidate_budget=1, **kwargs)


def test_shared_scene_truth_pose_goal_sensor_and_physical_time_are_used(monkeypatch):
    import auv3d.stage2_p25 as module
    cfg = tiny()
    expected = make_stage2_scenario('maneuvering3d', cfg)
    observed_pose = []
    original = module.IdealObjectSonar.observe

    def record(sensor, eta, stamp, snapshot, terrain):
        observed_pose.append((np.asarray(eta).copy(), stamp, sensor.cfg, terrain))
        return original(sensor, eta, stamp, snapshot, terrain)

    monkeypatch.setattr(module.IdealObjectSonar, 'observe', record)
    result = run_p25_episode(cfg, 'maneuvering3d')
    np.testing.assert_array_equal(result['states'][0, 1:4], expected.start_eta[:3])
    np.testing.assert_array_equal(result['goal'], expected.goal)
    np.testing.assert_array_equal(result['terrain'].depths, expected.terrain.depths)
    assert result['summary']['scene_metadata']['method_independent_scene'] is True
    assert [s['time_s'] for s in result['obstacles']] == [0., .5, 1.]
    np.testing.assert_allclose(np.diff(result['states'][:, 0]), cfg.physics_dt)
    expected.obstacles.step(.5)
    np.testing.assert_allclose(result['obstacles'][1]['positions'], expected.obstacles.snapshot().positions, atol=1e-10)
    expected.obstacles.step(.5)
    np.testing.assert_allclose(result['obstacles'][-1]['positions'], expected.obstacles.snapshot().positions, atol=1e-10)
    assert result['obstacles'][-1]['time_s'] == result['states'][-1, 0]
    assert all(pose.shape == (6,) and pose[3] == 0. and sensor == cfg.sensor_config()
               for pose, _, sensor, _ in observed_pose)
    assert result['summary']['maximum_observed_obstacles'] == 3


def test_delay_preserves_measurement_timestamps_and_hides_undelivered_truth(monkeypatch):
    import auv3d.stage2_p25 as module
    received = []
    original = module.DWA3D.plan

    def record(planner, state, goal, terrain, tracks, now):
        received.append((now, [list(track.times) for track in tracks]))
        assert all(track.times[-1] <= now-.5+1e-9 for track in tracks)
        return original(planner, state, goal, terrain, tracks, now)

    monkeypatch.setattr(module.DWA3D, 'plan', record)
    result = run_p25_episode(tiny(observation_delay_s=.5), 'static3d')
    assert received[0] == (0., [])
    assert received[1] == (.5, [[0.]])
    assert result['summary']['maximum_used_observation_age_s'] == pytest.approx(.5)
    assert result['summary']['maximum_tracked_obstacles'] == 1


def test_original_save_and_plot_format_cannot_be_mistaken_for_remus(tmp_path, monkeypatch):
    captured = []
    original = Figure.savefig

    def record(fig, path, *args, **kwargs):
        captured.append(fig)
        original(fig, path, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', record)
    result = run_p25_episode(tiny(), 'crossing3d')
    assert result['states'].shape == (51, 10)
    assert result['planning'].shape == (2, 7)
    assert np.asarray(result['selected_plans'][0]['states']).shape[1] == 6
    assert result['summary']['backend'] == 'p25_kinematic_3d_rk4'
    assert result['summary']['remus6dof_avoidance_claim'] is False
    assert 'different vehicle dynamics' in result['summary']['comparison_scope']
    assert result['summary']['status'] == 'timeout'
    assert not result['summary']['reached_goal']
    assert result['summary']['failure_reason']
    save_episode(result, tmp_path/'run')
    summary = json.loads((tmp_path/'run/summary.json').read_text())
    assert summary['remus6dof_avoidance_claim'] is False
    assert (tmp_path/'run/trajectory.csv').is_file()
    assert (tmp_path/'run/trajectory_3d.png').stat().st_size > 1000
    assert any('not REMUS 6-DOF avoidance' in fig.axes[0].get_title() for fig in captured)
    with pytest.raises(FileExistsError):
        save_episode(result, tmp_path/'run')


def test_initial_true_sphere_collision_is_reported_without_relaxed_radius(monkeypatch):
    import auv3d.stage2_p25 as module
    cfg = tiny()
    start = np.array([30., 100., 35., 0., 0., 0.])
    scene = Scene('test', HeightField.flat(),
                  LinearObstacles([[31., 100., 35.]], [[0, 0, 0]], [1.]),
                  start, np.array([210., 100., 35.]))
    monkeypatch.setattr(module, 'make_stage2_scenario', lambda *_: scene)
    result = run_p25_episode(cfg, 'static3d')
    assert result['summary']['status'] == 'obstacle_collision'
    assert result['summary']['minimum_model_clearance_m'] == pytest.approx(-cfg.body_radius_m)
    assert not result['summary']['reached_goal']
    assert result['events'][0]['kind'] == 'collision'


def test_config_mapping_preserves_current_and_compatible_integer_timing():
    cfg = tiny(current_speed=.3, current_direction_deg=90., observation_delay_s=.5)
    mapped = p25_config(cfg)
    np.testing.assert_allclose(mapped.current_ned_mps, [0., .3, 0.], atol=1e-14)
    assert mapped.sonar == cfg.sensor_config()
    assert mapped.planner.integration_dt == cfg.physics_dt
    assert mapped.planner.control_dt == cfg.control_dt
    assert mapped.planner.rollout_dt == pytest.approx(.1)
    assert mapped.prediction_dt_s == pytest.approx(cfg.horizon_s/3.)
    assert mapped.vehicle.body_radius_m == cfg.body_radius_m
    assert mapped.planner.safety_margin_m == cfg.safety_margin_m


def test_nonzero_roll_is_explicitly_unrepresentable_in_p25(monkeypatch):
    import auv3d.stage2_p25 as module
    cfg = tiny()
    scene = make_stage2_scenario('empty3d', cfg)
    scene.start_eta[3] = .1
    monkeypatch.setattr(module, 'make_stage2_scenario', lambda *_: scene)
    with pytest.raises(ValueError, match='nonzero initial roll'):
        run_p25_episode(cfg, 'empty3d')


@pytest.mark.parametrize('pitch_deg', [-29.9, 0., 29.9])
def test_long_horizon_candidates_near_pitch_envelope_are_filtered_before_euler(pitch_deg):
    cfg = p25_config(replace(tiny(), horizon_s=24.))
    state = np.array([30., 100., 35., 1., np.deg2rad(pitch_deg), 0.])
    commands = np.array([[1., np.deg2rad(q), 0.] for q in (-4., -1., 0., 1., 4.)])
    legal = pitch_legal_commands(state, commands, cfg)
    assert np.any(legal[:, 1] == 0.)
    assert np.all(np.abs(np.rad2deg(state[4]+legal[:, 1]*24.)) <= 30.+1e-9)
    assert len(legal) < len(commands)
    plan = DWA3D(cfg).plan(state, np.array([210., 100., 35.]), HeightField.flat(), [], 0.)
    assert np.isfinite(plan.states).all()
    assert np.rad2deg(np.abs(plan.states[:, 4])).max() <= 30.+1e-9


def test_default_24_second_lookahead_runs_without_euler_batch_abort():
    result = run_p25_episode(replace(tiny(), horizon_s=24.), 'empty3d')
    assert result['summary']['all_states_finite']
    assert result['summary']['status'] == 'timeout'


def test_safe_pitch_adapter_preserves_original_feasible_selection_and_clearance():
    cfg = p25_config(replace(tiny(), horizon_s=9.))
    state = np.array([30., 100., 35., 1., .1, 0.])
    goal = np.array([210., 100., 35.])
    terrain = HeightField.flat()
    track = Track(0, 4., [0.], [np.array([65., 107., 35.])])
    original = OriginalDWA3D(cfg).plan(state, goal, terrain, [track], 0.)
    adapted = DWA3D(cfg).plan(state, goal, terrain, [track], 0.)
    np.testing.assert_allclose(adapted.command, original.command)
    np.testing.assert_allclose(adapted.states, original.states)
    assert adapted.valid_count == original.valid_count
    assert adapted.minimum_predicted_clearance_m == pytest.approx(original.minimum_predicted_clearance_m)
