"""Causal inputs, scene separation and replay checks for frozen calibration."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from auv3d import stage4_calibration as calibration
from auv3d.obstacles import LinearObstacles
from auv3d.stage2_config import Stage2Config


def _json(path, content):
    path.write_text(json.dumps(content), encoding='utf-8')


def _position(time):
    return np.array([15., 25., 35.]) + time * np.array([1., 2., -.1])


@pytest.fixture
def episodes(tmp_path, monkeypatch):
    def factory(scenario, cfg):
        assert scenario == 'crossing3d'
        return SimpleNamespace(obstacles=LinearObstacles([_position(0)], [[1., 2., -.1]], [2.]))
    monkeypatch.setattr(calibration, 'make_stage2_scenario', factory)
    rows = []
    for index in range(24):
        case_id = f'cal{index:02d}'
        row = {'case_id': case_id, 'scenario_id': f'scene{index:02d}', 'scenario': 'crossing3d',
               'split': 'calibration', 'method': 'B0', 'seed': index}
        rows.append(row)
        directory = tmp_path / 'cases' / case_id
        directory.mkdir(parents=True)
        _json(directory / 'config.json', Stage2Config(seed=index).to_dict())
        _json(directory / 'summary.json', {'backend': 'remus6dof', 'scenario': 'crossing3d',
                                           'seed': index, 'status': 'success'})
        np.savetxt(directory / 'states.csv', [[0., 10., 20., 30., np.pi / 2],
                                               [20., 30., 60., 28., np.pi / 2]], delimiter=',',
                   header='time_s,north_m,east_m,depth_m,yaw_rad', comments='')
        _json(directory / 'planning_diagnostics.json', [
            {'time_s': 0., 'tracks': [{'object_id': 0, 'radius': 2., 'times_s': [0.],
                                      'positions': [_position(0).tolist()]}]},
            {'time_s': 20., 'tracks': [{'object_id': 0, 'radius': 2., 'times_s': [19., 20.],
                                       'positions': [_position(19).tolist(), _position(20).tolist()]}]}])
        _json(directory / 'observations.json', [
            {'measurement_time_s': time, 'delivered_time_s': time,
             'objects': [{'object_id': 0, 'position': _position(time).tolist(), 'radius': 2.}]}
            for time in [0., 19., 20.]])
        _json(directory / 'obstacles.json', [
            {'time_s': time, 'ids': [0], 'positions': [_position(time).tolist()], 'radii': [2.]}
            for time in [0., 20.]])
    return tmp_path, rows


def test_library_scene_shape_causal_ca_and_local_coordinates(episodes):
    root, rows = episodes
    metadata = calibration.build_library(root, rows, root / 'library.npz')
    with np.load(root / 'library.npz', allow_pickle=False) as library:
        assert library['pred_local'].shape == (24, 10, 1, 121, 3)
        assert library['label_local'].shape == library['pred_local'].shape
        assert library['mask'].shape == (24, 10, 1)
        assert library['mask'][:, :2].all()
        assert not library['mask'][:, 2:].any()
        assert np.isnan(library['pred_local'][:, 2:]).all()
        assert np.all(library['radii'][:, :2] == 2.)
        assert np.allclose(library['times_s'], np.arange(121) * .2)
        assert np.allclose(library['pred_local'][0, 0, 0, 0], [5., -5., 5.])
        # The t=0 predictor saw ONE observation: it cannot know the true velocity.
        assert np.allclose(library['pred_local'][0, 0, 0], [5., -5., 5.])
        assert np.allclose(library['pred_local'][:, 1], library['label_local'][:, 1], atol=1e-9)
        assert np.allclose(library['b1_scene_scores_m'], 24. * np.linalg.norm([1., 2., -.1]))
        assert not library['empty_scene_mask'].any()
        assert json.loads(str(library['metadata_json']))['statistical_unit'] == 'complete_scene'
    assert metadata['online_future_truth_available'] is False
    assert metadata['calibration_scene_count'] == 24
    assert metadata['scenes'][0]['replayed_snapshots_checked'] == 2
    assert metadata['scenes'][0]['maximum_replay_position_error_m'] < 1e-9


def test_test_and_tuning_roles_are_never_read(episodes):
    root, rows = episodes
    rows += [{'case_id': 'nonexistent_test_future', 'split': 'test', 'method': 'B0'},
             {'case_id': 'nonexistent_tuning_future', 'split': 'tuning', 'method': 'B0'},
             {'case_id': 'nonexistent_B2', 'split': 'calibration', 'method': 'B2'}]
    calibration.build_library(root, rows, root / 'library.npz')
    with np.load(root / 'library.npz', allow_pickle=False) as library:
        assert len(library['case_ids']) == 24
        assert not any('nonexistent' in case_id for case_id in library['case_ids'])


def test_complete_empty_obstacle_scene_is_explicit(episodes, monkeypatch):
    root, rows = episodes
    monkeypatch.setattr(calibration, 'make_stage2_scenario',
                        lambda *args: SimpleNamespace(obstacles=LinearObstacles([], [], [])))
    for row in rows:
        directory = root / 'cases' / row['case_id']
        _json(directory / 'planning_diagnostics.json', [{'time_s': 0., 'tracks': []}])
        _json(directory / 'obstacles.json', [{'time_s': 0., 'ids': [], 'positions': [], 'radii': []}])
    metadata = calibration.build_library(root, rows, root / 'library.npz')
    with np.load(root / 'library.npz', allow_pickle=False) as library:
        assert library['empty_scene_mask'].all()
        assert np.all(library['b1_scene_scores_m'] == 0.)
        assert not library['mask'].any()
        assert np.isnan(library['pred_local']).all()
    assert all(scene['empty_obstacle_scene'] for scene in metadata['scenes'])


@pytest.mark.parametrize('mutation,match', [
    ('future_history', 'causal'), ('pending_observation', 'causally delivered'),
    ('unobserved_position', 'causally delivered'), ('truth_mismatch', 'motion disagrees'),
    ('radius_mismatch', 'radius disagrees'), ('no_targets', 'cannot impute zero'),
    ('duplicate_anchor', 'duplicate planning'), ('missing_state', 'exactly one actual'),
])
def test_invalid_or_missing_labels_do_not_become_zero(episodes, mutation, match):
    root, rows = episodes
    directory = root / 'cases' / rows[0]['case_id']
    plans = json.loads((directory / 'planning_diagnostics.json').read_text())
    observations = json.loads((directory / 'observations.json').read_text())
    snapshots = json.loads((directory / 'obstacles.json').read_text())
    if mutation == 'future_history':
        plans[0]['tracks'][0]['times_s'] = [1.]
    elif mutation == 'pending_observation':
        observations[0]['delivered_time_s'] = 1.
    elif mutation == 'unobserved_position':
        plans[0]['tracks'][0]['positions'][0][0] += 1.
    elif mutation == 'truth_mismatch':
        snapshots[0]['positions'][0][0] += 1.
    elif mutation == 'radius_mismatch':
        plans[0]['tracks'][0]['radius'] = 3.
    elif mutation == 'no_targets':
        plans = [{'time_s': 0., 'tracks': []}]
    elif mutation == 'duplicate_anchor':
        plans.append(plans[0])
    elif mutation == 'missing_state':
        plans[0]['time_s'] = 40.
    _json(directory / 'planning_diagnostics.json', plans)
    _json(directory / 'observations.json', observations)
    _json(directory / 'obstacles.json', snapshots)
    with pytest.raises(ValueError, match=match):
        calibration.build_library(root, rows, root / 'invalid.npz')
    assert not (root / 'invalid.npz').exists()


@pytest.mark.parametrize('mutation,match', [
    ('count', 'exactly 24'), ('duplicate_case', 'case_ids must be unique'),
    ('duplicate_scene', 'scenario identities'), ('error', 'program-error'),
    ('escape', 'within run_root'), ('seed', 'configuration disagree'),
    ('physics', '.02 s physics'), ('predictor', 'CA baseline'),
])
def test_manifest_and_actual_configuration_are_checked(episodes, mutation, match):
    root, rows = episodes
    if mutation == 'count':
        rows = rows[:23]
    elif mutation == 'duplicate_case':
        rows[1]['case_id'] = rows[0]['case_id']
    elif mutation == 'duplicate_scene':
        rows[1]['scenario_id'] = rows[0]['scenario_id']
    elif mutation == 'error':
        rows[0]['program_error'] = True
    elif mutation == 'escape':
        rows[0]['result_dir'] = '../outside'
    elif mutation == 'seed':
        rows[0]['seed'] = 999
    elif mutation in ('physics', 'predictor'):
        config_path = root / 'cases' / rows[0]['case_id'] / 'config.json'
        config = json.loads(config_path.read_text())
        if mutation == 'physics':
            config['physics_dt'] = .01
        else:
            config['obstacle_predictor'] = 'pplnm'
        _json(config_path, config)
    with pytest.raises(ValueError, match=match):
        calibration.build_library(root, rows, root / 'invalid.npz')
    assert not (root / 'invalid.npz').exists()


def test_existing_library_is_preserved_before_reading(episodes):
    root, rows = episodes
    out = root / 'preserve.npz'
    out.write_bytes(b'existing experiment')
    with pytest.raises(FileExistsError):
        calibration.build_library(root, rows, out)
    assert out.read_bytes() == b'existing experiment'


@pytest.mark.parametrize('mutation,match', [
    ('summary_B2', 'summary must identify B0'),
    ('error_artifact', 'program-error artifacts'),
    ('metadata_B1', 'experiment metadata must identify B0'),
    ('metadata_test_role', 'metadata must declare calibration split'),
    ('metadata_wrong_case', 'case_id disagrees'),
    ('registered_parameters', 'registered common B0 parameters'),
])
def test_actual_source_must_be_completed_registered_B0(episodes, mutation, match):
    root, rows = episodes
    directory = root / 'cases' / rows[0]['case_id']
    if mutation == 'summary_B2':
        summary = json.loads((directory / 'summary.json').read_text())
        summary['risk_method'] = 'B2'
        _json(directory / 'summary.json', summary)
    elif mutation == 'error_artifact':
        _json(directory / 'program_error.json', {'error': 'plot failed after summary was saved'})
    elif mutation == 'metadata_B1':
        _json(directory / 'risk_experiment.json', {'method': 'B1'})
    elif mutation == 'metadata_test_role':
        _json(directory / 'risk_experiment.json', {'method': 'B0', 'split': 'test'})
    elif mutation == 'metadata_wrong_case':
        _json(directory / 'risk_experiment.json', {'method': 'B0', 'job': {'case': {'case_id': 'different_case_id'}}})
    else:
        rows[0].update(family='crossing3d', config_overrides={})
    with pytest.raises(ValueError, match=match):
        calibration.build_library(root, rows, root / 'invalid.npz')
    assert not (root / 'invalid.npz').exists()


def test_production_parameter_normalization_accepts_json_reference_lists(episodes):
    from auv3d.stage4_protocol import config_for_case
    root, rows = episodes
    row = dict(rows[0], family='crossing3d', config_overrides={})
    directory = root / 'cases' / row['case_id']
    _json(directory / 'config.json', config_for_case(row).to_dict())
    _json(directory / 'risk_experiment.json', {'method': 'B0'})
    windows, metadata = calibration._load_scene(root, row)
    assert len(windows) == 2
    assert metadata['configuration'] == config_for_case(row).to_dict()


def test_local_yaw_frame_preserves_3d_distance_and_ned_depth():
    origin = np.array([70., -20., 30.])
    points = np.array([[80., -14., 35.], [77., -22., 28.]])
    local = calibration.yaw_local(points, origin, .73)
    assert np.allclose(np.linalg.norm(points - origin, axis=-1), np.linalg.norm(local, axis=-1))
    assert np.allclose(np.linalg.norm(points[0] - points[1]), np.linalg.norm(local[0] - local[1]))
    assert np.allclose(local[:, 2], [5., -2.])
