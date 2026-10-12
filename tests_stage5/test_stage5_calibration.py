from copy import deepcopy
import json

import numpy as np
import pytest

from auv3d.stage5_calibration import (
    ANCHOR_TIMES_S, QUERY_TIMES_S, _selected_rows, _validate_sample,
    build_library, generate_evaluation_samples, generate_scene_samples, risk_calibration_rows,
    speed_bin, stratified_ball_quantiles,
)
from auv3d.stage5_protocol import manifest


def _case():
    return next(r for r in manifest(split='calibration', noise_std=.02) if r['family'] == 'static3d')


@pytest.fixture(scope='module')
def acquired():
    row = _case()
    return row, generate_scene_samples(row)


def test_fixed_carrier_uses_delivered_histories_and_separate_aligned_truth(acquired):
    row, sample = acquired
    _validate_sample(sample, row)
    meta = sample['metadata']
    assert meta['source_is_REMUS_execution'] is False
    assert meta['future_truth_used_for_forecast'] is False
    assert meta['selected_anchor_s'] == list(ANCHOR_TIMES_S)
    for window in sample['windows']:
        assert window['label_world'].shape == (1, len(QUERY_TIMES_S), 3)
        for track in window['tracks']:
            assert max(track.times) <= window['anchor_s']
    # Static truth stays fixed while the noisy measured history actually varies.
    truth = sample['windows'][0]['label_world'][0]
    np.testing.assert_array_equal(truth, np.repeat(truth[:1], len(truth), axis=0))
    assert np.any(sample['windows'][0]['tracks'][0].positions[-1] != truth[0])


@pytest.mark.parametrize('key,value', [
    ('split', 'test'), ('case_id', 'other_case'), ('scene_id', 'other_scene'),
    ('noise_std_m', .1), ('seed', 7), ('phase', 'formal'),
    ('source_is_REMUS_execution', True), ('future_truth_used_for_forecast', True),
    ('completed_acquisition', False),
])
def test_wrong_source_provenance_rejected(acquired, key, value):
    row, original = acquired
    sample = deepcopy(original)
    sample['metadata'][key] = value
    with pytest.raises(ValueError):
        _validate_sample(sample, row)


def test_undelivered_measurement_and_missing_nonempty_labels_rejected(acquired):
    row, original = acquired
    sample = deepcopy(original)
    sample['metadata']['observations'] = []
    with pytest.raises(ValueError, match='undelivered'):
        _validate_sample(sample, row)
    sample = deepcopy(original)
    sample['windows'] = []
    with pytest.raises(ValueError, match='missing labels'):
        _validate_sample(sample, row)
    sample['metadata']['empty_obstacle_scene'] = True
    with pytest.raises(ValueError, match='missing labels'):
        _validate_sample(sample, row)
    sample = deepcopy(original)
    sample['windows'][0]['label_world'][0, -1, 0] = np.nan
    with pytest.raises(ValueError, match='finite aligned'):
        _validate_sample(sample, row)


def test_splits_noise_replicas_and_duplicate_scenes_never_enlarge_n():
    row = _case()
    with pytest.raises(ValueError, match='duplicate complete scenes'):
        _selected_rows([row, row])
    group = next(r for r in manifest(split='calibration', noise_std=.1) if r['scene_id'] == row['scene_id'])
    with pytest.raises(ValueError, match='duplicate complete scenes'):
        _selected_rows([row, group])
    with pytest.raises(ValueError, match='calibration-role'):
        _selected_rows(manifest(split='test', noise_std=.02))


def test_empty_terrain_exclusion_is_all_families_not_bad_seed_selection():
    rows = manifest('formal', split='calibration', noise_std=.02)
    selected = risk_calibration_rows(rows)
    assert len(rows) == 72 and len(selected) == 48
    assert {r['family'] for r in selected} == {'static3d', 'crossing3d', 'maneuvering3d', 'multi_crossing3d'}
    assert all(r['split'] == 'calibration' for r in selected)
    assert any(r['seed'] == 525006 for r in rows)
    assert not any(r['seed'] == 525006 for r in selected)
    assert len([r for r in rows if r['family'] == 'perlin_cruise_constrained3d']) == 12


def test_evaluation_truth_preserves_unseen_role_and_cannot_enter_calibration(tmp_path):
    row = next(r for r in manifest(split='test', noise_std=.02) if r['family'] == 'static3d')
    with pytest.raises(ValueError, match='calibration split'):
        generate_scene_samples(row)
    sample = generate_evaluation_samples(row)
    assert sample['metadata']['split'] == 'test'
    assert sample['metadata']['source_purpose'] == 'evaluation'
    _validate_sample(sample, row, purpose='evaluation')
    with pytest.raises(ValueError):
        build_library([row], tmp_path / 'leaked.npz', samples=[sample])
    # Relabelling the row cannot relabel actual source provenance.
    fake = deepcopy(row)
    fake['split'] = 'calibration'
    with pytest.raises(ValueError, match='split'):
        build_library([fake], tmp_path / 'leaked.npz', samples=[sample])


def test_independent_predictor_library_shapes_and_provenance(acquired, tmp_path):
    row, sample = acquired
    out = tmp_path / 'ca.npz'
    metadata = build_library([row], out, 'ca', samples=[sample])
    with np.load(out, allow_pickle=False) as data:
        assert data['pred_local'].shape == data['label_local'].shape == (1, 3, 1, 121, 3)
        assert data['observed_speeds_mps'].shape == data['mask'].shape
        assert data['mask'].all()
        assert data['active_scene_mask'].tolist() == [True]
        assert data['empty_scene_mask'].tolist() == [False]
        assert json.loads(str(data['metadata_json'])) == json.loads(json.dumps(metadata))
    assert metadata['predictor'] == 'ca' and metadata['noise_std_m'] == .02
    assert metadata['registered_scene_count'] == 1
    with pytest.raises(FileExistsError):
        build_library([row], out, 'ca', samples=[sample])


def test_empty_terrain_is_recorded_not_an_obstacle_zero_score(tmp_path):
    row = next(r for r in manifest(split='calibration', noise_std=0.) if r['family'] == 'terrain3d')
    sample = generate_scene_samples(row)
    assert sample['metadata']['empty_obstacle_scene'] is True
    assert sample['windows'] == []
    out = tmp_path / 'empty.npz'
    metadata = build_library([row], out, samples=[sample])
    assert metadata['active_scene_count'] == 0
    assert metadata['empty_scene_count'] == 1
    assert metadata['observed_speed_bin_scene_counts'] == [0, 0]


def test_per_time_group_quantile_uses_complete_scene_max_and_honest_rank():
    pred = np.zeros((4, 2, 2, 3, 3))
    label = np.zeros_like(pred)
    mask = np.ones((4, 2, 2), bool)
    speeds = np.zeros(mask.shape)
    speeds[:, :, 1] = .8
    for si in range(4):
        label[si, 1, 0, :, 0] = (si + 1) * np.array([1., 2., 3.])
        label[si, 0, 1, :, 1] = (si + 1) * np.array([2., 3., 4.])
    result = stratified_ball_quantiles(pred, label, mask, speeds, .2)
    assert result['scene_counts'].tolist() == [4, 4]
    np.testing.assert_array_equal(result['quantiles_m'][0], [4., 8., 12.])
    np.testing.assert_array_equal(result['quantiles_m'][1], [8., 12., 16.])
    # Three scenes cannot provide finite alpha=.2 corrected order statistics.
    small = stratified_ball_quantiles(pred[:3], label[:3], mask[:3], speeds[:3], .2)
    assert np.isinf(small['quantiles_m']).all()
    # An absent bin is unknown, not zero; inactive padding has no label meaning.
    mask[:, :, 1] = False
    result = stratified_ball_quantiles(pred, label, mask, speeds, .2)
    assert result['scene_counts'].tolist() == [4, 0]
    assert np.isnan(result['scene_scores_m'][:, 1]).all()
    assert np.isinf(result['quantiles_m'][1]).all()


@pytest.mark.parametrize('value', [-1., np.nan, np.inf])
def test_invalid_observed_speed_rejected(value):
    with pytest.raises(ValueError):
        speed_bin(value)


def test_speed_groups_use_observed_boundary_without_future_truth():
    assert speed_bin(0.) == speed_bin(.24999) == 0
    assert speed_bin(.25) == speed_bin(10.) == 1
