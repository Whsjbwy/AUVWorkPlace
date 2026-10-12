from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from auv3d.stage4_protocol import manifest as old_manifest
from auv3d.stage5_protocol import (
    FAMILIES, NOISE_LEVELS, PHASES, common_config, config_for_case,
    make_scene, manifest, protocol_record, serialize,
)


@pytest.mark.parametrize('phase,count,cal', [('pilot', 48, 24), ('formal', 96, 72)])
def test_whole_scene_splits_noise_replicas_and_new_seed_identity(phase, count, cal):
    rows = manifest(phase)
    assert len(rows) == count * 3
    assert len({r['scene_id'] for r in rows}) == count
    assert len({r['case_id'] for r in rows}) == len(rows)
    old_seeds = {r['seed'] for r in old_manifest()}
    assert not old_seeds & {r['seed'] for r in rows}
    for scene_id in {r['scene_id'] for r in rows}:
        group = [r for r in rows if r['scene_id'] == scene_id]
        assert {r['noise_std_m'] for r in group} == set(NOISE_LEVELS)
        assert len({r['split'] for r in group}) == 1
        assert len({r['seed'] for r in group}) == 1
    record = protocol_record(phase)
    assert record['unique_geometry_count'] == count
    assert record['split_counts'] == dict(dev=6, tune=6, calibration=cal, test=12)
    assert not record['future_truth_used_online']
    assert not record['physical_and_safety_parameters_modified']


def test_new_phase_geometry_does_not_reuse_pilot_geometry_or_noise_as_samples():
    pilot, formal = manifest('pilot', noise_std=.02), manifest('formal', noise_std=.02)
    assert not {r['scene_id'] for r in pilot} & {r['scene_id'] for r in formal}
    assert not {r['seed'] for r in pilot} & {r['seed'] for r in formal}
    for family in FAMILIES[:4]:
        p = {tuple(r['layout_shift_m']) for r in pilot if r['family'] == family}
        f = {tuple(r['layout_shift_m']) for r in formal if r['family'] == family}
        assert len(p) == 8 and len(f) == 16 and not p & f


@pytest.mark.parametrize('phase', PHASES)
def test_all_registered_cases_preserve_physics_candidates_and_acceptance(phase):
    base = common_config().to_dict()
    protected = ('physics_dt', 'control_dt', 'horizon_s', 'candidate_budget',
        'candidate_scheme', 'shortlist_policy', 'safety_margin_m', 'goal_tolerance_m',
        'prediction_margin_m', 'body_radius_m', 'rpm_values', 'nominal_rpm', 'depth_limit_m',
        'warmup_s', 'max_time_s', 'emergency_policy')
    for row in manifest(phase):
        cfg = config_for_case(row)
        assert all(cfg.to_dict()[key] == base[key] for key in protected)
    assert base['safety_margin_m'] == 1 and base['goal_tolerance_m'] == 5


@pytest.mark.parametrize('family', FAMILIES[:4])
def test_layout_translation_changes_real_geometry_not_motion_law(family):
    row = next(r for r in manifest(noise_std=.02) if r['family'] == family)
    cfg = config_for_case(row)
    from auv3d.scenarios import make_stage2_scenario
    original, changed = make_stage2_scenario(family, cfg), make_scene(row, cfg)
    shift = row['layout_shift_m']
    assert np.linalg.norm(shift) > 0
    for time_step in (0., .02, 10., 30.):
        if time_step:
            original.obstacles.step(time_step)
            changed.obstacles.step(time_step)
        np.testing.assert_allclose(changed.obstacles.snapshot().positions,
            original.obstacles.snapshot().positions + shift, atol=1e-10, rtol=0)
        np.testing.assert_array_equal(changed.obstacles.snapshot().radii, original.obstacles.snapshot().radii)
    np.testing.assert_array_equal(changed.start_eta, original.start_eta)
    np.testing.assert_array_equal(changed.goal, original.goal)


def test_noise_replicas_share_geometry_and_mutation_does_not_change_registration():
    group = manifest()[:3]
    scenes = [make_scene(row) for row in group]
    for scene in scenes[1:]:
        np.testing.assert_array_equal(scene.obstacles.snapshot().positions, scenes[0].obstacles.snapshot().positions)
    bad = deepcopy(group[0])
    bad['config_overrides']['safety_margin_m'] = .1
    with pytest.raises(ValueError, match='method-independent'):
        config_for_case(bad)
    assert manifest()[0] == group[0]
    with pytest.raises(ValueError, match='common physical'):
        make_scene(group[0], replace(config_for_case(group[0]), prediction_margin_m=0.))


def test_test_identity_cannot_be_relabeled_to_calibration_or_new_noise_seed():
    row = manifest(split='test', noise_std=.02)[0]
    for change in ({'split': 'calibration'}, {'seed': row['seed'] + 50}, {'phase': 'formal'}):
        fake = deepcopy(row)
        fake.update(change)
        with pytest.raises(ValueError, match='registration'):
            config_for_case(fake)


@pytest.mark.parametrize('change', [
    {'seed': True}, {'seed': 7}, {'split': 'random_window'},
    {'layout_shift_m': [4, 0, 0]}, {'layout_shift_m': [0, 0, 1]},
    {'case_id': '../bad'}, {'noise_std_m': .03}, {'family': 'empty3d'},
])
def test_malformed_case_registration_rejected(change):
    row = manifest()[0]
    row.update(change)
    with pytest.raises(ValueError):
        config_for_case(row)


def test_pre_registration_json_exclusive_and_no_nonfinite_edges(tmp_path):
    path = tmp_path / 'protocol.json'
    text = serialize(path, 'formal')
    assert json.loads(text)['alpha'] == .1
    assert json.loads(text)['speed_bin_edges_mps'] == [0., .25, None]
    assert text == serialize(phase='formal')
    with pytest.raises(FileExistsError):
        serialize(path, 'formal')
