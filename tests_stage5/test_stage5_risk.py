"""Meaningful source, geometry, stratification and margin-separation checks."""
from dataclasses import replace
import json

import numpy as np
import pytest

from auv3d.remus_planner import _clearance_positions, _forecast_tracks
from auv3d.sensors import Track
from auv3d.stage2_config import Stage2Config
from auv3d.stage3_forecasting import ForecastContext
from auv3d.stage4_risk import ecp_scene_scores, upper_quantile
from auv3d.stage5_forecasting import forecaster_contract, make_forecaster
from auv3d.stage5_risk import (SPEED_BIN_METADATA, interval_quantiles_at_samples,
                              make_policy, speed_bin_indices, time_ball_scores,
                              time_ecp_scores, time_quantiles)
from auv3d.terrain import HeightField
from auv3d.stage2_simulation import run_remus_episode


def library(tmp_path, cfg, *, n=20, predictor='ca', metadata_change=None, mutate=None):
    times = np.array([0., .2, .4])
    pred = np.zeros((n, 2, 2, 3, 3))
    pred[..., 0] = 60.
    labels = pred.copy()
    # Different time profiles and groups must not collapse to one joint ball.
    for s in range(n):
        labels[s, :, 0, :, 0] -= np.array([.1, .3, .5])*(s+1)/n
        labels[s, :, 1, :, 0] -= np.array([.2, 1., 2.])*(s+1)/n
    mask = np.ones((n, 2, 2), bool)
    speeds = np.broadcast_to([0., 1.], mask.shape).copy()
    ids = [f'cal_{s}' for s in range(n)]
    scene_ids = [f'geometry_{s}' for s in range(n)]
    metadata = {'schema': 'stage5-risk-v1', 'split': 'calibration',
                'predictor': predictor, 'predictor_contract': forecaster_contract(predictor, cfg),
                'noise_std_m': cfg.sensor_noise_std_m, 'case_ids': ids,
                'speed_bin_edges_mps': SPEED_BIN_METADATA,
                'scenes': [{'split': 'calibration', 'case_id': cid, 'scene_id': sid,
                            'noise_std_m': cfg.sensor_noise_std_m}
                           for cid, sid in zip(ids, scene_ids)]}
    metadata.update(metadata_change or {})
    values = dict(pred_local=pred, label_local=labels, mask=mask,
                  radii=np.ones(mask.shape), observed_speeds_mps=speeds,
                  times_s=times, case_ids=np.array(ids), scene_ids=np.array(scene_ids),
                  empty_scene_mask=np.zeros(n, bool))
    if mutate:
        mutate(values)
    path = tmp_path/f'library_{len(list(tmp_path.iterdir()))}.npz'
    np.savez(path, **values, metadata_json=json.dumps(metadata))
    return path, values


def setup(cfg, track=None):
    points = np.array([[20., 50., 30.], [20.1, 50., 30.], [20.2, 50., 30.]])
    states = np.zeros((3, 16))
    states[:, 0], states[:, 1:4] = [0., .2, .4], points
    history = track or Track(1, 2., [0., 1.], [np.array([45., 50., 30.])]*2)
    context = ForecastContext(cfg, make_forecaster('ca', cfg))
    context.begin({1: 2.5})
    terrain = HeightField.flat(260., 200., 90.)
    base = _clearance_positions(points, terrain, _forecast_tracks([history], states[:, 0], 1.4, cfg, context), cfg)
    return states, terrain, history, context, base


def test_speed_bins_use_causal_observed_speed_with_exact_threshold():
    np.testing.assert_array_equal(speed_bin_indices([0., .249999, .25, 4.]), [0, 0, 1, 1])
    with pytest.raises(ValueError):
        speed_bin_indices([float('nan')])


def test_time_ball_keeps_time_speed_groups_and_missing_scene_group_is_not_zero(tmp_path):
    cfg = replace(Stage2Config(), horizon_s=.4)
    _, v = library(tmp_path, cfg)
    v['mask'][0, :, 0] = False
    v['pred_local'][0, :, 0] = np.nan
    v['label_local'][0, :, 0] = np.nan
    scores, present = time_ball_scores(v['pred_local'], v['label_local'], v['mask'], v['observed_speeds_mps'])
    assert scores.shape == (20, 2, 3)
    assert np.isnan(scores[0, 0]).all() and not present[0, 0]
    np.testing.assert_allclose(scores[-1], [[.1, .3, .5], [.2, 1., 2.]])
    q = time_quantiles(scores, present, .1)
    assert q[0, 0] < q[0, 2] < q[1, 2]


def test_time_ecp_uses_nearest_objects_separately_and_synchronous_swept_geometry():
    pred = np.zeros((1, 1, 3, 3, 3))
    labels = pred.copy()
    pred[0, 0, 0, :, 0] = 10.
    pred[0, 0, 1, :, 0] = 30.
    labels[0, 0, 0, :, 0] = 20.
    labels[0, 0, 1, :, 0] = 5.
    pred[0, 0, 2, :, 0] = labels[0, 0, 2, :, 0] = 50.
    scores, present = time_ecp_scores(np.zeros((3, 3)), pred, labels,
        np.ones((1, 1, 3)), np.ones((1, 1, 3), bool), np.array([[[0., 0., 1.]]]))
    # min pred=9, min label=4; not max individual difference (=25).
    np.testing.assert_allclose(scores, [[[5., 5.], [0., 0.]]])
    assert present.all()
    # A sphere passes through the ego between endpoints; endpoint-only misses it.
    pred[0, 0, 0, :, 0] = [10., 10., 10.]
    labels[0, 0, 0, :, 0] = [-10., 10., 10.]
    scores, _ = time_ecp_scores(np.zeros((3, 3)), pred, labels,
        np.ones((1, 1, 3)), np.ones((1, 1, 3), bool), np.array([[[0., 0., 1.]]]))
    assert scores[0, 0, 0] == 10.


def test_interval_radii_never_interpolate_down_across_a_peak():
    q = interval_quantiles_at_samples([1., 9.], [0., .2, .4], [0., .1, .2, .3, .4])
    np.testing.assert_array_equal(q, [1., 1., 9., 9., 9.])
    with pytest.raises(ValueError):
        interval_quantiles_at_samples([1., 9.], [0., .2, .4], [.5])


@pytest.mark.parametrize('method', ['A0', 'A1', 'A2', 'A3', 'A4', 'A5'])
def test_all_methods_keep_no_obstacle_terrain_clearance_exact(tmp_path, method):
    cfg = replace(Stage2Config(), horizon_s=.4)
    path, _ = library(tmp_path, cfg)
    policy = make_policy(method, path, cfg=cfg, fixed_margin_m=1.)
    states, terrain, _, context, _ = setup(cfg)
    clear, record = policy.adjust(states, terrain, [], 1.4, cfg, context, 12.3)
    assert clear == 12.3 and record['quantile_m'] == 0.


def test_replace_removes_only_fixed_and_age_retaining_physical_and_memory(tmp_path):
    cfg = replace(Stage2Config(), horizon_s=.4)
    states, terrain, history, context, base = setup(cfg)
    add, ra = make_policy('A1', cfg=cfg, fixed_margin_m=1.3, combination='add').adjust(
        states, terrain, [history], 1.4, cfg, context, base)
    repl, rr = make_policy('A1', cfg=cfg, fixed_margin_m=1.3, combination='replace').adjust(
        states, terrain, [history], 1.4, cfg, context, base)
    assert repl-add == pytest.approx(cfg.prediction_margin_m+.4)
    assert add == pytest.approx(base-1.3)
    assert rr['tracks'][0]['removed_components'] == {
        'fixed_prediction_margin_m': cfg.prediction_margin_m, 'stage2_age_margin_m': pytest.approx(.4)}
    assert rr['memory_extra_uncertainty_retained'] is True
    # Recompute the expected envelope with the original 2.5m memory bound.
    expected = _clearance_positions(states[:, 1:4], terrain,
        [(np.repeat(history.positions[-1][None], 3, axis=0), history.radius_m+cfg.body_radius_m+2.5+1.3)], cfg)
    assert repl == pytest.approx(expected)


@pytest.mark.parametrize('field,value', [
    ('split', 'test'), ('predictor', 'cv'), ('noise_std_m', .2),
    ('predictor_contract', {'predictor': 'ca', 'history_dt_s': 9}),
    ('case_ids', ['wrong']), ('speed_bin_edges_mps', [0., 1., 'inf'])])
def test_library_matches_actual_predictor_noise_roles_and_parameters(tmp_path, field, value):
    cfg = replace(Stage2Config(), horizon_s=.4)
    path, _ = library(tmp_path, cfg, metadata_change={field: value})
    with pytest.raises(ValueError, match='mismatch'):
        make_policy('A2', path, cfg=cfg)


def test_missing_scene_cannot_be_called_empty_to_supply_a_zero_score(tmp_path):
    cfg = replace(Stage2Config(), horizon_s=.4)
    def mutate(v):
        v['mask'][0] = False
    path, _ = library(tmp_path, cfg, mutate=mutate)
    with pytest.raises(ValueError, match='empty'):
        make_policy('A2', path, cfg=cfg)


@pytest.mark.parametrize('field,value', [('split', 'test'), ('case_id', 'another'),
                                       ('scene_id', 'another'), ('noise_std_m', .2)])
def test_source_roles_cannot_be_hidden_by_top_level_calibration_declaration(tmp_path, field, value):
    cfg = replace(Stage2Config(), horizon_s=.4)
    path, v = library(tmp_path, cfg)
    # An apparently valid top-level declaration must not disguise a test source.
    with np.load(path, allow_pickle=False) as data:
        payload = {name: data[name].copy() for name in data.files}
    metadata = json.loads(str(payload['metadata_json'].item()))
    metadata['scenes'][0][field] = value
    payload['metadata_json'] = np.array(json.dumps(metadata))
    edited = tmp_path/'wrong_source.npz'
    np.savez(edited, **payload)
    with pytest.raises(ValueError, match='source role'):
        make_policy('A2', edited, cfg=cfg)


def test_stratified_insufficient_scenes_reject_instead_of_cap_or_pool_missing_bins(tmp_path):
    cfg = replace(Stage2Config(), horizon_s=.4)
    def mutate(v):
        v['mask'][:15, :, 0] = False  # Five genuinely stationary scenes remain.
    path, _ = library(tmp_path, cfg, mutate=mutate)
    for method in ('A2', 'A5'):
        with pytest.raises(ValueError, match='bin 0.*n=5'):
            make_policy(method, path, cfg=cfg, alpha=.1)
    # Joint form does not silently impose unused speed-stratum constraints.
    assert np.isfinite(make_policy('A3', path, cfg=cfg, alpha=.1).ball_q)


@pytest.mark.parametrize('method', ['A2', 'A3', 'A4', 'A5'])
def test_candidate_risk_is_time_aligned_and_does_not_mutate_forecast_or_states(tmp_path, method):
    cfg = replace(Stage2Config(), horizon_s=.4)
    path, _ = library(tmp_path, cfg)
    policy = make_policy(method, path, cfg=cfg)
    states, terrain, history, context, base = setup(cfg)
    original = states.copy()
    forecast = context.forecast(history, 1.4+states[:, 0], 1.4)
    old_radius = forecast.enclosing_radii_m.copy()
    clear, record = policy.adjust(states, terrain, [history], 1.4, cfg, context, base)
    assert clear <= base+1e-10 and np.isfinite(clear)
    np.testing.assert_array_equal(states, original)
    np.testing.assert_array_equal(forecast.enclosing_radii_m, old_radius)
    assert record['tracks'][0]['speed_bin'] == 0
    assert record['applied_q_offsets_s'] == [0., .2, .4]
    assert len(record['tracks'][0]['applied_q_m']) == 3
    json.dumps(record, allow_nan=False)


def test_joint_ecp_preserves_stage4_score_form_not_per_bin_nearest_min(tmp_path):
    cfg = replace(Stage2Config(), horizon_s=.4)
    path, v = library(tmp_path, cfg)
    policy = make_policy('A4', path, cfg=cfg)
    states, terrain, history, context, base = setup(cfg)
    _, record = policy.adjust(states, terrain, [history], 1.4, cfg, context, base)
    local = states[:, 1:4]-states[0, 1:4]
    expected = ecp_scene_scores(local, v['pred_local'], v['label_local'],
                                v['radii'], v['mask'], cfg.body_radius_m)
    assert record['quantile_m'] == pytest.approx(upper_quantile(expected, .1))


def test_online_predictor_mismatch_and_unfrozen_fixed_margin_are_rejected(tmp_path):
    cfg = replace(Stage2Config(), horizon_s=.4)
    with pytest.raises(ValueError, match='explicit'):
        make_policy('A1', cfg=cfg)
    states, terrain, history, _, base = setup(cfg)
    context = ForecastContext(cfg, make_forecaster('cv', cfg))
    with pytest.raises(ValueError, match='online predictor'):
        make_policy('A1', cfg=cfg, fixed_margin_m=1.).adjust(
            states, terrain, [history], 1.4, cfg, context, base)


@pytest.mark.parametrize('method', ['A0', 'A1', 'A2', 'A3', 'A4', 'A5'])
def test_all_methods_complete_real_remus_episode_and_summary(tmp_path, method):
    cfg = replace(Stage2Config(), horizon_s=.4, control_dt=.2, max_time_s=.2,
                  warmup_s=0., candidate_budget=1)
    path, _ = library(tmp_path, cfg)
    policy = make_policy(method, path, cfg=cfg, fixed_margin_m=.2)
    result = run_remus_episode(cfg, 'static3d', risk_policy=policy,
                              forecaster=make_forecaster('ca', cfg))
    assert result['summary']['risk_method'] == method
    assert policy.description in result['summary']['prediction']
    assert len(result['states']) >= 2 and np.isfinite(result['states']).all()
    assert result['planning_diagnostics']
