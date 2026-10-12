import json
from dataclasses import replace

import numpy as np
import pytest

from auv3d.stage4_risk import (RiskPolicy, ecp_scene_scores, local_positions,
                              sphere_scene_scores, upper_quantile)
from auv3d.stage2_config import Stage2Config
from auv3d.stage3_forecasting import ForecastContext
from auv3d.remus_planner import RemusDWA, trajectory_clearance
from auv3d.remus import RemusPlant
from auv3d.sensors import Track
from auv3d.terrain import HeightField


def library(path, count=24, split='calibration'):
    times = np.arange(0., 24.01, 1.)
    pred = np.zeros((count, 1, 1, len(times), 3))
    pred[..., 0] = 80.
    label = pred.copy()
    label[..., 0] -= np.arange(count)[:, None, None, None] / 10
    np.savez_compressed(path, pred_local=pred, label_local=label,
                        mask=np.ones(pred.shape[:3], bool), radii=np.ones(pred.shape[:3]),
                        times_s=times, case_ids=np.array([f'cal_{i}' for i in range(count)]),
                        empty_scene_mask=np.zeros(count, bool),
                        metadata_json=json.dumps({'split': split}))
    return path


@pytest.mark.parametrize('alpha,expected', [(.1, 22.), (.2, 19.), (.3, 17.), (.05, 23.)])
def test_finite_sample_scene_order_statistic(alpha, expected):
    assert upper_quantile(np.arange(24.), alpha) == expected


def test_small_sample_is_infinite_without_empirical_cap():
    assert np.isinf(upper_quantile(np.arange(12.), .05))


@pytest.mark.parametrize('scores,alpha', [([], .1), ([np.nan], .1), ([-1], .1), ([1], 0), ([1], 1)])
def test_quantile_rejects_invalid_parameters(scores, alpha):
    with pytest.raises(ValueError):
        upper_quantile(scores, alpha)


def test_yaw_frame_preserves_ned_and_distance():
    p = np.array([[11., 20., 33.], [10., 22., 29.]])
    local = local_positions(p, [10., 20., 30.], np.pi/2)
    np.testing.assert_allclose(local, [[0., -1., 3.], [2., 0., -1.]], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(local, axis=1), np.linalg.norm(p-[10, 20, 30], axis=1))


def test_ecp_takes_two_independent_minima_not_max_obstacle_error():
    # Object 1's error is 5m but object 0 already limits clearance to 2m.
    pred = np.array([[[[[2., 0, 0], [2., 0, 0]], [[10., 0, 0], [10., 0, 0]]]]])
    actual = pred.copy()
    actual[0, 0, 1, :, 0] = 5.
    score = ecp_scene_scores(np.zeros((2, 3)), pred, actual,
                             np.zeros((1, 1, 2)), np.ones((1, 1, 2), bool), 0.)
    assert score[0] == 0
    assert sphere_scene_scores(pred, actual, np.ones((1, 1, 2), bool))[0] == 5.


def test_different_radius_nearest_gap_can_switch_obstacle():
    pred = np.array([[[[[3., 0, 0], [3., 0, 0]], [[10., 0, 0], [10., 0, 0]]]]])
    actual = pred.copy()
    actual[0, 0, 1, :, 0] = 9.
    scores = ecp_scene_scores(np.zeros((2, 3)), pred, actual,
                              np.array([[[0., 8.]]]), np.ones((1, 1, 2), bool), 0.)
    np.testing.assert_allclose(scores, [1.])


def test_swept_geometry_detects_between_sample_approach():
    pred = np.array([[[[[2., 2., 0], [2., -2., 0]]]]])
    actual = pred.copy()
    actual[..., 0] = 1.
    scores = ecp_scene_scores(np.zeros((2, 3)), pred, actual,
                              np.zeros((1, 1, 1)), np.ones((1, 1, 1), bool), 0.)
    np.testing.assert_allclose(scores, [1.])


@pytest.mark.parametrize('seed', range(10))
def test_candidate_score_bounded_by_same_scene_ball_and_nonnegative(seed):
    rng = np.random.default_rng(seed)
    pred = rng.normal(size=(5, 3, 2, 9, 3))*10
    labels = pred + rng.normal(size=pred.shape)
    mask = rng.random(pred.shape[:3]) > .2
    scores = ecp_scene_scores(rng.normal(size=(9, 3)), pred, labels,
                              np.ones(pred.shape[:3]), mask, .8)
    assert np.all(scores >= 0)
    assert np.all(scores <= sphere_scene_scores(pred, labels, mask)+1e-12)


def test_missing_slots_do_not_become_zero_errors_or_nan_scores():
    pred = np.full((2, 1, 1, 2, 3), np.nan)
    labels = pred.copy()
    score = ecp_scene_scores(np.zeros((2, 3)), pred, labels,
                             np.ones((2, 1, 1)), np.zeros((2, 1, 1), bool), .8)
    np.testing.assert_array_equal(score, [0., 0.])


@pytest.mark.parametrize('method', ['B1', 'B2'])
def test_library_configuration_and_split_checks(tmp_path, method):
    with pytest.raises(ValueError, match='calibration split'):
        RiskPolicy(method, library(tmp_path/'wrong.npz', split='test'))
    with pytest.raises(ValueError, match='insufficient'):
        RiskPolicy(method, library(tmp_path/'small.npz', count=12), .05)
    risk = RiskPolicy(method, library(tmp_path/'valid.npz'))
    assert risk.metadata['calibration_scene_count'] == 24
    assert risk.metadata['online_update'] is False


@pytest.mark.parametrize('method', ['B1', 'B2'])
def test_nonzero_absolute_clock_uses_relative_physics_offsets(tmp_path, method):
    policy = RiskPolicy(method, library(tmp_path/'valid.npz'))
    cfg = Stage2Config()
    states = np.zeros((25, 16))
    states[:, 0] = np.arange(25.)
    states[:, 1] = np.arange(25.)
    states[:, 3] = 30.
    terrain = HeightField.flat(200, 200, 90.)
    track = Track(0, 1., [40., 41., 42.], [np.array([80., 0, 30.])]*3)
    context = ForecastContext(cfg)
    base = trajectory_clearance(states, terrain, [track], 42., cfg, context)
    adjusted, record = policy.adjust(states, terrain, [track], 42., cfg, context, base)
    assert adjusted <= base
    assert record['quantile_m'] >= 0
    assert np.isfinite(adjusted)


def test_b0_hook_preserves_full_plant_plan_and_shortlist():
    cfg = replace(Stage2Config(), horizon_s=.2, control_dt=.2)
    plant = RemusPlant(depth=30., rpm=1100, eta0=np.array([50., 50., 30., 0., 0., 0.]), initialize_references=True)
    goal = np.array([100., 20., 30.])
    terrain = HeightField.flat(200, 200, 90.)
    first = RemusDWA(cfg).plan(plant, goal, terrain, [], 0.)
    second = RemusDWA(cfg, risk_policy=RiskPolicy('B0')).plan(plant, goal, terrain, [], 0.)
    np.testing.assert_array_equal(first.action.as_array(), second.action.as_array())
    np.testing.assert_array_equal(first.states, second.states)
    assert first.shortlist_indices == second.shortlist_indices
    assert first.minimum_predicted_clearance_m == second.minimum_predicted_clearance_m
    assert len(second.risk_records) == cfg.candidate_budget


@pytest.mark.parametrize('method', ['B1', 'B2'])
def test_risk_does_not_inflate_terrain_or_unseen_objects(tmp_path, method):
    risk = RiskPolicy(method, library(tmp_path/'cal.npz'))
    cfg = Stage2Config()
    states = np.zeros((25, 16))
    states[:, 0] = np.arange(25.)
    states[:, 1:4] = [50., 50., 87.]
    terrain = HeightField.flat(200, 200, 90.)
    track = Track(0, 1., [0.], [np.array([180., 180., 30.])])
    context = ForecastContext(cfg)
    base = trajectory_clearance(states, terrain, [track], 0., cfg, context)
    adjusted, _ = risk.adjust(states, terrain, [track], 0., cfg, context, base)
    assert adjusted == base  # The limiting terrain has no prediction error ball.
    unseen, record = risk.adjust(states, terrain, [], 0., cfg, context, base)
    assert unseen == base
    assert record['quantile_m'] == 0.


def test_emergency_uses_same_risk_check_and_keeps_real_momentum():
    class LargeRisk:
        def adjust(self, states, terrain, tracks, now, cfg, context, base):
            return base-1000., {'quantile_m': 1000.}
    cfg = replace(Stage2Config(), horizon_s=.2, control_dt=.2)
    plant = RemusPlant(depth=30., rpm=1100, eta0=np.array([50., 50., 30., 0., 0., 0.]), initialize_references=True)
    plant.nu[0] = 1.5
    original = plant.nu.copy()
    plan = RemusDWA(cfg, risk_policy=LargeRisk()).plan(plant, [150., 50., 30.],
                                                    HeightField.flat(200, 200, 90.), [], 0.)
    assert plan.emergency and not plan.emergency_verified_safe
    assert plan.action.rpm == 0
    assert plan.states[-1, 7] > 1.  # RPM0 is coasting, never instant stopping.
    assert len(plan.risk_records) == cfg.candidate_budget+1
    np.testing.assert_array_equal(plant.nu, original)


def test_candidate_off_axis_error_is_smaller_than_independent_ball():
    pred = np.array([[[[[0., 20., 0.], [0., 20., 0.]]]]])
    labels = pred.copy()
    labels[..., 0] += 5.  # Tangential error moves obstacle further away.
    score = ecp_scene_scores(np.zeros((2, 3)), pred, labels,
                             np.ones((1, 1, 1)), np.ones((1, 1, 1), bool), .8)
    assert score[0] == 0.
    assert sphere_scene_scores(pred, labels, np.ones((1, 1, 1), bool))[0] == 5.


def test_missing_scene_labels_cannot_silently_become_zero(tmp_path):
    original = library(tmp_path/'original.npz')
    with np.load(original, allow_pickle=False) as data:
        arrays = dict(data)
    arrays['mask'][0] = False
    broken = tmp_path/'broken.npz'
    np.savez_compressed(broken, **arrays)
    with pytest.raises(ValueError, match='missing labels'):
        RiskPolicy('B2', broken)
