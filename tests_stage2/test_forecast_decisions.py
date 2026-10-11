"""Functional safeguards for offline forecast attribution from saved points."""
from dataclasses import replace
import json

import numpy as np
import pytest

from auv3d.obstacles import ObstacleSnapshot
from auv3d.stage2_config import Stage2Config
from auv3d.stage2_simulation import run_remus_episode, save_remus_episode
from scripts_stage2.analyze_forecast_decisions import analyze_case, pointwise_decision


def synthetic_saved_points():
    """Geometry fixture: unknown initial obstacle velocity makes CA miss arrival.

    These synthetic points test attribution, without claiming a vehicle response
    experiment. The separate saved-run test below uses the actual original plant.
    """
    cfg = replace(Stage2Config(), horizon_s=.8, control_dt=.8,
                  prediction_output_dt=.2, observation_period_s=.2, warmup_s=0.)
    offsets = np.arange(5)*.2
    rows = np.zeros((5, 16))
    rows[:, 0] = offsets
    rows[:, 1:4] = [30., 100., 35.]
    plan = {'time_s':0., 'states':rows.tolist(), 'feasible':True,
            'validation_backend':'original_remus6dof'}
    detail = {'time_s':0., 'feasible':True, 'tracks':[
        {'object_id':1, 'radius':1., 'times_s':[0.], 'positions':[[34., 100., 35.]]}]}
    truth, actual = {}, {}
    for offset in offsets:
        tick = round(offset/cfg.physics_dt)
        truth[tick] = ObstacleSnapshot(np.array([1, 2]),
            np.array([[34.-2.5*offset, 100., 35.], [33.-.5*offset, 100., 35.]]),
            np.ones(2))
        actual[tick] = np.array([30., 100., 35.])
    observations = [{'delivered_time_s':-5., 'objects':[{'object_id':2}]}]
    return cfg, plan, detail, truth, actual, observations


@pytest.mark.parametrize('end_time,executed_c_count', [(.4, 0), (.8, 2)])
def test_tracked_forecast_counterexample_requires_an_executed_matching_vehicle_point(end_time, executed_c_count):
    cfg, plan, detail, truth, actual, observations = synthetic_saved_points()
    decision, hazards, samples = pointwise_decision(
        plan, detail, 1.4, end_time, cfg, truth, actual, observations)
    assert decision['tracked_counterexample_held_sample_count'] == executed_c_count
    assert decision['tracked_counterexample_full_horizon_sample_count'] == 2
    assert decision['vehicle_held_sample_execution_max_error_m'] == 0.
    tracked = [row for row in hazards if row['tracked_at_decision']]
    assert all(row['category'] == 'tracked_ca_forecast_false_safe' for row in tracked)
    assert all(row['true_physical_point_clearance_m'] < 1. for row in tracked)
    assert all(row['ca_inflated_point_clearance_m'] > 1. for row in tracked)
    assert all(row['category'] == 'previously_delivered_but_not_currently_tracked'
               for row in hazards if not row['tracked_at_decision'])
    if end_time == .4:
        assert all(row['scope'] == 'unexecuted_counterfactual_future' for row in tracked)
        assert all(row['vehicle_sample_execution_error_m'] is None for row in tracked)
    assert len(samples) == 5


def test_execution_mismatch_blocks_a_forecast_only_attribution():
    cfg, plan, detail, truth, actual, observations = synthetic_saved_points()
    actual[10][0] += .1
    with pytest.raises(ValueError, match='confounded'):
        pointwise_decision(plan, detail, 1.4, .8, cfg, truth, actual, observations)


def tiny_config():
    return replace(Stage2Config(), horizon_s=.4, control_dt=.2, max_time_s=.4,
                   observation_period_s=.2, prediction_output_dt=.2,
                   warmup_s=0., candidate_budget=1)


def test_saved_original_remus_points_and_same_seed_truth_are_verified(tmp_path):
    source, out = tmp_path/'actual', tmp_path/'offline'
    save_remus_episode(run_remus_episode(tiny_config(), 'empty3d'), source, plot=False)
    summary = analyze_case(source, out)
    assert summary['feasible_decisions_analyzed'] == 2
    assert summary['verified_recorded_truth_snapshots'] == 3
    assert summary['maximum_recorded_truth_position_difference_m'] == 0.
    assert summary['vehicle_held_sample_execution_max_error_m'] == 0.
    assert summary['new_vehicle_rollouts'] == 0 and not summary['feedback_to_planner']
    assert not summary['counterexample_C_demonstrated_at_saved_executed_sample']
    assert not summary['absence_of_sampled_counterexample_proves_safety']
    assert (out/'tracked_forecast_samples.csv').is_file()
    with pytest.raises(FileExistsError):
        analyze_case(source, out)


def test_inconsistent_recorded_obstacle_truth_cannot_support_attribution(tmp_path):
    source, out = tmp_path/'actual', tmp_path/'offline'
    save_remus_episode(run_remus_episode(tiny_config(), 'static3d'), source, plot=False)
    path = source/'obstacles.json'
    snapshots = json.loads(path.read_text(encoding='utf-8'))
    snapshots[0]['positions'][0][0] += 1.
    path.write_text(json.dumps(snapshots), encoding='utf-8')
    with pytest.raises(ValueError, match='differs from an actual recorded snapshot'):
        analyze_case(source, out)
    assert not out.exists()


def test_nonzero_execution_delay_is_not_silently_attributed_as_an_immediate_reference(tmp_path):
    source = tmp_path/'actual'
    save_remus_episode(run_remus_episode(tiny_config(), 'empty3d'), source, plot=False)
    path = source/'config.json'
    cfg = json.loads(path.read_text(encoding='utf-8'))
    cfg['command_delay_s'] = .2
    path.write_text(json.dumps(cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='command_delay_s=0'):
        analyze_case(source, tmp_path/'offline')
