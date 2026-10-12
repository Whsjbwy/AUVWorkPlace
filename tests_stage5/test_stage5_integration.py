"""The opt-in Stage5 methods preserve the original engineering reference."""
from dataclasses import replace
import numpy as np

from auv3d.stage2_config import Stage2Config
from auv3d.stage2_simulation import run_remus_episode
from auv3d.stage5_forecasting import make_forecaster
from auv3d.stage5_risk import make_policy


def test_A0_preserves_original_dynamics_decisions_observations_and_collision_geometry():
    cfg = replace(Stage2Config(),horizon_s=2.,control_dt=1.,max_time_s=3.,
                  warmup_s=0.,candidate_budget=3,sensor_noise_std_m=.02)
    original = run_remus_episode(cfg,'static3d')
    stage5 = run_remus_episode(cfg,'static3d',risk_policy=make_policy('A0',cfg=cfg),
                               forecaster=make_forecaster('ca',cfg))
    for field in ('states','controls','clearance','prediction_errors'):
        np.testing.assert_array_equal(stage5[field],original[field])
    assert stage5['observations'] == original['observations']
    for actual,expected in zip(stage5['selected_plans'],original['selected_plans'],strict=True):
        assert actual['action'] == expected['action']
        np.testing.assert_array_equal(actual['states'],expected['states'])
    assert stage5['summary']['actual_safety_margin_met'] == original['summary']['actual_safety_margin_met']
    assert not stage5['summary']['calibrated_radii_applied']


def test_injected_predictor_is_identified_by_its_actual_implementation():
    cfg = replace(Stage2Config(),horizon_s=.4,control_dt=.2,max_time_s=.2,
                  warmup_s=0.,candidate_budget=1)
    result = run_remus_episode(cfg,'static3d',forecaster=make_forecaster('filtered_cv',cfg),
                               risk_policy=make_policy('A0',cfg=cfg,predictor='filtered_cv'))
    assert result['summary']['obstacle_predictor'] == 'filtered_cv'
    assert result['summary']['prediction'].startswith('filtered_cv obstacle forecast')
    assert result['config']['obstacle_predictor'] == 'ca'  # retained legacy field
    assert result['planning_diagnostics'][0]['forecast_records'][0]['predictor'] == 'filtered_cv'
