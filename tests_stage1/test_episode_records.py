from dataclasses import replace
import json
import numpy as np

from auv3d.config import ExperimentConfig
from auv3d.simulation import run_episode, save_episode


def test_selected_rollouts_preserve_planning_time_and_execution_label(tmp_path):
    cfg = ExperimentConfig()
    cfg = replace(cfg, planner=replace(cfg.planner, max_time_s=1.0))
    result = run_episode(cfg)
    assert result['summary']['status'] == 'timeout'
    assert 'max_time_s' in result['summary']['failure_reason']
    assert 'P25' in result['summary']['execution_model']
    for plan in result['selected_plans']:
        executed = result['states'][np.isclose(result['states'][:, 0], plan['time_s'])][0]
        np.testing.assert_allclose(np.asarray(plan['states'])[0, :3], executed[1:4])
        assert np.asarray(plan['states']).shape == (46, 6)
        assert np.isfinite(plan['states']).all()
    save_episode(result, tmp_path / 'episode', plot=False)
    stored = json.loads((tmp_path / 'episode/selected_plans.json').read_text())
    assert stored == result['selected_plans']
