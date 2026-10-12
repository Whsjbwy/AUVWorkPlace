from dataclasses import replace
import json
import numpy as np
import pytest

from auv3d.stage2_config import Stage2Config
from auv3d.stage2_simulation import run_remus_episode, save_remus_episode
from scripts_stage5.render_saved import load_saved_result, render_saved


def test_saved_render_retains_exact_execution_and_actual_injected_predictor(tmp_path):
    cfg = replace(Stage2Config(),warmup_s=0.,max_time_s=.4,control_dt=.2,
                  horizon_s=.4,candidate_budget=1)
    from auv3d.stage5_forecasting import make_forecaster
    episode = run_remus_episode(cfg,'crossing3d',forecaster=make_forecaster('cv',cfg))
    raw = tmp_path/'raw'
    save_remus_episode(episode,raw,plot=False)
    loaded = load_saved_result(raw)
    np.testing.assert_array_equal(loaded['states'],episode['states'])
    assert loaded['config']['obstacle_predictor'] == 'cv'
    assert json.loads((raw/'config.json').read_text())['obstacle_predictor'] == 'ca'
    assert loaded['obstacles'] == episode['obstacles']
    assert loaded['selected_plans'][-1]['time_s'] == episode['selected_plans'][-1]['time_s']
    record = render_saved(raw,tmp_path/'figures',animate=False)
    assert record['new_simulation_executed'] is False
    assert (tmp_path/'figures/replay_3d.html').stat().st_size > 1000
    with pytest.raises(FileExistsError):
        render_saved(raw,tmp_path/'figures',animate=False)
