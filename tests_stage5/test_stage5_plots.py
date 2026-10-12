from dataclasses import replace
import copy
import json

import pytest

from auv3d.stage2_config import Stage2Config
from auv3d.stage2_simulation import run_remus_episode, save_remus_episode
from auv3d.stage5_plots import plot_comparison


def test_real_saved_remus_plots_include_terrain_separate_forecasts_and_errors(tmp_path):
    cfg = replace(Stage2Config(),max_time_s=.4,control_dt=.2,horizon_s=.4,
                  warmup_s=0.,candidate_budget=1)
    episode = run_remus_episode(cfg,'static3d')
    episode['summary']['scene_metadata']['scene_id'] = 'unit_scene'
    for name in ('A0','A5'):
        save_remus_episode(copy.deepcopy(episode),tmp_path/name,plot=False)
    sources = plot_comparison({name:tmp_path/name for name in ('A0','A5')},
                              tmp_path/'comparison',anchor_s=.2)
    assert sources['available_forecasts'] == {'A0':True,'A5':True}
    for name in ('actual_trajectories_3d.png','predicted_and_actual_3d.png',
                 'clearance_execution_error_control.png'):
        assert (tmp_path/'comparison'/name).stat().st_size > 10000
    with pytest.raises(FileExistsError):
        plot_comparison({'A0':tmp_path/'A0'},tmp_path/'comparison')
    summary = json.loads((tmp_path/'A5/summary.json').read_text())
    summary['scene_metadata']['scene_id'] = 'different_geometry'
    (tmp_path/'A5/summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError,match='geometry'):
        plot_comparison({'A0':tmp_path/'A0','A5':tmp_path/'A5'},tmp_path/'wrong')


def test_early_terminated_forecast_is_reported_missing_without_substituting_time(tmp_path):
    cfg = replace(Stage2Config(),max_time_s=.2,control_dt=.2,horizon_s=.4,
                  warmup_s=0.,candidate_budget=1)
    save_remus_episode(run_remus_episode(cfg,'empty3d'),tmp_path/'A0',plot=False)
    record = plot_comparison({'A0':tmp_path/'A0'},tmp_path/'late',anchor_s=40.)
    assert record['available_forecasts'] == {'A0':False}


@pytest.mark.parametrize('mapping,anchor', [({},0),({'P25':'no-file'},0),({'A0':'no-file'},-1)])
def test_invalid_comparison_rejected(mapping,anchor,tmp_path):
    with pytest.raises(ValueError):
        plot_comparison(mapping,tmp_path/'invalid',anchor_s=anchor)
