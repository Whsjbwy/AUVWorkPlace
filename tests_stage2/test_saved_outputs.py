"""Offline artifact loading cannot silently invent truth or overwrite a run."""
from dataclasses import replace
import json
import numpy as np
import pytest

from auv3d.stage2_config import Stage2Config
from auv3d.stage2_simulation import run_remus_episode, save_remus_episode
from auv3d.stage2_outputs import load_saved_remus_episode, render_saved_remus_episode


def test_legacy_loader_verifies_obstacle_truth_and_adds_terminal_without_changing_source(tmp_path):
    cfg = replace(Stage2Config(), max_time_s=1., control_dt=.5, horizon_s=2.,
                  observation_period_s=.5, candidate_budget=1, warmup_s=0.)
    original = run_remus_episode(cfg,'crossing3d')
    source = tmp_path/'source'
    save_remus_episode(original,source,plot=False)
    summary = json.loads((source/'summary.json').read_text())
    summary.pop('goal_ned_m',None)
    (source/'summary.json').write_text(json.dumps(summary))
    old_snapshots = [row for row in original['obstacles'] if row['time_s'] < 1.-1e-9]
    old_text = json.dumps(old_snapshots)
    (source/'obstacles.json').write_text(old_text)
    result, manifest = load_saved_remus_episode(source)
    np.testing.assert_array_equal(result['states'],original['states'])
    np.testing.assert_array_equal(result['goal'],original['goal'])
    assert result['obstacles'][-1]['time_s'] == 1.
    assert manifest['terminal_snapshot_added'] is True
    assert manifest['verified_original_snapshot_count'] == 2
    assert manifest['terminal_snapshot_time_matches_actual_endpoint'] is True
    assert manifest['vehicle_controller_state_restored'] is False
    assert manifest['new_closed_loop_simulation_executed'] is False
    assert manifest['new_collision_safety_validation_performed'] is False
    assert 'legacy summary' in manifest['goal_source']
    assert (source/'obstacles.json').read_text() == old_text

    # An incompatible stored sample must fail before a replacement is invented.
    altered = json.loads(old_text)
    altered[0]['positions'][0][0] += 1.
    (source/'obstacles.json').write_text(json.dumps(altered))
    with pytest.raises(ValueError,match='differs from saved snapshot'):
        load_saved_remus_episode(source)


def test_offline_rendering_refuses_to_overwrite_any_existing_output(tmp_path):
    out = tmp_path/'existing_result'
    out.mkdir()
    marker = out/'existing_record.txt'
    marker.write_text('retain this original experiment')
    with pytest.raises(FileExistsError,match='choose a new offline rendering directory'):
        render_saved_remus_episode(tmp_path/'missing_source',out)
    assert marker.read_text() == 'retain this original experiment'
    assert len(list(out.iterdir())) == 1
