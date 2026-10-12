"""Depth-envelope-dominated summaries must contain JSON-native booleans."""
from dataclasses import replace
import json

import pytest

from auv3d.stage2_config import Stage2Config
from auv3d import stage2_simulation


@pytest.mark.parametrize("immediate_goal", [False, True])
def test_legal_depth_envelope_summary_serializes_boolean_results(tmp_path, monkeypatch,
                                                               immediate_goal):
    cfg = replace(Stage2Config(), max_time_s=.04, depth_limit_m=50.,
                  terrain_depth_m=200.)
    if immediate_goal:
        # A test-only zero-distance task covers the success branch while
        # retaining the original arrival tolerance and physical model. It is
        # not included in research navigation statistics.
        original_factory = stage2_simulation.make_stage2_scenario
        def same_position_goal(name, settings):
            scene = original_factory(name, settings)
            scene.goal = scene.start_eta[:3].copy()
            return scene
        monkeypatch.setattr(stage2_simulation, "make_stage2_scenario", same_position_goal)
    result = stage2_simulation.run_remus_episode(cfg, "empty3d")
    summary = result["summary"]
    assert type(summary["actual_safety_margin_met"]) is bool
    assert type(summary["avoidance_acceptance_passed"]) is bool
    assert type(summary["reached_goal"]) is bool
    assert summary["actual_safety_margin_met"] is True
    assert summary["reached_goal"] is immediate_goal
    assert summary["avoidance_acceptance_passed"] is immediate_goal
    target = tmp_path/"actual_episode"
    stage2_simulation.save_remus_episode(result, target, animate=False, plot=False)
    saved = json.loads((target/"summary.json").read_text(encoding="utf-8"))
    for field in ("actual_safety_margin_met", "avoidance_acceptance_passed", "reached_goal"):
        assert type(saved[field]) is bool
        assert saved[field] is summary[field]
