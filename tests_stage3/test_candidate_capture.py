"""The fixed offline capture cannot command the live REMUS or choose outcomes."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from auv3d.adapters.remus_guidance import ReferenceAction, RemusGuidance
from auv3d.remus import RemusPlant
from auv3d.remus_planner import RemusDWA
from auv3d.sensors import Track
from auv3d.stage2_config import Stage2Config
from auv3d.stage2_simulation import run_remus_episode
from auv3d.terrain import HeightField
from scripts_stage3.candidate_diagnostics import FirstFailureCapture


def test_first_failure_capture_has_full_isolated_controller_previous_goal_and_uncertainty(tmp_path):
    cfg = replace(Stage2Config(), horizon_s=.4, control_dt=.2)
    plant = RemusPlant(depth=30., heading_deg=0., rpm=1100.,
                       eta0=[30., 100., 30., .03, -.02, .04], initialize_references=True)
    plant.nu[0] = 1.5
    plant.actual[2] = 1000.
    terrain = HeightField.flat(260., 200., 90.)
    track = Track(4, 4., [0.], [np.array([150., 100., 30.])])
    scene = SimpleNamespace(terrain=terrain, goal=[210., 100., 30.])
    capture = FirstFailureCapture(cfg)
    planner = RemusDWA(cfg)
    planner.set_memory_uncertainty({4: 3.25})
    first = planner.plan(plant, scene.goal, terrain, [track], 0.)
    capture(plant, [track], 0., first, scene)
    assert capture.frame is None
    RemusGuidance.step(plant, ReferenceAction(34., .2, 900.), .02)
    second = planner.plan(plant, [190., 110., 25.], terrain, [track], plant.time_s)
    second = replace(second, feasible=False)  # Defined unit selection fixture.
    capture(plant, [track], plant.time_s, second, scene)
    frozen_eta = plant.eta.copy()
    frozen_integral = plant.vehicle.z_int
    plant.eta[0] = -999.
    plant.vehicle.z_int = -999.
    track.positions[0][:] = -999.
    np.testing.assert_array_equal(capture.frame['plant'].eta, frozen_eta)
    assert capture.frame['plant'].vehicle.z_int == frozen_integral
    np.testing.assert_array_equal(capture.frame['previous_action'].as_array(), first.command)
    np.testing.assert_array_equal(capture.frame['planning_goal'], [190., 110., 25.])
    assert capture.frame['uncertainty_by_id'] == {4: 3.25}
    evidence = capture.evaluate(tmp_path)
    saved = json.loads((tmp_path/'candidate_first_failure_diagnostic.json').read_text(encoding='utf-8'))
    assert evidence['captured'] and evidence['checked_frame_count'] == 1
    assert saved['time_s'] == .02 and saved['previous_action'] == first.command.tolist()
    assert saved['captured_plant']['vehicle_fields']['z_int'] == frozen_integral
    assert saved['planning_goal_ned_m'] != saved['final_task_goal_ned_m']
    assert saved['uncertainty_by_id'] == {'4': 3.25}
    assert saved['tracks'][0]['positions'] == [[150., 100., 30.]]
    assert saved['grid']['candidate_count'] == 45
    assert set(saved['grid']['policy_shortlist_feasible_counts']) == {'diverse', 'ranked', 'coverage'}


def test_observer_leaves_actual_closed_loop_execution_unchanged(monkeypatch):
    cfg = replace(Stage2Config(), horizon_s=.4, control_dt=.2, max_time_s=.4, warmup_s=0.)
    actual_plan = RemusDWA.plan

    def mark_second_call_failed(self, plant, goal, terrain, tracks, now):
        result = actual_plan(self, plant, goal, terrain, tracks, now)
        return replace(result, feasible=False) if now >= .2 else result

    monkeypatch.setattr(RemusDWA, 'plan', mark_second_call_failed)
    plain = run_remus_episode(cfg, 'empty3d')
    capture = FirstFailureCapture(cfg)
    observed = run_remus_episode(cfg, 'empty3d', on_plan=capture)
    np.testing.assert_array_equal(observed['states'], plain['states'])
    np.testing.assert_array_equal(observed['controls'], plain['controls'])
    np.testing.assert_array_equal(observed['clearance'], plain['clearance'])
    assert capture.frame['now'] == .2
    assert capture.frame['previous_action'].as_array().tolist() == plain['selected_plans'][0]['action']


def test_no_failure_capture_does_not_invent_full_grid_safety(tmp_path, monkeypatch):
    import scripts_stage3.candidate_diagnostics as module

    def forbidden_rollout(*args, **kwargs):
        pytest.fail('no failed frame means no offline rollout')

    monkeypatch.setattr(module, 'evaluate_reference_grid', forbidden_rollout)
    summary = FirstFailureCapture(Stage2Config()).evaluate(tmp_path)
    assert not summary['captured'] and summary['checked_frame_count'] == 0
    assert summary['full_grid_safe_count'] is None
