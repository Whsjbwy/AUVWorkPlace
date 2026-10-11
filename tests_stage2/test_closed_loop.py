from dataclasses import replace
import json
import numpy as np
import pytest

from auv3d.stage2_config import Stage2Config
from auv3d.stage2_simulation import run_remus_episode,save_remus_episode
from auv3d.scenarios import Scene
from auv3d.obstacles import LinearObstacles
from auv3d.terrain import HeightField


def tiny_config(**kwargs):
    return replace(Stage2Config(),warmup_s=0.,max_time_s=3.,control_dt=1.,
                   horizon_s=2.,candidate_budget=1,**kwargs)


def test_closed_loop_reads_physical_feedback_and_keeps_predictions_separate(monkeypatch):
    import auv3d.stage2_simulation as module
    actual_calls=[]
    original=module.RemusDWA.plan
    def record(self,plant,goal,terrain,tracks,now):
        actual_calls.append((now,plant.eta.copy(),plant.nu.copy()))
        return original(self,plant,goal,terrain,tracks,now)
    monkeypatch.setattr(module.RemusDWA,'plan',record)
    result=run_remus_episode(tiny_config(),'empty3d')
    assert result['summary']['backend']=='remus6dof'
    assert result['summary']['status']=='timeout'
    assert result['states'].shape==(151,16)
    for stamp,eta,nu in actual_calls:
        row=result['states'][round(stamp/.02)]
        np.testing.assert_array_equal(row[1:7],eta)
        np.testing.assert_array_equal(row[7:13],nu)
    assert result['summary']['prediction_position_max_error_m']<1e-10
    assert len(result['selected_plans'])==3
    assert result['controls'].shape==(150,16)
    assert result['summary']['failure_reason']


def test_observation_delay_retains_measurement_time_without_future_truth(monkeypatch):
    import auv3d.stage2_simulation as module
    received=[]
    original=module.RemusDWA.plan
    def record(self,plant,goal,terrain,tracks,now):
        received.append((now,[list(track.times) for track in tracks]))
        assert all(track.times[-1]<=now-1.+1e-9 for track in tracks)
        return original(self,plant,goal,terrain,tracks,now)
    monkeypatch.setattr(module.RemusDWA,'plan',record)
    result=run_remus_episode(tiny_config(observation_delay_s=1.),'static3d')
    assert received[0][1]==[]
    assert received[1][1][0]==[0.]
    assert result['summary']['maximum_observed_obstacles']==1


def test_actual_three_dimensional_collision_reports_failure(monkeypatch):
    import auv3d.stage2_simulation as module
    scene=Scene('test',HeightField.flat(),LinearObstacles([[30.,100.,35.]],[[0,0,0]],[4.]),
                np.array([30.,100.,35.,0.,0.,0.]),np.array([210.,100.,35.]))
    monkeypatch.setattr(module,'make_stage2_scenario',lambda *_:scene)
    result=run_remus_episode(tiny_config())
    assert result['summary']['status']=='obstacle_collision'
    assert not result['summary']['reached_goal']
    assert result['summary']['minimum_model_clearance_m']<0
    assert result['events'][0]['kind']=='collision'


def test_dynamic_obstacle_clock_matches_physical_execution():
    result=run_remus_episode(tiny_config(),'maneuvering3d')
    assert result['summary']['all_states_finite']
    assert [item['time_s'] for item in result['obstacles']]==[0.,1.,2.,3.]
    assert result['obstacles'][-1]['time_s']==result['states'][-1,0]
    assert np.allclose(np.diff(result['states'][:,0]),.02)
    assert not np.array_equal(result['obstacles'][0]['positions'],result['obstacles'][-1]['positions'])


def test_trajectory_controls_error_and_events_saved_without_overwrite(tmp_path):
    result=run_remus_episode(tiny_config(),'crossing3d')
    save_remus_episode(result,tmp_path/'run',plot=False)
    for name in ['states.csv','controls.csv','planning.csv','prediction_errors.csv',
                 'events.json','selected_plans.json','obstacles.json','terrain.npz','summary.json','config.json']:
        assert (tmp_path/'run'/name).is_file()
    planned=json.loads((tmp_path/'run/selected_plans.json').read_text())
    assert planned[0]['validation_backend']=='original_remus6dof'
    with pytest.raises(FileExistsError):save_remus_episode(result,tmp_path/'run',plot=False)


def test_short_reachable_closed_loop_task_succeeds(monkeypatch):
    import auv3d.stage2_simulation as module
    original=module.make_stage2_scenario
    def shorter(name,cfg):
        scene=original(name,cfg)
        scene.goal=scene.start_eta[:3]+np.array([5.,0.,0.])
        return scene
    monkeypatch.setattr(module,'make_stage2_scenario',shorter)
    cfg=replace(tiny_config(),warmup_s=20.,max_time_s=8.,goal_tolerance_m=.8)
    result=run_remus_episode(cfg,'empty3d')
    assert result['summary']['status']=='success'
    assert result['summary']['minimum_model_clearance_m']>0


def test_arrival_does_not_erase_unsafe_emergency_validation(monkeypatch):
    import auv3d.stage2_simulation as module
    original_scene=module.make_stage2_scenario
    original_plan=module.RemusDWA.plan
    def shorter(name,cfg):
        scene=original_scene(name,cfg)
        scene.goal=scene.start_eta[:3]+np.array([5.,0.,0.])
        return scene
    def unsafe(self,*args):
        result=original_plan(self,*args)
        # Inject a failed emergency forecast without changing actual collision
        # geometry or execution; outcome classification must retain the failure.
        return replace(result,feasible=False,emergency=True,emergency_verified_safe=False,
                       reason='no_verified_feasible_reference_emergency_not_safe')
    monkeypatch.setattr(module,'make_stage2_scenario',shorter)
    monkeypatch.setattr(module.RemusDWA,'plan',unsafe)
    cfg=replace(tiny_config(),warmup_s=20.,max_time_s=8.,goal_tolerance_m=.8)
    summary=run_remus_episode(cfg,'empty3d')['summary']
    assert summary['reached_goal']
    assert summary['status']=='success'  # Navigation outcome, separately labelled.
    assert not summary['avoidance_acceptance_passed']
    assert summary['safety_validation_status']=='failed_unsafe_emergency_forecast'
    assert summary['safety_failure_reason']
