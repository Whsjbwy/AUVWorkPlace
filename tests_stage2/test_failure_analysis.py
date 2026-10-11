"""Offline budget diagnosis preserves causality and captured controller state."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np

from auv3d.adapters.remus_guidance import ReferenceAction
from auv3d.remus import RemusPlant
from auv3d.scenarios import make_stage2_scenario
from auv3d.sensors import Track
from auv3d.stage2_config import Stage2Config
from auv3d.stage2_failure_analysis import (
    FailureObserver, analyze_obstacle_forecasts, captured_plant_record, classify_failure,
    analyze_actual_margin,
)


def test_failure_categories_do_not_claim_global_unavoidability_or_safe_emergency():
    row={'prediction_failure':None}
    missing=classify_failure({'feasible_count':1,'candidates':[row]},False)
    assert missing['category']=='shortlist_omission'
    assert missing['unsafe_emergency'] is True
    no_grid=classify_failure({'feasible_count':0,'candidates':[row]},True)
    assert no_grid['category']=='no_safe_reference_in_current_discrete_grid'
    assert 'does not prove global unavoidability' in no_grid['scope']
    failed=classify_failure({'feasible_count':0,'candidates':[{'prediction_failure':'nonfinite'}]},False)
    assert failed['category']=='no_verified_reference_with_numerical_prediction_failures'


def test_observer_keeps_independent_full_controller_and_track_state():
    cfg=replace(Stage2Config(),warmup_s=0.)
    scene=make_stage2_scenario('crossing3d',cfg)
    plant=RemusPlant(eta0=scene.start_eta,current_speed=0.)
    plant.vehicle.z_int=1.23
    plant.vehicle.e_psi_int=4.56
    track=Track(0,4.,[0.],[scene.obstacles.snapshot().positions[0]])
    action=ReferenceAction(35.,0.,1100.)
    plan=SimpleNamespace(action=action,feasible=True,emergency_verified_safe=False,
                         reason='verified_reference',minimum_predicted_clearance_m=2.,
                         planning_ms=1.,verified_count=9)
    observer=FailureObserver(cfg)
    observer(plant,[track],0.,plan,scene)
    record=captured_plant_record(observer.frames[0].plant)
    assert record['vehicle_fields']['z_int']==1.23
    assert record['vehicle_fields']['e_psi_int']==4.56
    assert record['vehicle_fields']['z_d']==plant.vehicle.z_d
    observer.frames[0].plant.eta[0]=-999.
    observer.frames[0].plant.vehicle.z_int=999.
    observer.frames[0].tracks[0].positions[0][:]=-999.
    assert plant.eta[0]==30.
    assert plant.vehicle.z_int==1.23
    assert np.all(track.positions[0]>0.)


def test_obstacle_ca_error_uses_same_id_and_future_time_separately_from_vehicle(tmp_path):
    cfg=replace(Stage2Config(),warmup_s=0.)
    scene=make_stage2_scenario('crossing3d',cfg)
    initial=scene.obstacles.snapshot()
    scene.obstacles.step(1.)
    later=scene.obstacles.snapshot()
    track=Track(0,4.,[0.,1.],[initial.positions[0],later.positions[0]])
    plant=RemusPlant(eta0=scene.start_eta,current_speed=0.)
    plant.time_s=1.
    action=ReferenceAction(35.,0.,1100.)
    plan=SimpleNamespace(action=action,feasible=True,emergency_verified_safe=False,
                         reason='verified_reference',minimum_predicted_clearance_m=2.,
                         planning_ms=1.,verified_count=9)
    observer=FailureObserver(cfg)
    observer(plant,[track],1.,plan,scene)
    states=np.array([[0.,*plant.eta,*plant.nu,*plant.actual],
                     [1.,*plant.eta,*plant.nu,*plant.actual]])
    snapshots=[{'time_s':stamp,'ids':snapshot.ids.tolist(),'positions':snapshot.positions.tolist(),
                'radii':snapshot.radii.tolist()} for stamp,snapshot in ((0.,initial),(1.,later))]
    summary=analyze_obstacle_forecasts({'states':states,'summary':{'scenario':'crossing3d'},
                                      'obstacles':snapshots},observer,tmp_path)
    assert summary['sample_count']==121
    assert summary['at_2s']['rmse_m']<1e-8
    assert summary['at_24s']['rmse_m']<1e-8
    assert set(summary['by_object_id'])=={'0'}
    assert summary['vehicle_prediction_error_included'] is False
    assert summary['future_truth_used_by_planner'] is False
    assert summary['verified_recorded_truth_snapshots']==2
    assert (tmp_path/'obstacle_forecast_errors.csv').is_file()
    assert (tmp_path/'obstacle_forecast_error.png').stat().st_size>1000


def test_actual_margin_violation_is_reported_after_feasible_plan_without_emergency(tmp_path):
    cfg=replace(Stage2Config(),warmup_s=0.)
    states=np.zeros((4,16))
    states[:,0]=[0.,.02,.04,.06]
    planning=np.array([[0.,1.,45.,9.,9.,2.,1.,1.,0.]])
    result={'states':states,'planning':planning,
            'clearance':np.array([[.02,1.2,.2],[.04,.8,-.2],[.06,1.,0.]]),
            'planning_diagnostics':[{'time_s':0.,'feasible':True,'emergency_verified_safe':False}]}
    summary=analyze_actual_margin(result,cfg,tmp_path)
    assert summary['actual_margin_met'] is False
    assert summary['violating_physics_interval_count']==1
    assert summary['violating_intervals_after_feasible_plan']==1
    assert summary['violating_intervals_after_emergency_plan']==0
    assert summary['collision_physics_interval_count']==0
    assert summary['contiguous_interval_groups'][0]['first_interval_start_s']==.02
    assert np.isclose(summary['violating_interval_total_duration_s'],.02)
    saved=np.loadtxt(tmp_path/'actual_margin_diagnostics'/'violating_intervals.csv',delimiter=',',skiprows=1,ndmin=2)
    assert saved[0,6]==0.
