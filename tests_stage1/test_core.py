import json
from dataclasses import replace
import numpy as np
import pytest
from auv3d.config import ExperimentConfig,TerrainConfig,ObstacleConfig,SonarConfig
from auv3d.geometry import (wrap_pi,rotation_zyx,kinematic_derivative,kinematic_step,
                           segment_point_distance,swept_sphere_clearance)
from auv3d.terrain import HeightField
from auv3d.obstacles import LinearObstacles,PIObstacles,ObstacleSnapshot
from auv3d.sensors import Observation,Track,TrackMemory,IdealObjectSonar
from auv3d.prediction import make_history,constant_acceleration,CAPredictor,SphericalCalibration
from auv3d.planner import candidate_commands,rollout,DWA3D
from auv3d.remus import (verify_sources,parameter_audit,RemusPlant,regression,
                         load_model,validate_vehicle_interface,source_root,REQUIRED_SOURCES)

@pytest.mark.parametrize('angle',[-20,-np.pi,0,np.pi,20])
def test_wrap(angle):assert -np.pi<=wrap_pi(angle)<np.pi

@pytest.mark.parametrize('angles',[(0,0,0),(.2,.3,1.),(-.3,-.5,2.3)])
def test_rotation(angles):
    R=rotation_zyx(*angles);np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-12);assert np.linalg.det(R)==pytest.approx(1.)

@pytest.mark.parametrize('pitch',[0.,.2,-.4])
def test_p25_equation2(pitch):
    state=np.array([10,20,30,2,pitch,.5]);cmd=np.array([2.,.03,.04]);d=kinematic_derivative(state,cmd)
    np.testing.assert_allclose(d[:3],[2*np.cos(.5)*np.cos(pitch),2*np.sin(.5)*np.cos(pitch),-2*np.sin(pitch)])
    assert d[4]==.03;assert d[5]==pytest.approx(.04/np.cos(pitch))

def test_ned_pitch_up_is_smaller_depth():assert kinematic_step([0,0,50,2,.2,0],[2,0,0],1)[2]<50

def test_current_added_once_in_kinematic_extension():
    s=kinematic_step([0,0,30,1,0,0],[1,0,0],1,(0,.5,0));np.testing.assert_allclose(s[:3],[1,.5,30])

def test_kinematic_input_not_mutated():
    s=np.array([1,2,3,1,.2,.3]);copy=s.copy();kinematic_step(s,[1,.1,.1],.1);np.testing.assert_equal(s,copy)

def test_kinematic_vectorization():
    states=np.tile([10.,20,30,1,0,0],(5,1));out=kinematic_step(states,np.tile([1,0,0],(5,1)),.5)
    assert out.shape==(5,6);np.testing.assert_allclose(out[:,0],10.5)

def test_pitch_singularity():
    with pytest.raises(ValueError):kinematic_derivative(np.array([0,0,30,1,np.pi/2,0]),[1,0,0])

@pytest.mark.parametrize('dt',[0.,-.1,float('nan')])
def test_invalid_dt(dt):
    with pytest.raises(ValueError):kinematic_step([0,0,30,1,0,0],[1,0,0],dt)

def test_segment_distance():assert segment_point_distance([-2,0,0],[2,0,0],[0,3,0])==pytest.approx(3)

def test_swept_dynamic_crossing_detected():
    assert swept_sphere_clearance([-5,0,0],[5,0,0],[0,-5,0],[0,5,0],1)==pytest.approx(-1)

def test_swept_different_times_do_not_collide():
    assert swept_sphere_clearance([-5,0,0],[5,0,0],[0,5,0],[0,15,0],1)>0

def test_swept_static_tunneling():assert swept_sphere_clearance([-20,0,0],[20,0,0],[0,0,0],[0,0,0],2)==pytest.approx(-2)

def test_zero_length_segment():assert segment_point_distance([0,0,0],[0,0,0],[1,2,2])==pytest.approx(3)

def test_default_config_valid():ExperimentConfig().validate()

@pytest.mark.parametrize('field,value',[('min_speed',3.),('pitch_max_deg',90.),('acceleration_max',0.),('depth_limit_m',-1.)])
def test_invalid_vehicle_config(field,value):
    c=ExperimentConfig()
    with pytest.raises(ValueError):replace(c,vehicle=replace(c.vehicle,**{field:value})).validate()

def test_incompatible_time_grid():
    c=ExperimentConfig()
    with pytest.raises(ValueError):replace(c,planner=replace(c.planner,rollout_dt=.3)).validate()

def test_forecast_time_mismatch():
    c=ExperimentConfig()
    with pytest.raises(ValueError):replace(c,prediction_dt_s=1.).validate()

def test_config_roundtrip(tmp_path):
    c=ExperimentConfig();p=tmp_path/'c.json';p.write_text(json.dumps(c.to_dict()));read=ExperimentConfig.load(p)
    np.testing.assert_equal(read.current_ned_mps,c.current_ned_mps);assert read.planner==c.planner

def test_perlin_seed_depth_and_shape():
    c=TerrainConfig(grid_spacing=10);a=HeightField.perlin(c,1);b=HeightField.perlin(c,1);other=HeightField.perlin(c,2)
    assert a.depths.shape==(101,101);assert a.depths.min()==pytest.approx(50);assert a.depths.max()==pytest.approx(200)
    np.testing.assert_array_equal(a.depths,b.depths);assert not np.allclose(a.depths,other.depths)

def test_bilinear_exact_plane():
    h=HeightField(np.array([0,10.]),np.array([0,10.]),np.array([[100.,110],[120,130]]),np.sqrt(5))
    assert h.depth_at([3,4])==pytest.approx(111.)

def test_terrain_boundary_surface_and_bottom():
    h=HeightField.flat()
    assert h.clearance([10,10,30],1)>0
    assert h.clearance([10,10,0],1)<0
    assert h.clearance([10,10,201],1)<0
    assert h.clearance([-1,10,30],1)<0

def test_terrain_segment_tunneling():
    h=HeightField.flat();assert h.segment_clearance([10,10,30],[10,10,210],1)<0

def test_obstacle_table_values_not_vehicle_values():
    c=ExperimentConfig();assert c.obstacles.r_max_deg_s==15;assert c.vehicle.r_max_deg_s==5
    assert c.obstacles.kp==(0.6,1.,.8);assert c.obstacles.count==120

def test_pi_obstacle_placement_and_bounds():
    c=replace(ObstacleConfig(),count=10);h=HeightField.flat();o=PIObstacles(c,h,11);p=o.snapshot().positions
    dist=np.linalg.norm(p[:,None]-p[None],axis=-1);dist+=np.eye(len(p))*1e6
    assert dist.min()>=25;assert np.all(h.clearance(p,o.radii)>0)
    for _ in range(100):o.step(.1)
    assert np.all((o.states[:,3]>=.2)&(o.states[:,3]<=2.5));assert np.isfinite(o.states).all()

def test_obstacle_reproducible():
    c=replace(ObstacleConfig(),count=2);a=PIObstacles(c,HeightField.flat(),9);b=PIObstacles(c,HeightField.flat(),9)
    for _ in range(20):a.step(.1);b.step(.1)
    np.testing.assert_array_equal(a.states,b.states)

def test_empty_obstacles():
    a=LinearObstacles([],[],[]);a.step(.1);assert a.snapshot().positions.shape==(0,3)

@pytest.mark.parametrize('pos,expected',[( [50,0,0],True),([-50,0,0],False),([20,50,0],False),([50,0,20],False),([200,0,0],False)])
def test_body_sonar_fov(pos,expected):
    eta=np.array([100,100,50,0,0,0]);s=IdealObjectSonar(replace(SonarConfig(),terrain_occlusion=False),0)
    snap=ObstacleSnapshot(np.array([0]),np.array([eta[:3]+pos]),np.array([1.]))
    assert bool(s.observe(eta,0,snap,HeightField.flat()))==expected

def test_sonar_turns_with_heading():
    eta=np.array([100,100,50,0,0,np.pi/2]);s=IdealObjectSonar(replace(SonarConfig(),terrain_occlusion=False),0)
    snap=ObstacleSnapshot(np.array([0]),np.array([[100,150,50]]),np.array([1.]))
    assert len(s.observe(eta,0,snap,HeightField.flat()))==1

def test_sonar_sphere_extent_not_only_center():
    eta=np.array([100,100,50,0,0,0]);s=IdealObjectSonar(replace(SonarConfig(),terrain_occlusion=False),0)
    snap=ObstacleSnapshot(np.array([0]),np.array([[120,100,53]]),np.array([4.]))
    assert len(s.observe(eta,0,snap,HeightField.flat()))==1

def test_observations_copied_not_true_future_reference():
    o=LinearObstacles([[150,100,50]],[[1,0,0]],[2]);s=IdealObjectSonar(SonarConfig(),0)
    obs=s.observe([100,100,50,0,0,0],0,o.snapshot(),HeightField.flat());old=obs[0].position.copy();o.step(10)
    np.testing.assert_array_equal(obs[0].position,old)

def test_track_future_rejected():
    m=TrackMemory()
    with pytest.raises(ValueError):m.update([Observation(0,2.,np.zeros(3),1.)],1.)

def test_track_retention_then_expiry():
    m=TrackMemory(3);m.update([Observation(0,0.,np.zeros(3),1.)],0)
    m.update([],2);assert len(m.current(2))==1;m.update([],4);assert len(m.current(4))==0

def test_duplicate_observation_time_rejected():
    m=TrackMemory();o=Observation(0,0,np.zeros(3),1);m.update([o],0)
    with pytest.raises(ValueError):m.update([o],0)

def test_cold_start_one_sample_no_hidden_velocity():
    h=make_history([0],[[3,4,5]],3);assert h.real_mask.sum()==1
    np.testing.assert_equal(h.features[:,3:],0);np.testing.assert_equal(h.features[:,:3],np.tile([3,4,5],(6,1)))

def test_equation11_backward_linear():
    h=make_history([0,1],[[0,0,0],[2,0,0]],1)
    np.testing.assert_allclose(h.features[:,0],[-8,-6,-4,-2,0,2]);assert h.real_mask.sum()==2

def test_ca_constant_velocity():
    t=np.arange(6);p=np.column_stack([t,2*t,3*t]);h=make_history(t,p)
    forecast=constant_acceleration(h.features,[1,2,3]);np.testing.assert_allclose(forecast,[[6,12,18],[7,14,21],[8,16,24]])

def test_feature_shape_and_no_future_input():
    h=make_history([0,1,2],[[0,0,0],[1,1,1],[2,2,2]])
    assert h.features.shape==(6,9);assert h.latest_time_s==2

def test_negative_forecast_rejected():
    with pytest.raises(ValueError):constant_acceleration(np.zeros((6,9)),[-1])

def test_calibration_and_save_reload(tmp_path):
    rng=np.random.default_rng(1);truth=rng.normal(size=(200,3,3));pred=truth+.3*rng.normal(size=truth.shape)
    speeds=rng.uniform(.2,2.5,200);c=SphericalCalibration().fit(pred,truth,speeds)
    assert np.all(c.radius(1.,[1,2,3])>=0)
    p=tmp_path/'cal.json';c.save(p);d=SphericalCalibration.load(p);np.testing.assert_equal(c.quantiles,d.quantiles)
    with pytest.raises(ValueError):c.radius(1.,[4.])

def test_calibration_small_bin_falls_back():
    c=SphericalCalibration().fit(np.ones((100,3,3)),np.zeros((100,3,3)),np.ones(100))
    np.testing.assert_equal(c.quantiles[0],c.global_q);assert c.counts[0]==0

def test_calibration_insufficient_data_rejected():
    with pytest.raises(ValueError):SphericalCalibration().fit(np.ones((3,3,3)),np.zeros((3,3,3)),np.ones(3))

def test_dwa_candidates_respect_limits():
    c=ExperimentConfig();s=np.array([50,50,30,1,0,0]);a=candidate_commands(s,[1,0,0],c)
    assert np.all(abs(a[:,0]-1)<=c.vehicle.acceleration_max+1e-9)
    assert np.max(abs(a[:,1]))<=np.deg2rad(c.vehicle.q_max_deg_s)+1e-9
    assert np.max(abs(a[:,2]))<=np.deg2rad(c.vehicle.r_max_deg_s)+1e-9

def test_rollout_shape_and_three_dimensional_motion():
    c=ExperimentConfig();s=np.array([50,50,30,1,0,0]);a=np.array([[1,.02,.02]])
    p=rollout(s,a,c);assert p.shape==(1,46,6);assert p[0,-1,0]>50;assert p[0,-1,1]>50;assert p[0,-1,2]<30

def test_dwa_no_obstacle_progress():
    c=ExperimentConfig();s=np.array([50,50,30,1,0,0]);r=DWA3D(c).plan(s,np.array([150,80,45]),HeightField.flat(),[],0)
    assert r.feasible;assert r.command[0]>0;assert r.valid_count<=r.candidate_count

def test_dwa_failure_is_not_silent_success():
    c=ExperimentConfig();t=Track(0,10);t.append(Observation(0,0,np.array([51,50,30]),10))
    r=DWA3D(c).plan(np.array([50,50,30,1,0,0]),np.array([150,50,30]),HeightField.flat(),[t],0)
    assert not r.feasible;assert r.command[0]>=.7-1e-9

def test_repository_sources_available():
    report=verify_sources()
    assert all(v['exists'] for v in report['files'].values())
    assert report['selected_root']==str(source_root().resolve())

def test_local_comment_edit_is_accepted(tmp_path):
    import shutil
    for rel in REQUIRED_SOURCES:
        p=tmp_path/rel;p.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_root()/rel,p)
    with (tmp_path/'vehicles/remus100.py').open('a',encoding='utf-8') as f:
        f.write('\n# Ordinary local documentation edit.\n')
    assert verify_sources(tmp_path)['files']['vehicles/remus100.py']['exists']

def test_missing_source_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError,match='remus100.py'):
        verify_sources(tmp_path)

def test_remus_uses_normal_existing_repository_import():
    import importlib
    cls,gnc=load_model()
    vehicle_module=importlib.import_module('python_vehicle_simulator.vehicles.remus100')
    assert cls is vehicle_module.remus100
    assert gnc is importlib.import_module('python_vehicle_simulator.lib.gnc')
    from pathlib import Path
    assert Path(vehicle_module.__file__).resolve()==source_root().resolve()/'vehicles/remus100.py'

def test_conflicting_installed_module_path_rejected(monkeypatch,tmp_path):
    _,gnc=load_model()
    monkeypatch.setattr(gnc,'__file__',str(tmp_path/'gnc.py'))
    with pytest.raises(RuntimeError,match='expected repository source'):
        load_model()

@pytest.mark.parametrize('name,value',[
    ('nu',np.zeros(5)),('u_actual',np.zeros(2)),('M',-np.eye(6)),
    ('T_delta',0.),('T_n',float('nan')),('dimU',2),('dynamics',None),
])
def test_remus_runtime_interface_rejects_invalid_contract(name,value):
    cls,_=load_model();vehicle=cls();setattr(vehicle,name,value)
    with pytest.raises((ValueError,TypeError)):
        validate_vehicle_interface(vehicle)

def test_remus_mass_and_matrix():
    a=parameter_audit();assert a['remus']['mass_computed_from_code_kg']==pytest.approx(31.0293849758)
    assert a['remus']['positive_definite_mass_matrix'];assert a['unchanged']

@pytest.mark.parametrize('depth',[-1,101,200])
def test_remus_depth_guard_preserved(depth):
    with pytest.raises(ValueError):RemusPlant(depth=depth)

@pytest.mark.parametrize('rpm',[-1,1526,3000])
def test_remus_rpm_guard_preserved(rpm):
    with pytest.raises(ValueError):RemusPlant(rpm=rpm)

def test_remus_direct_command_guard():
    r=RemusPlant()
    with pytest.raises(ValueError):r.step(command=[1,0,1000])

def test_remus_current_not_added_twice():
    r=RemusPlant(eta0=[0,0,30,0,0,0]);r.nu=np.array([1,0,0,0,0,0.])
    np.testing.assert_allclose(r.ground_velocity(),[1,0,0]);assert not np.allclose(r.relative_velocity(),r.nu[:3])

def test_remus_ned_pitch_up_reduces_depth():
    r=RemusPlant(eta0=[0,0,30,0,.2,0]);r.nu=np.array([1,0,0,0,0,0.])
    assert r.ground_velocity()[2]<0
    np.testing.assert_allclose(r.gnc.Rzyx(*r.eta[3:]),rotation_zyx(*r.eta[3:]))

def test_remus_body_rates_need_full_euler_transform_at_nonzero_roll():
    _,gnc=load_model();roll,pitch=.2,.3;q,r=.03,.04
    eta=np.array([0,0,30,roll,pitch,0.]);nu=np.array([0,0,0,0,q,r])
    after=gnc.attitudeEuler(eta.copy(),nu,.01)
    angular=(after[3:]-eta[3:])/.01
    assert angular[1]==pytest.approx(q*np.cos(roll)-r*np.sin(roll))
    assert angular[2]==pytest.approx((q*np.sin(roll)+r*np.cos(roll))/np.cos(pitch))
    assert not np.isclose(angular[1],q)

@pytest.mark.parametrize('dt',[0.,-.1,.03,float('nan')])
def test_remus_invalid_timestep_rejected(dt):
    with pytest.raises(ValueError):RemusPlant().step(dt)

@pytest.mark.parametrize('command',[[0,0],[float('nan'),0,1000],[0,float('inf'),1000]])
def test_remus_invalid_actuator_shape_or_number_rejected(command):
    with pytest.raises(ValueError):RemusPlant().step(command=command)

def test_remus_200s_original_regression():
    result,data=regression();assert result['matches_readme_rounding'];assert result['n_records']==10001;assert np.isfinite(data).all()
