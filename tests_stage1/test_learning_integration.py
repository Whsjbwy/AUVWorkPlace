"""Physical/model compatibility and exact CPU training-resume tests."""
from pathlib import Path
import json
import numpy as np
import pytest
from auv3d.config import ExperimentConfig
from auv3d.planner import DWA3D
from auv3d.prediction import CAPredictor,SphericalCalibration,pplnm_contract


def calibrated():
    obj=SphericalCalibration()
    obj.fit(np.zeros((40,3,3)),np.ones((40,3,3)),np.ones(40))
    obj.metadata={'predictor_name':CAPredictor.name,'prediction_dt_s':3.}
    return obj


def test_correct_calibration_identity_accepted():
    DWA3D(ExperimentConfig(),CAPredictor(3.),calibrated())


def test_calibration_wrong_predictor_rejected():
    c=calibrated();c.metadata['predictor_name']='some_other_predictor'
    with pytest.raises(ValueError,match='identity'):DWA3D(ExperimentConfig(),CAPredictor(3.),c)


def test_calibration_wrong_time_step_rejected():
    c=calibrated();c.metadata['prediction_dt_s']=1.
    with pytest.raises(ValueError,match='time step'):DWA3D(ExperimentConfig(),CAPredictor(3.),c)


@pytest.mark.parametrize('field,value',[
    ('architecture','different_model'),('input_dim',10),('output_dim',2),
    ('history_length',5),('horizon_steps',4),('epochs_completed',12),
    ('training_label','another_training_run'),
])
def test_calibration_wrong_model_parameter_rejected(field,value):
    p=CAPredictor(3.);p.name='pplnm_reimplementation'
    p.checkpoint_meta=pplnm_contract(3.,150,'synthetic_training')
    c=calibrated();c.metadata=p.checkpoint_meta.copy();c.metadata[field]=value
    with pytest.raises(ValueError,match=field):DWA3D(ExperimentConfig(),p,c)


def test_calibration_wrong_horizon_dimension_rejected():
    c=calibrated();c.global_q=c.global_q[:2]
    with pytest.raises(ValueError,match='horizon'):DWA3D(ExperimentConfig(),CAPredictor(3.),c)


def test_predictor_wrong_time_step_without_calibration_rejected():
    with pytest.raises(ValueError,match='time step'):DWA3D(ExperimentConfig(),CAPredictor(1.))


def test_calibration_legacy_without_metadata_rejected():
    c=calibrated();c.metadata={}
    with pytest.raises(ValueError):DWA3D(ExperimentConfig(),CAPredictor(3.),c)


def test_training_resume_matches_uninterrupted(tmp_path):
    torch=pytest.importorskip('torch')
    from auv3d.data import generate_dataset
    from auv3d.train import train_predictor
    path=tmp_path/'data.npz'
    generate_dataset(path,n_trajectories=30,seconds=50,dt=1.,seed=43)
    full=tmp_path/'full';chunk=tmp_path/'chunk'
    train_predictor(path,full,epochs=4,threads=2)
    report=train_predictor(path,chunk,epochs=4,threads=2,chunk_epochs=2)
    assert report['epochs_completed']==2
    assert not (chunk/'training_report.json').exists()  # No early test evaluation.
    train_predictor(path,chunk,epochs=4,threads=2,chunk_epochs=2,resume=True)
    a=torch.load(full/'pplnm.pt',map_location='cpu',weights_only=True)
    b=torch.load(chunk/'pplnm.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    assert json.loads((chunk/'training_progress.json').read_text())['status']=='complete'
    with pytest.raises(ValueError,match='same dataset'):
        train_predictor(path,chunk,epochs=5,resume=True)
    with pytest.raises(ValueError,match='same dataset'):
        train_predictor(path,chunk,epochs=4,resume=True,batch_size=128)
    changed=tmp_path/'changed.npz'
    generate_dataset(changed,n_trajectories=30,seconds=50,dt=1.,seed=44)
    with pytest.raises(ValueError,match='same dataset'):
        train_predictor(changed,chunk,epochs=4,resume=True)


@pytest.mark.parametrize('directory',['pplnm_150','pplnm_smoke'])
def test_provided_weights_load_predict_and_match_calibration(directory):
    pytest.importorskip('torch')
    from auv3d.prediction import PPLNMPredictor
    from auv3d.sensors import Track
    root=Path(__file__).resolve().parents[1]/'artifacts_stage1'/directory
    predictor=PPLNMPredictor(root/'pplnm.pt')
    calibration=SphericalCalibration.load(root/'pplnm_calibration.json')
    DWA3D(ExperimentConfig(),predictor,calibration)
    t=np.arange(6)*predictor.dt
    positions=np.column_stack([50.+t,np.full(6,50.),np.full(6,30.)])
    track=Track(0,2.,t.tolist(),list(positions))
    result=predictor.predict(track,t[-1]+np.arange(4)*predictor.dt)
    assert result.shape==(4,3) and np.isfinite(result).all()
    np.testing.assert_allclose(result[0],positions[-1])
    with pytest.raises(ValueError,match='horizon'):
        predictor.predict(track,[t[-1]+predictor.max_horizon_s+1.])


def test_learned_stale_track_uses_explicit_fallback():
    from auv3d.sensors import Track
    from auv3d.terrain import HeightField
    class LimitedPredictor(CAPredictor):
        max_horizon_s=9.
        def predict(self,track,absolute_times):
            raise AssertionError('stale learned predictor must not be called')
    planner=DWA3D(ExperimentConfig(),LimitedPredictor(3.))
    track=Track(0,2.,[0.],[np.array([100.,100.,30.])])
    result=planner.plan(np.array([50.,50.,30.,1.,0.,0.]),np.array([150.,50.,30.]),HeightField.flat(),[track],1.)
    assert result.prediction_fallback_count==1


@pytest.mark.parametrize('field,value',[
    ('dt',float('nan')),('xstd',[0.]*9),('yscale',[1.,1.]),
])
def test_invalid_checkpoint_parameters_rejected(tmp_path,field,value):
    torch=pytest.importorskip('torch')
    from auv3d.prediction import PPLNMPredictor
    source=Path(__file__).resolve().parents[1]/'artifacts_stage1'/'pplnm_150'/'pplnm.pt'
    checkpoint=torch.load(source,map_location='cpu',weights_only=True)
    checkpoint[field]=value
    path=tmp_path/'invalid.pt';torch.save(checkpoint,path)
    with pytest.raises(ValueError):PPLNMPredictor(path)


def test_new_model_bad_thread_count():
    pytest.importorskip('torch')
    from auv3d.prediction import PPLNMPredictor
    with pytest.raises(ValueError,match='threads'):
        PPLNMPredictor('unused_path',threads=0)


def test_emergency_action_respects_angular_acceleration():
    from auv3d.terrain import HeightField
    cfg=ExperimentConfig();planner=DWA3D(cfg)
    planner.previous=np.array([1.,np.deg2rad(4.),np.deg2rad(5.)])
    # Invalid shallow state ensures no feasible candidate; an emergency still
    # cannot request a rate change exceeding the configured dynamic window.
    old=planner.previous.copy()
    state=np.array([50.,50.,.1,1.,0.,0.])
    result=planner.plan(state,np.array([150.,50.,30.]),HeightField.flat(),[],0.)
    assert not result.feasible
    assert np.all(abs(result.command[1:]-old[1:])<=np.deg2rad(4)+1e-12)


@pytest.mark.parametrize('field,value', [('goal_weight',float('nan')),('speed_samples',2.5),('safety_margin_m',-1)])
def test_bad_planner_config_rejected(field,value):
    from dataclasses import replace
    cfg=ExperimentConfig()
    with pytest.raises(ValueError):replace(cfg,planner=replace(cfg.planner,**{field:value})).validate()
