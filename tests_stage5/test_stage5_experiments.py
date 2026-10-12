"""Stage 5 outcomes, independent freezing and exact offline score semantics."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from auv3d.stage2_config import Stage2Config
from auv3d.stage5_protocol import manifest, config_for_case
from auv3d import stage5_experiments as exp
from auv3d import stage5_analysis as analysis


def case(split='test',noise=.02):
    return manifest('pilot',split=split,noise_std=noise)[0]


def record(status='success',safe=True):
    return dict(program_error=False,planning_cycle_count=4,
        verified_candidate_count=36,feasible_candidate_count=18,
        no_feasible_candidate_count=1,summary=dict(status=status,reached_goal=status=='success',
        actual_safety_margin_met=safe,unsafe_emergency_forecast_count=1,
        planning_exceeds_control_period_count=2,duration_s=200.,all_states_finite=True))


def test_physical_success_is_separate_from_forecast_and_missing_labels():
    job = exp.ExperimentJob(case())
    row = exp.episode_metrics(job,record(safe=False),cfg=config_for_case(job.case))
    assert row['navigation_success'] and row['actual_clearance_margin_violation']
    assert row['safe_arrival'] is False
    assert row['unsafe_emergency_forecast']
    assert row['feasible_fraction']==.5 and row['planning_deadline_miss_fraction']==.5
    assert row['no_feasible_candidate_fraction']==.25
    assert row['calibration_coverage'] is None and row['candidate_false_reject_count'] is None


def test_program_error_does_not_impute_collision_or_success():
    job = exp.ExperimentJob(case())
    row = exp.episode_metrics(job,dict(program_error=True),cfg=config_for_case(job.case))
    assert row['status']=='program_error'
    assert row['navigation_success'] is None and row['collision'] is None
    assert row['safe_arrival'] is None


@pytest.mark.parametrize('bad',['../case','a/b','a\\b','.','..',''])
def test_output_path_escape_rejected(bad):
    item = dict(case(),case_id=bad)
    with pytest.raises(ValueError,match='case_id'):
        exp.select_cases([item])


def test_noise_variants_are_not_independent_geometry_counts():
    rows = manifest('pilot',split='test')
    assert len(rows)==36 and len({row['scene_id'] for row in rows})==12
    assert len(exp.select_cases(rows,noise_std=.02))==12


@pytest.mark.parametrize('kwargs',[dict(method='A6'),dict(predictor='pplnm'),
    dict(method='A2'),dict(method='A0',alpha=.1),dict(method='A1',fixed_margin_m=1.),
    dict(planning_delay_mode='measured',planning_delay_s=.5)])
def test_method_artifact_and_delay_validation(kwargs):
    with pytest.raises(ValueError):
        exp.ExperimentJob(case(),**kwargs).validate()


def test_registry_never_falls_back_to_another_predictor_or_noise(tmp_path):
    path = tmp_path/'ca.npz';path.write_bytes(b'placeholder')
    mapping = {('ca',.02):str(path)}
    assert exp.artifact_path(mapping,'ca',.02)==str(path)
    with pytest.raises(FileNotFoundError):exp.artifact_path(mapping,'cv',.02)
    with pytest.raises(FileNotFoundError):exp.artifact_path(mapping,'ca',0.)


def test_registry_relative_paths_and_duplicate_keys(tmp_path):
    registry = tmp_path/'registry.json'
    entry = dict(predictor='ca',noise_std_m=.02,path='pool.npz')
    registry.write_text(json.dumps({'calibration_libraries':[entry]}),encoding='utf8')
    assert exp.load_artifact_registry(registry)[('ca',.02)]==str(tmp_path/'pool.npz')
    registry.write_text(json.dumps({'calibration_libraries':[entry,entry]}),encoding='utf8')
    with pytest.raises(ValueError,match='duplicate'):exp.load_artifact_registry(registry)


def test_grouping_keeps_risk_predictor_noise_and_delay_factors_separate():
    jobs = [exp.ExperimentJob(case()),exp.ExperimentJob(case(),predictor='cv'),
        exp.ExperimentJob(case(noise=0.)),exp.ExperimentJob(case(),planning_delay_mode='fixed',planning_delay_s=.4)]
    rows = [exp.episode_metrics(job,record('timeout'),cfg=config_for_case(job.case)) for job in jobs]
    groups = exp.grouped_results(rows)
    assert len(groups)==4 and all(row['timeout_rate']==1 for row in groups)


def _tune_evidence(root,values):
    root.mkdir()
    rows,jobs=[],[]
    for margin,status,safe in values:
        for item in manifest('pilot',split='tune',noise_std=.02):
            job = exp.ExperimentJob(item,method='A1',fixed_margin_m=margin)
            jobs.append(dict(job.__dict__,job_id=job.job_id))
            rows.append(exp.episode_metrics(job,record(status,safe),cfg=config_for_case(job.case)))
    (root/'EXPERIMENT_CONFIG.json').write_text(json.dumps(dict(jobs=jobs)),encoding='utf8')
    (root/'EXPERIMENT_METRICS.json').write_text(json.dumps(rows),encoding='utf8')


def test_freezing_uses_safe_arrival_and_preserves_all_tune_points(tmp_path):
    root = tmp_path/'tune'
    _tune_evidence(root,[(0.,'success',False),(.75,'timeout',True),(3.,'success',True)])
    out = tmp_path/'frozen.json'
    result=analysis.freeze_fixed_margin([root],out)
    assert result['test_data_used'] is False and result['source_split']=='tune'
    assert result['selections'][0]['fixed_margin_m']==3.
    assert len(result['selections'][0]['candidates'])==3
    assert exp.frozen_margin(out,'ca',.02,'add',phase='pilot')==3.
    with pytest.raises(ValueError):exp.frozen_margin(out,'cv',.02,'add',phase='pilot')
    with pytest.raises(ValueError,match='tune-only'):
        exp.frozen_margin(out,'ca',.02,'add',phase='formal')
    with pytest.raises(FileExistsError):analysis.freeze_fixed_margin([root],out)


def test_freezing_rejects_test_registration_before_reading_any_metrics(tmp_path):
    job = exp.ExperimentJob(case(),method='A1',fixed_margin_m=3.,tuning_path='placeholder')
    (tmp_path/'EXPERIMENT_CONFIG.json').write_text(json.dumps(dict(jobs=[job.__dict__])),encoding='utf8')
    with pytest.raises(ValueError,match='independent A1 tune'):
        analysis.freeze_fixed_margin([tmp_path],tmp_path/'frozen.json')
    assert not (tmp_path/'frozen.json').exists()


def test_freezing_rejects_unpaired_tuning_margins(tmp_path):
    root=tmp_path/'tune';_tune_evidence(root,[(0.,'timeout',True),(3.,'success',True)])
    path=root/'EXPERIMENT_METRICS.json'
    rows=json.loads(path.read_text(encoding='utf8'));rows[1]['case_id']='another'
    path.write_text(json.dumps(rows),encoding='utf8')
    with pytest.raises(ValueError,match='independent tune scenes'):
        analysis.freeze_fixed_margin([root],tmp_path/'frozen.json')


def test_primary_artifact_pool_must_match_actual_phase(tmp_path):
    path=tmp_path/'pool.npz'
    metadata=dict(schema='stage5-risk-v1',split='calibration',role='calibration_only',
        predictor='ca',noise_std_m=.02,phase='pilot')
    np.savez(path,metadata_json=np.asarray(json.dumps(metadata)))
    paths={('ca',.02):str(path)}
    assert exp.artifact_path(paths,'ca',.02,phase='pilot')==str(path)
    with pytest.raises(ValueError,match='phase'):
        exp.artifact_path(paths,'ca',.02,phase='formal')


def test_delay_summary_maps_to_actual_physical_latency_columns():
    job=exp.ExperimentJob(case(),planning_delay_mode='measured')
    data=record();data['summary'].update(planning_computation_delay_mean_s=.6,
                                       planning_computation_delay_max_s=1.2)
    row=exp.episode_metrics(job,data,cfg=config_for_case(job.case))
    assert row['physical_planning_delay_mean_s']==.6
    assert row['physical_planning_delay_max_s']==1.2


def test_fixed_margin_scanning_cannot_use_independent_test_data(tmp_path):
    with pytest.raises(ValueError,match='restricted'):
        exp.run_study(tmp_path/'out',cases=[case()],methods=('A1',),fixed_margins=(0.,3.))
    assert not (tmp_path/'out').exists()


def test_existing_failure_directory_is_never_overwritten(tmp_path):
    path=tmp_path/'failed.log';path.write_text('kept',encoding='utf8')
    with pytest.raises(FileExistsError):exp.run_study(tmp_path,cases=[case()])
    assert path.read_text(encoding='utf8')=='kept'


def _score_fixture():
    from auv3d.sensors import Track
    cfg=replace(Stage2Config(),horizon_s=2.)
    track=Track(1,1.,[-1.,0.],[np.array([10.,0.,30.]),np.array([10.,0.,30.])])
    frame=SimpleNamespace(cfg=cfg,tracks=[track],now=0.,memory_uncertainty={})
    states=np.zeros((101,16));states[:,0]=np.arange(101)*.02;states[:,3]=30.
    forecast=SimpleNamespace(forecast=lambda track,times,now,extra:SimpleNamespace(positions=np.tile([10.,0.,30.],(len(times),1))))
    truth=SimpleNamespace(ids=np.array([1]),_positions=[np.array([[8.,0.,30.]]) for _ in range(101)],_ensure=lambda _:None)
    risk=dict(offsets_s=[0.,1.,2.],speed_bin_edges_mps=[0.,.25,'inf'],
        q_by_speed_bin_m=[[2.,2.,2.],[50.,50.,50.]],quantile_m=2.)
    return frame,SimpleNamespace(states=states),forecast,truth,risk


def test_per_time_ball_coverage_does_not_turn_missing_speed_bin_into_zero_label():
    args=_score_fixture()
    result=analysis._score_coverage(*args,'A2')
    assert result['stage5_score_covered']
    assert result['score_by_speed_bin_m'][1] is None
    assert all(v['label_count']==1 for v in result['coverage_by_horizon_s'].values())
    args[-1]['q_by_speed_bin_m'][0]=[1.,1.,1.]
    assert not analysis._score_coverage(*args,'A2')['stage5_score_covered']


@pytest.mark.parametrize('method',['A3','A4','A5'])
def test_joint_and_interval_scores_use_exact_method_definitions(method):
    args=_score_fixture()
    if method=='A5':args[-1]['q_by_speed_bin_m']=[[2.,2.],[50.,50.]]
    assert analysis._score_coverage(*args,method)['stage5_score_covered']
    args[-1]['quantile_m']=1.
    args[-1]['q_by_speed_bin_m'][0]=[1.]*(2 if method=='A5' else 3)
    assert not analysis._score_coverage(*args,method)['stage5_score_covered']


def test_cli_manifest_and_test_tune_override(capsys,tmp_path):
    import run_stage5
    assert run_stage5.main(['manifest','--phase','pilot'])==0
    assert 'stage5_risk_validation_v1' in capsys.readouterr().out
    with pytest.raises(ValueError,match='restricted'):
        run_stage5.main(['tune-margin','--split','test','--margins','0','3','--out',str(tmp_path/'out')])


def _raw_source(root):
    job=exp.ExperimentJob(case())
    directory=root/'cases'/job.job_id
    directory.mkdir(parents=True)
    summary=dict(backend='remus6dof',risk_method='A0',obstacle_predictor='ca',
        seed=job.case['seed'],scenario=job.case['scenario'],planning_delay_mode='none',status='success')
    for name in ('states.csv','controls.csv','selected_plans.json','planning_diagnostics.json',
                 'observations.json','obstacles.json'):
        (directory/name).write_text('[]',encoding='utf8')
    (directory/'summary.json').write_text(json.dumps(summary),encoding='utf8')
    (directory/'config.json').write_text(json.dumps(config_for_case(job.case).to_dict()),encoding='utf8')
    (directory/'stage5_experiment.json').write_text(json.dumps(dict(job=job.__dict__,actual_predictor='ca')),encoding='utf8')
    (root/'EXPERIMENT_CONFIG.json').write_text(json.dumps(dict(workers=1,jobs=[dict(job.__dict__,job_id=job.job_id)])),encoding='utf8')
    (root/'EXPERIMENT_RECORDS.json').write_text(json.dumps([dict(job_id=job.job_id,program_error=False,summary=summary)]),encoding='utf8')
    return directory


@pytest.mark.parametrize('field,value',[('risk_method','A5'),('obstacle_predictor','cv'),
    ('seed',123),('scenario','crossing3d'),('planning_delay_mode','measured')])
def test_offline_analysis_rejects_wrong_actual_method_predictor_scene(tmp_path,field,value):
    directory=_raw_source(tmp_path)
    path=directory/'summary.json';saved=json.loads(path.read_text(encoding='utf8'))
    saved[field]=value;path.write_text(json.dumps(saved),encoding='utf8')
    with pytest.raises(ValueError,match='actual summary'):
        analysis.load_episodes([tmp_path])


def test_offline_analysis_rejects_wrong_actual_config_and_metadata(tmp_path):
    directory=_raw_source(tmp_path)
    cfg=directory/'config.json';saved=json.loads(cfg.read_text(encoding='utf8'))
    saved['current_speed']+=.1;cfg.write_text(json.dumps(saved),encoding='utf8')
    with pytest.raises(ValueError,match='actual configuration'):analysis.load_episodes([tmp_path])
    cfg.write_text(json.dumps(config_for_case(case()).to_dict()),encoding='utf8')
    path=directory/'stage5_experiment.json';saved=json.loads(path.read_text(encoding='utf8'))
    saved['actual_predictor']='cv';path.write_text(json.dumps(saved),encoding='utf8')
    with pytest.raises(ValueError,match='actual stage5 metadata'):analysis.load_episodes([tmp_path])


def _real_shadow(monkeypatch, *, mutate=False):
    from auv3d.remus import RemusPlant
    from auv3d.adapters.remus_guidance import ReferenceAction
    import auv3d.remus_planner as planning
    plant=RemusPlant(depth=30.,current_speed=0.,initialize_references=True)
    action=ReferenceAction(30.,0.,800.)
    frame=SimpleNamespace(plant=plant,cfg=Stage2Config(),previous_action=None,
        memory_uncertainty={},goal=np.array([100.,0.,30.]),terrain=None,tracks=[],now=2.,
        original_action=action,original_feasible=True,original_predicted_clearance_m=2.)
    calls=[]
    class Planner:
        def __init__(self,cfg,*,risk_policy,forecaster):self.method=risk_policy.method
        def set_memory_uncertainty(self,value):pass
        def plan(self,plant,goal,terrain,tracks,now):
            calls.append(self.method)
            if mutate and self.method=='A1':plant.vehicle.z_d+=1.
            return SimpleNamespace(action=action,feasible=True,minimum_predicted_clearance_m=2.,
                shortlist_indices=(0,),valid_count=1,verified_count=1,planning_ms=1.)
    monkeypatch.setattr(planning,'RemusDWA',Planner)
    monkeypatch.setattr(analysis,'_policy_and_forecaster',lambda job,cfg:(SimpleNamespace(method=job.method),None))
    jobs=[exp.ExperimentJob(case('tune')),exp.ExperimentJob(case('tune'),method='A1',fixed_margin_m=0.)]
    return frame,jobs,calls


def test_shadow_clones_real_module_bearing_plant_and_rotates_timed_order(monkeypatch):
    frame,jobs,calls=_real_shadow(monkeypatch)
    rows=analysis.shadow_frame(frame,jobs)
    assert calls[0]=='A0' and len(calls)==3 # archive check is not a timed sample
    assert len(rows)==2 and all(row['untimed_A0_replay_checked'] for row in rows)
    expected=(round(frame.now/frame.cfg.control_dt)+jobs[0].case['seed'])%2
    assert rows[0]['method']==jobs[expected].method


def test_shadow_detects_controller_reference_mutation_even_with_same_pose(monkeypatch):
    frame,jobs,_=_real_shadow(monkeypatch,mutate=True)
    with pytest.raises(RuntimeError,match='controller.z_d'):
        analysis.shadow_frame(frame,jobs)


def test_pointwise_coverage_retained_separately_from_full_path_intersection():
    label=dict(selected_forecast_false_feasible=False,prediction_errors_by_horizon={},
        stage5_score_covered=False,coverage_by_horizon_s={
            '3':dict(covered=1,label_count=1),'6':dict(covered=0,label_count=1)})
    result=analysis.summarize_labels([label],'A2')
    assert result['calibration_coverage']==0.
    assert result['calibration_coverage_by_horizon']['3']==dict(covered=1,label_count=1)
    assert 'not pointwise' in result['calibration_coverage_scope']


def test_formal_cli_calibration_uses_family_rule_not_seed_retry(monkeypatch,tmp_path,capsys):
    import run_stage5
    import auv3d.stage5_calibration as cal
    selected=[]
    def fake_build(rows,out,*,predictors):
        selected.extend(rows)
        return []
    monkeypatch.setattr(cal,'build_libraries',fake_build)
    assert run_stage5.main(['build-calibration','--phase','formal','--noise','.02',
        '--predictors','ca','--out',str(tmp_path/'cal')])==0
    assert len(selected)==48 and {r['family'] for r in selected}==set(cal.OBSTACLE_FAMILIES)
    registry=json.loads((tmp_path/'cal'/'ARTIFACT_REGISTRY.json').read_text(encoding='utf8'))
    assert registry['registered_calibration_geometry_count']==72
    assert registry['included_calibration_geometry_count']==48
    assert 'never by seed/outcome' in registry['calibration_family_scope']
    capsys.readouterr()
