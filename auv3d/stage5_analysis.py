"""Post-execution Stage 5 labels, tune-only freezing and state counterfactuals."""
from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path

import numpy as np

from .stage4_experiments import write_json, write_csv
from .stage5_experiments import (ExperimentJob, METRIC_FIELDS, episode_metrics,
                                grouped_results, _policy_and_forecaster)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


TUNING_RULE = ('Maximize arrived tasks with no collision and actual 1m margin; '
    'then minimize collisions, actual margin violations, timeouts, summed '
    'no-feasible decisions, and finally fixed margin. Failures remain in every denominator.')


def freeze_fixed_margin(study_roots, output):
    """Freeze A1 using complete paired tune data; never read a test episode."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('existing tuned margin is preserved')
    groups = defaultdict(lambda: defaultdict(list))
    phases, sources = set(), []
    for root in study_roots:
        root = Path(root).resolve()
        registration = _read(root/'EXPERIMENT_CONFIG.json')
        jobs = registration['jobs']
        if not jobs or any(job['method'] != 'A1' or job['case']['split'] != 'tune'
                           or job.get('tuning_path') for job in jobs):
            raise ValueError('fixed-margin freezing only accepts independent A1 tune scans')
        # Check registration before reading metrics or any raw result.
        rows = _read(root/'EXPERIMENT_METRICS.json')
        if len(rows) != len(jobs) or any(row['split'] != 'tune' or row['method'] != 'A1' for row in rows):
            raise ValueError('tune metrics disagree with registered independent jobs')
        for row in rows:
            if (row['program_error'] or row['navigation_success'] is None
                    or row['collision'] is None or row['actual_clearance_margin_violation'] is None):
                raise ValueError('resolve incomplete/program-error tuning evidence before freezing')
            if row['planning_delay_mode'] != 'none' or row['planning_delay_s'] != 0:
                raise ValueError('A1 freeze uses the predeclared no-delay tune condition only')
            phases.add(row['phase'])
            key = (row['predictor'], row['noise_std_m'], row['risk_combination'])
            groups[key][row['fixed_margin_m']].append(row)
        sources.append(str(root))
    if len(phases) != 1:
        raise ValueError('do not pool pilot/formal fixed-margin tuning')
    selections = []
    for key, levels in sorted(groups.items()):
        from .stage5_protocol import manifest
        expected_ids = {row['case_id'] for row in manifest(next(iter(phases)),split='tune',noise_std=key[1])}
        ids = None
        candidates = []
        for margin, rows in sorted(levels.items()):
            current = [row['case_id'] for row in rows]
            if len(set(current)) != len(current):
                raise ValueError('duplicate tuning case/margin evidence')
            if ids is not None and set(current) != ids:
                raise ValueError('every tuning margin must use identical paired cases')
            ids = set(current)
            if ids != expected_ids:
                raise ValueError('every fixed margin must cover all registered independent tune scenes')
            safe_arrival = sum(row['navigation_success'] and not row['collision']
                and not row['actual_clearance_margin_violation'] for row in rows)
            entry = dict(fixed_margin_m=margin, episode_count=len(rows),
                safe_arrival_count=safe_arrival,
                arrival_count=sum(row['navigation_success'] for row in rows),
                collision_count=sum(row['collision'] for row in rows),
                actual_margin_violation_count=sum(row['actual_clearance_margin_violation'] for row in rows),
                timeout_count=sum(row['timeout'] for row in rows),
                no_feasible_candidate_count=sum(row['no_feasible_candidate_count'] for row in rows),
                unsafe_emergency_episode_count=sum(bool(row['unsafe_emergency_forecast']) for row in rows),
                total_duration_s=sum(row['duration_s'] for row in rows))
            candidates.append(entry)
        best = min(candidates,key=lambda row:(-row['safe_arrival_count'],row['collision_count'],
            row['actual_margin_violation_count'],row['timeout_count'],
            row['no_feasible_candidate_count'],row['fixed_margin_m']))
        selections.append(dict(predictor=key[0],noise_std_m=key[1],risk_combination=key[2],
            fixed_margin_m=best['fixed_margin_m'],source_case_ids=sorted(ids),candidates=candidates))
    if not selections:
        raise ValueError('no complete independent tuning groups')
    record = dict(schema='stage5_fixed_margin_tuning_v1',source_split='tune',
        source_phase=next(iter(phases)),test_data_used=False,selection_rule=TUNING_RULE,
        source_roots=sources,selections=selections)
    output.parent.mkdir(parents=True,exist_ok=True)
    write_json(output,record)
    return record


def load_episodes(roots):
    episodes, seen = [], set()
    for root in roots:
        root = Path(root).resolve()
        registration = _read(root/'EXPERIMENT_CONFIG.json')
        records = {row['job_id']:row for row in _read(root/'EXPERIMENT_RECORDS.json')}
        for entry in registration['jobs']:
            data = {key:value for key,value in entry.items() if key != 'job_id'}
            job = ExperimentJob(**data)
            job.validate()
            if entry['job_id'] != job.job_id or job.job_id in seen:
                raise ValueError('duplicate or inconsistent registered job IDs')
            seen.add(job.job_id)
            directory = root/'cases'/job.job_id
            record = records[job.job_id]
            if record['program_error'] or (directory/'program_error.json').exists():
                raise ValueError('incomplete program-error episodes require separate reporting; do not impute labels')
            for name in ('summary.json','config.json','states.csv','controls.csv',
                         'selected_plans.json','planning_diagnostics.json','observations.json',
                         'obstacles.json','stage5_experiment.json'):
                if not (directory/name).is_file():
                    raise FileNotFoundError(f'completed raw episode is required: {directory/name}')
            from .stage5_protocol import config_for_case
            from .stage2_config import Stage2Config
            actual_summary = _read(directory/'summary.json')
            expected_summary = dict(backend='remus6dof',risk_method=job.method,
                obstacle_predictor=job.predictor,seed=job.case['seed'],scenario=job.case['scenario'],
                planning_delay_mode=job.planning_delay_mode)
            if any(actual_summary.get(key) != value for key,value in expected_summary.items()):
                raise ValueError('actual summary method/predictor/scene/delay differs from the registered job')
            if Stage2Config.load(directory/'config.json').to_dict() != config_for_case(job.case).to_dict():
                raise ValueError('actual configuration differs from its registered common case')
            actual_metadata = _read(directory/'stage5_experiment.json')
            if (actual_metadata.get('job') != job.__dict__
                    or actual_metadata.get('actual_predictor') != job.predictor):
                raise ValueError('actual stage5 metadata differs from its registered method/predictor factors')
            if record.get('summary') != actual_summary:
                raise ValueError('study record disagrees with the actual raw episode summary')
            episodes.append(dict(job=job,directory=directory,record=record,
                                 workers=registration['workers']))
    return episodes


def summarize_labels(labels, method):
    errors = defaultdict(lambda:dict(sample_count=0,squared_error_sum_m2=0.))
    coverage = []
    per_time = defaultdict(lambda:dict(covered=0,label_count=0))
    for label in labels:
        for horizon,values in label['prediction_errors_by_horizon'].items():
            errors[horizon]['sample_count'] += values['sample_count']
            errors[horizon]['squared_error_sum_m2'] += values['squared_error_sum_m2']
        if label.get('stage5_score_covered') is not None:
            coverage.append(label['stage5_score_covered'])
        for stamp,value in label.get('coverage_by_horizon_s',{}).items():
            per_time[stamp]['covered'] += value['covered']
            per_time[stamp]['label_count'] += value['label_count']
    rmses = {h:(float(np.sqrt(v['squared_error_sum_m2']/v['sample_count']))
                 if v['sample_count'] else None) for h,v in errors.items()}
    return dict(evaluated_selected_reference_anchor_count=len(labels),
        candidate_false_accept_count=sum(label['selected_forecast_false_feasible'] for label in labels),
        candidate_false_reject_count=None,
        candidate_misclassification_scope='Selected reference held for the entire 24s at registered anchors; not an all-grid false-reject rate or actual task collision count.',
        calibration_coverage=float(np.mean(coverage)) if coverage else None,
        calibration_coverage_count=sum(coverage) if coverage else None,
        calibration_label_count=len(coverage),calibration_coverage_by_horizon=dict(per_time),
        calibration_coverage_scope=('No calibrated score term in A0/A1; coverage not assigned.'
            if method in ('A0','A1') else
            'Full selected-reference intersection over every labeled time and observed speed bin; not pointwise coverage. Per-horizon covered/label counts are separate. Conditional on nonempty observations, separate from engineering-envelope coverage and not a closed-loop guarantee.'),
        prediction_RMSE_by_horizon=rmses,prediction_errors_by_horizon=dict(errors),
        prediction_RMSE_scope='Matched absolute-time predictor errors from delivered causal histories; labels are evaluated offline only.')


def _score_coverage(frame, prediction, forecaster, truth, risk, method):
    """Compare frozen per-bin/time quantiles with independent offline labels."""
    if method in ('A0','A1') or not frame.tracks:
        return dict(stage5_score_covered=None,coverage_by_horizon_s={})
    from .stage5_risk import time_ball_scores, time_ecp_scores
    from .stage3_forecasting import observation_speed
    offsets = np.asarray(risk.get('offsets_s',risk.get('applied_q_offsets_s')),float)
    times = frame.now+offsets
    truth._ensure(round(float(times[-1])/frame.cfg.physics_dt))
    ids = {int(ident):index for index,ident in enumerate(truth.ids)}
    indices = np.rint(times/frame.cfg.physics_dt).astype(int)
    if not np.allclose(indices*frame.cfg.physics_dt,times,atol=1e-8,rtol=0):
        raise ValueError('offline score labels must align with the physical clock')
    forecasts, labels, radii, speeds = [], [], [], []
    for track in frame.tracks:
        if track.object_id not in ids:
            raise ValueError('observed target is absent from independently replayed truth')
        forecast = forecaster.forecast(track,times,frame.now,
            frame.memory_uncertainty.get(track.object_id,0.))
        forecasts.append(forecast.positions)
        labels.append([truth._positions[index][ids[track.object_id]] for index in indices])
        radii.append(track.radius_m)
        speeds.append(observation_speed(track))
    pred, label = np.asarray(forecasts)[None,None],np.asarray(labels)[None,None]
    mask = np.ones((1,1,len(frame.tracks)),dtype=bool)
    speeds = np.asarray(speeds)[None,None]
    edges = np.asarray(risk['speed_bin_edges_mps'],float)
    edges[-1] = np.inf
    if method in ('A3','A4'):
        from .stage4_risk import sphere_scene_scores, ecp_scene_scores
        if method == 'A3':
            score = float(sphere_scene_scores(pred,label,mask)[0])
        else:
            path = np.column_stack([np.interp(offsets,prediction.states[:,0],prediction.states[:,column])
                                    for column in (1,2,3)])
            score = float(ecp_scene_scores(path,pred,label,np.asarray(radii)[None,None],
                                          mask,frame.cfg.body_radius_m)[0])
        covered = bool(score <= float(risk['quantile_m'])+1e-9)
        return dict(stage5_score_covered=covered,
            coverage_by_horizon_s={f'{offsets[-1]:g}':dict(covered=int(covered),label_count=1)},
            joint_scene_score_m=score,
            calibrated_score_scope='One joint full-horizon selected-reference score, all observed objects; not per-speed maximum of separate nearest gaps.')
    if method == 'A2':
        scores,present = time_ball_scores(pred,label,mask,speeds,edges)
    else:
        path = np.column_stack([np.interp(offsets,prediction.states[:,0],prediction.states[:,column])
                                for column in (1,2,3)])
        scores,present = time_ecp_scores(path,pred,label,np.asarray(radii)[None,None],
            mask,speeds,edges,frame.cfg.body_radius_m)
    score = np.asarray(scores)[0]
    present = np.asarray(present)[0]
    q = np.asarray(risk['q_by_speed_bin_m'],float)
    if q.shape != score.shape:
        raise ValueError('calibrated per-time quantiles must match score bins and horizons')
    if not np.any(present):
        return dict(stage5_score_covered=None,coverage_by_horizon_s={})
    active = score[present]
    covered = active <= q[present]+1e-9
    if not np.isfinite(active).all() or np.isnan(q[present]).any():
        raise ValueError('active offline calibration scores and quantiles cannot be missing')
    horizon_offsets = offsets if method == 'A2' else offsets[1:]
    per_time = {f'{stamp:g}':dict(covered=int(np.sum(covered[:,index])),
                label_count=int(len(covered))) for index,stamp in enumerate(horizon_offsets)}
    return dict(stage5_score_covered=bool(np.all(covered)),coverage_by_horizon_s=per_time,
        score_by_speed_bin_m=[row.tolist() if present[i] else None for i,row in enumerate(score)],
        calibrated_score_scope='Frozen score quantiles at calibration query times; observed speed bins only; no missing-bin zero labels.')


def _assert_numeric_state_equal(before, after, path='state'):
    """Check full numeric controller state, including nested reference memory."""
    if isinstance(before,dict):
        if not isinstance(after,dict) or before.keys() != after.keys():
            raise RuntimeError(f'shadow prediction changed {path} fields')
        for key,value in before.items():
            _assert_numeric_state_equal(value,after[key],path+'.'+str(key))
    elif isinstance(before,(list,tuple)):
        if not isinstance(after,(list,tuple)) or len(before) != len(after):
            raise RuntimeError(f'shadow prediction changed {path} sequence')
        for index,(left,right) in enumerate(zip(before,after)):
            _assert_numeric_state_equal(left,right,f'{path}[{index}]')
    elif isinstance(before,np.ndarray):
        if not isinstance(after,np.ndarray) or not np.array_equal(before,after):
            raise RuntimeError(f'shadow prediction changed {path} array')
    elif isinstance(before,(int,float,complex,np.generic,bool,str,type(None))):
        if type(before) != type(after) or before != after:
            raise RuntimeError(f'shadow prediction changed {path} value')
    elif hasattr(before,'__dict__'):
        _assert_numeric_state_equal(vars(before),vars(after),path)


def shadow_frame(frame, jobs):
    """Untimed archive check followed by rotated same-state measured decisions."""
    from .remus_planner import RemusDWA
    from .adapters.remus_guidance import clone_plant
    original = clone_plant(frame.plant)
    jobs = list(jobs)
    if not jobs or jobs[0].method != 'A0':
        raise ValueError('same-state attribution requires A0 as the first reference')

    def unchanged():
        for key in ('eta','nu','actual','time_s'):
            _assert_numeric_state_equal(getattr(original,key),getattr(frame.plant,key),'plant.'+key)
        _assert_numeric_state_equal(vars(original.vehicle),vars(frame.plant.vehicle),'controller')

    def evaluate(job):
        policy,forecaster = _policy_and_forecaster(job,frame.cfg)
        planner = RemusDWA(frame.cfg,risk_policy=policy,forecaster=forecaster)
        planner.previous = copy.deepcopy(frame.previous_action)
        planner.set_memory_uncertainty(frame.memory_uncertainty)
        result = planner.plan(frame.plant,frame.goal,frame.terrain,frame.tracks,frame.now)
        unchanged()
        return result

    # This archive check is deliberately not a measured method sample.
    baseline = evaluate(jobs[0])
    if (not np.allclose(baseline.action.as_array(),frame.original_action.as_array(),atol=1e-7,rtol=0)
            or baseline.feasible != frame.original_feasible
            or not np.isclose(baseline.minimum_predicted_clearance_m,frame.original_predicted_clearance_m,atol=1e-7,rtol=0)):
        raise RuntimeError('recomputed A0 differs from the archived decision; attribution is invalid')
    # The registered seed balances parity across scenes; planning cycle rotates
    # order across anchors. Neither term depends on an experiment outcome.
    rotation = (int(round(frame.now/frame.cfg.control_dt))+int(jobs[0].case['seed'])) % len(jobs)
    ordered = jobs[rotation:]+jobs[:rotation]
    rows = []
    for index,job in enumerate(ordered):
        result = evaluate(job)
        if result.shortlist_indices != baseline.shortlist_indices:
            raise RuntimeError('risk methods must share their fully verified shortlist')
        rows.append(dict(case_id=job.case['case_id'],scene_id=job.case['scene_id'],method=job.method,
            predictor=job.predictor,noise_std_m=job.noise,alpha=job.alpha,combination=job.combination,
            planning_time_s=frame.now,action=result.action.as_array().tolist(),feasible=result.feasible,
            valid_count=result.valid_count,verified_count=result.verified_count,
            changed_action_from_A0=not np.allclose(result.action.as_array(),baseline.action.as_array(),atol=1e-9,rtol=0),
            planning_ms=result.planning_ms,deadline_miss=result.planning_ms>frame.cfg.control_dt*1000,
            timing_order_index=index,timing_rotation=rotation,untimed_A0_replay_checked=True,
            scope='Offline same complete controller state and shortlist; untimed A0 archive check excluded; timing order rotates by registered seed and planning cycle. Isolated process timing must be scheduled externally.'))
    return rows


def analyze_studies(roots, output, *, anchors=(20.,40.,60.), shadow=False):
    from .stage2_config import Stage2Config
    from .stage4_diagnostics import replay_anchors, OfflineTruthReplay
    from .stage5_protocol import make_scene
    from .adapters.remus_guidance import predict_reference
    from .stage3_forecasting import ForecastContext
    from .remus_planner import trajectory_clearance
    out = Path(output).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('existing analyses are preserved')
    episodes = load_episodes(roots)
    out.mkdir(parents=True,exist_ok=True)
    (out/'episodes').mkdir()
    metrics, all_shadow = [], []
    for episode in episodes:
        job,directory,record = episode['job'],episode['directory'],episode['record']
        cfg = Stage2Config.load(directory/'config.json')
        policy,forecaster = _policy_and_forecaster(job,cfg)
        truth = OfflineTruthReplay(cfg,job.case['scenario'],_read(directory/'obstacles.json'),
            scene=make_scene(job.case,cfg),forecaster=forecaster)
        labels = []
        for frame in replay_anchors(directory,cfg,job.case['scenario'],anchors,scene=make_scene(job.case,cfg)):
            prediction = predict_reference(frame.plant,frame.original_action,cfg.horizon_s,cfg.physics_dt,record_controls=False)
            context = ForecastContext(cfg,predictor=forecaster)
            context.begin(frame.memory_uncertainty)
            base = trajectory_clearance(prediction.states,frame.terrain,frame.tracks,frame.now,cfg,context)
            risk = {}
            if policy is not None:
                _,risk = policy.adjust(prediction.states,frame.terrain,frame.tracks,frame.now,cfg,context,base)
            label = truth.evaluate(prediction.states,frame.tracks,frame.now,feasible=frame.original_feasible,
                uncertainty_by_id=frame.memory_uncertainty)
            label.update(_score_coverage(frame,prediction,forecaster,truth,risk,job.method))
            labels.append(label)
            if shadow and job.method == 'A0' and job.case['split'] == 'test':
                alternatives = [item['job'] for item in episodes if item['job'].case['case_id']==job.case['case_id']
                    and item['job'].predictor==job.predictor
                    and item['job'].planning_delay_mode==job.planning_delay_mode
                    and item['job'].planning_delay_s==job.planning_delay_s]
                alternatives.sort(key=lambda item:item.method)
                all_shadow.extend(shadow_frame(frame,alternatives))
        offline = summarize_labels(labels,job.method)
        write_json(out/'episodes'/(job.job_id+'.json'),dict(job=job.__dict__,summary=offline,anchors=labels))
        enriched = dict(record,**offline)
        metrics.append(episode_metrics(job,enriched,episode['workers'],cfg=cfg))
        print(f'OFFLINE {job.job_id}: {len(labels)} selected anchors',flush=True)
    write_json(out/'EXPERIMENT_METRICS.json',metrics)
    write_csv(out/'EXPERIMENT_SUMMARY.csv',[dict(row,prediction_RMSE_by_horizon=json.dumps(row['prediction_RMSE_by_horizon']),
        calibration_coverage_by_horizon=json.dumps(row['calibration_coverage_by_horizon'])) for row in metrics],METRIC_FIELDS)
    write_json(out/'GROUPED_SUMMARY.json',grouped_results(metrics))
    write_json(out/'SAME_STATE_SHADOW.json',all_shadow)
    if all_shadow:
        write_csv(out/'SAME_STATE_SHADOW.csv',[dict(row,action=json.dumps(row['action'])) for row in all_shadow],tuple(all_shadow[0]))
    result = dict(episode_count=len(metrics),selected_reference_anchor_count=sum(row['evaluated_selected_reference_anchor_count'] for row in metrics),
        shadow_decision_count=len(all_shadow),output_directory=str(out),
        scope='Offline future truth is not supplied to execution; all registered research failures retained. Labels describe held-reference counterfactuals only.')
    write_json(out/'ANALYSIS_SUMMARY.json',result)
    return result
