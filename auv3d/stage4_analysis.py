"""Read-only paired action analysis of completed REMUS risk episodes.

Different risk methods generally reach different states. Matching wall/physical
time alone is therefore not a same-state causal action comparison. This module
reports that distinction and reserves exact-controller counterfactuals for the
separate replay diagnostics.
"""
from __future__ import annotations

import json
import copy
from pathlib import Path

import numpy as np

from .stage4_experiments import (write_csv, write_json, episode_metrics,
                                  ExperimentJob, grouped_results, METRIC_FIELDS)


ACTION_FIELDS = (
    'case_id', 'baseline_episode', 'comparison_episode', 'risk_method', 'risk_alpha',
    'planning_time_s', 'baseline_feasible', 'comparison_feasible',
    'action_changed', 'same_recorded_start_state', 'start_position_difference_m',
    'baseline_depth_reference_m', 'comparison_depth_reference_m',
    'baseline_heading_reference_rad', 'comparison_heading_reference_rad',
    'baseline_rpm_reference', 'comparison_rpm_reference', 'interpretation',
)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_saved_plans(source):
    """Validate saved plan times/actions without reconstructing a controller."""
    source = Path(source)
    summary = _read(source/'summary.json')
    if summary.get('backend') != 'remus6dof':
        raise ValueError('paired action comparisons require actual REMUS episodes')
    plans = _read(source/'selected_plans.json')
    previous = -float('inf')
    for plan in plans:
        stamp = float(plan['time_s'])
        action = np.asarray(plan['action'], dtype=float)
        states = np.asarray(plan['states'], dtype=float)
        if not np.isfinite(stamp) or stamp <= previous:
            raise ValueError('saved plan anchors must be finite and strictly increasing')
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError('saved plan action must be finite [depth,heading,rpm]')
        if states.ndim != 2 or states.shape[1] != 16 or not len(states) or not np.isfinite(states).all():
            raise ValueError('saved plan states must be finite [offset,eta6,nu6,actuator3]')
        if abs(states[0, 0]) > 1e-9:
            raise ValueError('saved plan must begin at offset zero')
        previous = stamp
    return summary, plans


def compare_actions(case_id, baseline, comparison):
    base_summary, base_plans = load_saved_plans(baseline)
    comp_summary, comp_plans = load_saved_plans(comparison)
    if (base_summary.get('seed') != comp_summary.get('seed')
            or base_summary.get('scenario') != comp_summary.get('scenario')):
        raise ValueError('paired actions must use the same scene and seed')
    if base_summary.get('risk_method', 'B0') != 'B0':
        raise ValueError('baseline action reference must be B0')
    base_by_time = {round(float(plan['time_s']), 9): plan for plan in base_plans}
    rows = []
    for plan in comp_plans:
        stamp = round(float(plan['time_s']), 9)
        if stamp not in base_by_time:
            continue
        original = base_by_time[stamp]
        a, b = np.asarray(original['action']), np.asarray(plan['action'])
        s0, s1 = np.asarray(original['states'])[0, 1:], np.asarray(plan['states'])[0, 1:]
        same = bool(np.allclose(s0, s1, atol=1e-9, rtol=0))
        row = dict.fromkeys(ACTION_FIELDS)
        row.update(case_id=case_id, baseline_episode=str(Path(baseline).resolve()),
                   comparison_episode=str(Path(comparison).resolve()),
                   risk_method=comp_summary.get('risk_method'),
                   risk_alpha=comp_summary.get('risk_metadata', {}).get('alpha'),
                   planning_time_s=stamp, baseline_feasible=bool(original['feasible']),
                   comparison_feasible=bool(plan['feasible']),
                   action_changed=bool(not np.allclose(a, b, atol=1e-9, rtol=0)),
                   same_recorded_start_state=same,
                   start_position_difference_m=float(np.linalg.norm(s0[:3]-s1[:3])),
                   baseline_depth_reference_m=float(a[0]), comparison_depth_reference_m=float(b[0]),
                   baseline_heading_reference_rad=float(a[1]), comparison_heading_reference_rad=float(b[1]),
                   baseline_rpm_reference=float(a[2]), comparison_rpm_reference=float(b[2]),
                   interpretation=('same recorded pose, velocity and actuators; controller memory is not '
                                   'reconstructed by this comparison' if same else
                                   'different executed states at a common planning time; this row is '
                                   'not a same-controller-state counterfactual'))
        rows.append(row)
    return rows


def summarize_actions(rows):
    groups = {}
    for row in rows:
        key = (row['case_id'], row['risk_method'], row['risk_alpha'])
        groups.setdefault(key, []).append(row)
    result = []
    for key, frames in sorted(groups.items(), key=lambda item: str(item[0])):
        changed = [row for row in frames if row['action_changed']]
        matched = [row for row in frames if row['same_recorded_start_state']]
        result.append(dict(case_id=key[0], risk_method=key[1], risk_alpha=key[2],
                           common_planning_anchor_count=len(frames),
                           changed_action_count=len(changed),
                           changed_action_fraction=len(changed)/len(frames),
                           first_changed_action_time_s=changed[0]['planning_time_s'] if changed else None,
                           same_recorded_start_state_count=len(matched),
                           changed_action_at_same_recorded_start_state_count=sum(row['action_changed'] for row in matched),
                           scope='Paired online episodes; after divergence this does not isolate risk at the same controller state.'))
    return result


def paired_action_study(baseline_root, comparison_roots, output):
    """Write new analyses; never modify or silently drop failed episodes."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('existing analysis is preserved; choose a fresh output directory')
    baseline_root = Path(baseline_root).resolve()
    rows, seen = [], set()
    for root in comparison_roots:
        root = Path(root).resolve()
        registration = _read(root/'EXPERIMENT_CONFIG.json')
        for job in registration['jobs']:
            if job['method'] == 'B0':
                continue
            key = (job['case_id'], job['method'], job['alpha'])
            if key in seen:
                raise ValueError('duplicate paired method/case/risk level; preserve studies separately')
            seen.add(key)
            level = f"{job['case_id']}__{job['method']}__alpha{job['alpha']:g}"
            comparison = root/'cases'/level
            # A missing or program-error episode is a blocking incomplete
            # analysis, never a reason to remove a difficult case silently.
            if not (comparison/'summary.json').is_file():
                raise FileNotFoundError(f'completed comparison evidence is missing: {comparison}')
            rows.extend(compare_actions(job['case_id'], baseline_root/'cases'/job['case_id'], comparison))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'PAIRED_ACTION_SELECTION.csv', rows, ACTION_FIELDS)
    summary = summarize_actions(rows)
    write_json(output/'PAIRED_ACTION_SUMMARY.json', summary)
    return {'paired_episode_count': len(seen), 'common_anchor_count': len(rows),
            'output_directory': str(output)}


def _study_episodes(roots):
    episodes, seen = [], set()
    for root in roots:
        root = Path(root).resolve()
        registration = _read(root/'EXPERIMENT_CONFIG.json')
        records_path = root/'EXPERIMENT_RECORDS.json'
        records = _read(records_path) if records_path.is_file() else []
        records_by_job = {row['job_id']: row for row in records if 'job_id' in row}
        cases = {row['case_id']: row for row in registration['registered_cases']}
        for job in registration['jobs']:
            key = (job['case_id'], job['method'], job['alpha'])
            if key in seen:
                raise ValueError('duplicate method/case/alpha across analysis roots')
            seen.add(key)
            suffix = '' if job['method'] == 'B0' else f"__{job['method']}__alpha{job['alpha']:g}"
            source = root/'cases'/f"{job['case_id']}{suffix}"
            if (source/'program_error.json').is_file():
                raise ValueError(f'raw episode has a recorded program error: {source}')
            if not (source/'summary.json').is_file():
                raise FileNotFoundError(f'completed raw episode is required: {source}')
            episodes.append(dict(job=job, case=cases[job['case_id']], source=source,
                                 record=records_by_job.get(source.name),
                                 workers=registration['workers']))
    return episodes


def summarize_labels(frames, method):
    """Weighted forecast RMSE and selected-reference empirical diagnostics."""
    errors = {}
    for frame in frames:
        for horizon, values in frame.get('prediction_errors_by_horizon', {}).items():
            total = errors.setdefault(str(horizon), {'sample_count': 0, 'squared_error_sum_m2': 0.})
            total['sample_count'] += int(values['sample_count'])
            total['squared_error_sum_m2'] += float(values['squared_error_sum_m2'])
    rmses = {horizon: (float(np.sqrt(values['squared_error_sum_m2']/values['sample_count']))
                      if values['sample_count'] else None)
             for horizon, values in errors.items()}
    coverage_key = 'coverage_ball' if method == 'B1' else 'coverage_ecp'
    covered = [row[coverage_key] for row in frames
               if row.get(coverage_key) is not None] if method != 'B0' else []
    return dict(
        evaluated_selected_reference_anchor_count=len(frames),
        candidate_false_accept_count=sum(bool(row['selected_forecast_false_feasible']) for row in frames),
        candidate_false_reject_count=None,
        candidate_misclassification_scope=(
            'Selected held-reference forecasts at preregistered anchors only; independent '
            'future truth over the full 24s, including unobserved obstacles. Not all '
            'candidates, not actual collision count, and not a full-grid false-reject rate.'),
        calibration_coverage=(float(np.mean(covered)) if covered else None),
        calibration_coverage_count=sum(bool(value) for value in covered) if covered else None,
        calibration_label_count=len(covered),
        calibration_coverage_scope=(
            'B0 has no additional calibrated term; no calibration coverage is assigned. '
            'Its original age-dependent engineering envelope is retained and is not '
            'represented by a zero calibrated quantile.' if method == 'B0' else
            'Empirical selected-reference score coverage at fixed anchors, conditional '
            'on a nonempty observed set; frozen extra calibration term, separate '
            'from the retained shared engineering envelope. This is not overall '
            'envelope coverage or a closed-loop coverage theorem.'),
        prediction_RMSE_by_horizon=rmses,
        prediction_errors_by_horizon=errors,
        prediction_RMSE_status='offline CA errors against synchronized truth from delivered histories only',
    )


def _shadow_frame(frame, policies, alphas):
    from .remus_planner import RemusDWA
    rows = []
    original = (frame.plant.eta.copy(), frame.plant.nu.copy(), frame.plant.actual.copy(),
                float(frame.plant.time_s))
    base = None
    for method, alpha in [('B0', None)] + [(method, alpha) for alpha in alphas for method in ('B1', 'B2')]:
        planner = RemusDWA(frame.cfg, risk_policy=policies.get((method, alpha)))
        planner.previous = copy.deepcopy(frame.previous_action)
        planner.set_memory_uncertainty(frame.memory_uncertainty)
        plan = planner.plan(frame.plant, frame.goal, frame.terrain, frame.tracks, frame.now)
        if base is None:
            if (not np.allclose(plan.action.as_array(), frame.original_action.as_array(),
                                atol=1e-7, rtol=0)
                    or bool(plan.feasible) != bool(frame.original_feasible)
                    or not np.isclose(plan.minimum_predicted_clearance_m,
                                      frame.original_predicted_clearance_m,
                                      atol=1e-7, rtol=0)):
                raise RuntimeError('recomputed B0 decision does not match archived action, feasibility or clearance; same-state attribution is invalid')
            base = plan
        if (not np.array_equal(frame.plant.eta, original[0])
                or not np.array_equal(frame.plant.nu, original[1])
                or not np.array_equal(frame.plant.actual, original[2])
                or frame.plant.time_s != original[3]):
            raise RuntimeError('offline candidate evaluation changed the replayed actual plant')
        row = dict(planning_time_s=frame.now, method=method, alpha=alpha,
                   selected_reference=plan.action.as_array().tolist(), feasible=bool(plan.feasible),
                   valid_candidate_count=plan.valid_count, full_verified_count=plan.verified_count,
                   selected_predicted_clearance_m=plan.minimum_predicted_clearance_m,
                   changed_action_from_B0=bool(not np.allclose(
                       plan.action.as_array(), base.action.as_array(), atol=1e-9, rtol=0)),
                   same_verified_shortlist_as_B0=plan.shortlist_indices == base.shortlist_indices,
                   emergency_verified_safe=bool(plan.emergency_verified_safe),
                   offline_shadow_planning_ms=plan.planning_ms,
                   risk_records=list(plan.risk_records),
                   scope='same reconstructed complete REMUS/controller state; offline cost is excluded from online latency')
        rows.append(row)
    return rows


def analyze_studies(roots, library, output, *, baseline_root=None, alphas=(.1,),
                    anchors=(20., 40., 60.), shadow_case_ids=None):
    """Post-execution truth labels and exact-state risk action counterfactuals.

    Selected-reference labels are evaluated for every completed episode. Shadow
    candidate selection is restricted to B0 independent test cases by default.
    Their full controller histories are replayed, not inferred from CSV poses.
    """
    from .stage2_config import Stage2Config
    from .stage4_risk import RiskPolicy
    from .adapters.remus_guidance import predict_reference
    from .remus_planner import trajectory_clearance
    from .stage3_forecasting import ForecastContext
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('existing analysis is preserved; choose a fresh output directory')
    if not Path(library).is_file():
        raise FileNotFoundError('offline calibrated diagnostics require an existing library')
    from .stage4_diagnostics import replay_anchors, OfflineTruthReplay
    alphas = tuple(alphas)
    anchors = tuple(anchors)
    if (not alphas or len(set(alphas)) != len(alphas)
            or any(not np.isfinite(alpha) or not 0 < alpha < 1 for alpha in alphas)):
        raise ValueError('analysis alpha levels must be unique and within (0,1)')
    if (not anchors or any(not np.isfinite(stamp) or stamp < 0 for stamp in anchors)
            or any(b <= a for a, b in zip(anchors, anchors[1:]))):
        raise ValueError('analysis anchors must be nonnegative, finite and increasing')
    episodes = _study_episodes(roots)
    policies = {(method, alpha): RiskPolicy(method, library_path=library, alpha=alpha)
                for alpha in alphas for method in ('B1', 'B2')}
    # Tuned runs may contain other preregistered levels than the shadow levels.
    for episode in episodes:
        job = episode['job']
        if job['method'] != 'B0' and (job['method'], job['alpha']) not in policies:
            policies[(job['method'], job['alpha'])] = RiskPolicy(
                job['method'], library_path=library, alpha=job['alpha'])
    shadow_ids = set(shadow_case_ids or ())
    eligible = {episode['job']['case_id'] for episode in episodes
                if episode['job']['method'] == 'B0' and episode['case']['split'] == 'test'}
    if shadow_ids - eligible:
        raise ValueError('shadow cases must be supplied B0 independent-test episodes')
    output.mkdir(parents=True, exist_ok=True)
    (output/'episodes').mkdir()
    summaries, all_shadow, metric_rows = [], [], []
    for episode in episodes:
        job, case, source = episode['job'], episode['case'], episode['source']
        cfg = Stage2Config.load(source/'config.json')
        truth = OfflineTruthReplay(cfg, case['scenario'], _read(source/'obstacles.json'))
        labels = []
        for frame in replay_anchors(source, cfg=cfg, scenario=case['scenario'], anchors=anchors):
            prediction = predict_reference(frame.plant, frame.original_action, cfg.horizon_s,
                                           cfg.physics_dt, record_controls=False)
            policy = policies.get((job['method'], job['alpha']))
            context = ForecastContext(cfg)
            context.begin(frame.memory_uncertainty)
            base_clear = trajectory_clearance(prediction.states, frame.terrain, frame.tracks,
                                              frame.now, cfg, context)
            risk = {'quantile_m': None}
            if policy is not None:
                _, risk = policy.adjust(prediction.states, frame.terrain, frame.tracks,
                                        frame.now, cfg, context, base_clear)
            label = truth.evaluate(prediction.states, frame.tracks, frame.now,
                                   quantile_m=risk['quantile_m'],
                                   feasible=frame.original_feasible,
                                   uncertainty_by_id=frame.memory_uncertainty)
            label.update(planning_time_s=frame.now, original_feasible=frame.original_feasible,
                         quantile_m=risk['quantile_m'])
            labels.append(label)
            if (job['method'] == 'B0' and case['split'] == 'test'
                    and (not shadow_ids or case['case_id'] in shadow_ids)):
                for shadow in _shadow_frame(frame, policies, alphas):
                    shadow.update(case_id=case['case_id'], family=case['family'], split=case['split'],
                                  scenario_seed=case['seed'])
                    all_shadow.append(shadow)
        summary = summarize_labels(labels, job['method'])
        summary.update(case_id=case['case_id'], split=case['split'], family=case['family'],
                       risk_method=job['method'], risk_alpha=job['alpha'], source=str(source),
                       actual_episode_status=_read(source/'summary.json')['status'])
        suffix = '' if job['method'] == 'B0' else f"__{job['method']}__alpha{job['alpha']:g}"
        write_json(output/'episodes'/f"{case['case_id']}{suffix}.json", {'summary': summary, 'anchors': labels})
        summaries.append(summary)
        record = dict(episode['record'] or {})
        details = _read(source/'planning_diagnostics.json')
        record.update(summary=_read(source/'summary.json'), program_error=False,
                      summary_path=str(source/'summary.json'),
                      planning_cycle_count=len(details),
                      no_feasible_candidate_count=sum(not row['feasible'] for row in details))
        record.update(summary)
        metric_job = ExperimentJob(case, job['method'], job['alpha'],
                                   str(Path(library).resolve()) if job['method'] != 'B0' else None)
        metric_rows.append(episode_metrics(metric_job, record, episode['workers']))
        print(f"OFFLINE {source.name}: {len(labels)} selected anchors", flush=True)
    write_json(output/'OFFLINE_EPISODE_METRICS.json', summaries)
    write_json(output/'UNIFIED_EXPERIMENT_METRICS.json', metric_rows)
    csv_metrics = [dict(row, prediction_RMSE_by_horizon=json.dumps(
        row['prediction_RMSE_by_horizon'], ensure_ascii=False)) for row in metric_rows]
    write_csv(output/'EXPERIMENT_SUMMARY.csv', csv_metrics, METRIC_FIELDS)
    groups = grouped_results(metric_rows)
    write_json(output/'GROUPED_SUMMARY.json', groups)
    curve_fields = list(dict.fromkeys(key for row in groups for key in row))
    write_csv(output/'SAFETY_EFFICIENCY_CURVES.csv', groups, curve_fields)
    write_json(output/'SAME_STATE_SHADOW_DECISIONS.json', all_shadow)
    shadow_fields = tuple(key for key in (all_shadow[0] if all_shadow else {
        'case_id': None, 'planning_time_s': None, 'method': None, 'alpha': None}))
    csv_shadow = [{key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value)
                   for key, value in row.items()} for row in all_shadow]
    write_csv(output/'SAME_STATE_SHADOW_DECISIONS.csv', csv_shadow, shadow_fields)
    if baseline_root is not None:
        comparison_roots = [root for root in roots if Path(root).resolve() != Path(baseline_root).resolve()]
        if comparison_roots:
            paired_action_study(baseline_root, comparison_roots, output/'paired_actions')
    outcome = {'episode_count': len(summaries), 'selected_reference_anchor_count': sum(
        summary['evaluated_selected_reference_anchor_count'] for summary in summaries),
        'same_state_shadow_decision_count': len(all_shadow),
        'same_state_shadow_changed_from_B0_count': sum(row['changed_action_from_B0'] for row in all_shadow),
        'output_directory': str(output),
        'scope': 'Offline future truth never supplied to execution or fitted risk policy; failed episodes remain included.'}
    write_json(output/'ANALYSIS_SUMMARY.json', outcome)
    return outcome
