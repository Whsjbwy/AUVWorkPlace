"""Assemble completed review evidence; preserve raw failures outside Git.

No file digests are calculated. The archive uses ordinary ZIP packaging and
includes complete numeric outputs for every budget/sensitivity experiment.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    'status', 'reached_goal', 'avoidance_acceptance_passed', 'safety_validation_status',
    'actual_safety_margin_met', 'duration_s', 'goal_distance_m',
    'minimum_model_clearance_m', 'valid_candidates_min', 'valid_candidates_mean',
    'valid_candidates_max', 'emergency_count', 'unsafe_emergency_forecast_count',
    'candidate_prediction_failure_count', 'actual_close_interaction_count',
    'maximum_simultaneous_close_interactions', 'maximum_observed_obstacles',
    'maximum_tracked_obstacles', 'planning_ms_median', 'planning_ms_p95',
    'planning_ms_max', 'planning_exceeds_control_period_count',
    'prediction_position_rmse_m', 'prediction_position_max_error_m', 'all_states_finite',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def assemble(budget, sensitivity, regression, final_tests, documents, out, archive, extras=()):
    budget, sensitivity, regression, final_tests, documents, out, archive = map(
        Path, (budget, sensitivity, regression, final_tests, documents, out, archive))
    study, sens = read(budget/'budget_study.json'), read(sensitivity/'study_summary.json')
    regression_report, final_report = read(regression/'regression_report.json'), read(final_tests/'regression_report.json')
    if not study.get('finished_at') or study.get('process_failure_count') != 0 or study['completed_count'] != 15:
        raise ValueError('the complete 15-case budget study must finish without program errors first')
    if sens.get('case_count') != 7 or sens.get('program_failures') != 0:
        raise ValueError('the complete seven-case sensitivity study is required')
    if not regression_report.get('all_passed') or not final_report.get('all_passed'):
        raise ValueError('regression and final tests must actually pass first')
    if out.exists() or archive.exists():
        raise FileExistsError('delivery directory and archive must both be new')
    for name in ('TEST_REPORT.md', 'CHANGELOG.md', 'FAILURE_ANALYSIS.md'):
        if not (documents/name).is_file():
            raise FileNotFoundError(documents/name)

    rows, configs = [], {}
    for record in study['records']:
        case_dir = budget/record['name']
        summary, cfg = read(case_dir/'summary.json'), read(case_dir/'config.json')
        case = read(case_dir/'case_study.json')
        forecasts = case['obstacle_forecast_metrics']
        diagnoses = case['failure_diagnostics']
        categories = diagnoses['categories']
        row = {'study':'budget', 'case':record['name'], 'scenario':summary['scenario'],
               'budget':cfg['candidate_budget'], 'seed':cfg['seed'],
               'command_delay_s':cfg['command_delay_s'], 'control_dt_s':cfg['control_dt'],
               'sensor_hfov_deg':cfg['sensor_hfov_deg'], 'sensor_vfov_deg':cfg['sensor_vfov_deg'],
               'collision':'collision' in summary['status'], 'timeout':summary['status']=='timeout',
               **{key:summary.get(key) for key in METRICS},
               'ca_obstacle_rmse_2s_m':forecasts['at_2s']['rmse_m'],
               'ca_obstacle_rmse_24s_m':forecasts['at_24s']['rmse_m'],
               'ca_obstacle_max_error_24s_m':forecasts['at_24s']['max_error_m'],
               'offline_shortlist_omission_frames':categories.get('shortlist_omission',0),
               'offline_no_safe_grid_reference_frames':categories.get('no_safe_reference_in_current_discrete_grid',0)}
        rows.append(row)
        configs[record['name']] = cfg
    for record in sens['rows']:
        name = record['case']
        summary, cfg = read(sensitivity/name/'summary.json'), read(sensitivity/name/'config.json')
        rows.append({'study':'sensitivity','case':name,'scenario':summary['scenario'],
                     'budget':cfg['candidate_budget'],'seed':cfg['seed'],
                     'command_delay_s':cfg['command_delay_s'],'control_dt_s':cfg['control_dt'],
                     'sensor_hfov_deg':cfg['sensor_hfov_deg'],'sensor_vfov_deg':cfg['sensor_vfov_deg'],
                     'collision':'collision' in summary['status'],'timeout':summary['status']=='timeout',
                     **{key:summary.get(key) for key in METRICS}})
        configs[name] = cfg

    out.mkdir(parents=True)
    for name in ('TEST_REPORT.md','CHANGELOG.md','FAILURE_ANALYSIS.md','TEST_SCOPE.md'):
        if (documents/name).exists():
            shutil.copy2(documents/name,out/name)
    if (documents/'figures').is_dir():
        shutil.copytree(documents/'figures',out/'figures')
    columns = list(rows[0])
    with (out/'EXPERIMENT_SUMMARY.csv').open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    configuration = {
        'created_at':datetime.now().astimezone().isoformat(),
        'repository':'Whsjbwy/AUVWorkPlace','branch':'research/stage2-baseline',
        'physics_dt_s':.02,'horizon_s':24.,'seed':7,'safety_margin_m':1.,
        'default_control_dt_s':2.,'default_budget':9,'compared_budgets':[9,18,45],
        'vehicle':'unchanged original REMUS100; complete controller/actuator forward validation',
        'obstacle_prediction':'CA; no PPLNM6DOF or ECP',
        'timing':'default synchronous; explicit fixed delay sensitivity is not measured wall-clock scheduling',
        'latency_note':study['latency_note'],'per_case_configurations':configs,
    }
    (out/'EXPERIMENT_CONFIG.json').write_text(json.dumps(configuration,indent=2,ensure_ascii=False),encoding='utf-8')
    # Copy complete current-study outputs, including every failed state and all
    # 45-candidate diagnosis records. Original source directories remain intact.
    shutil.copytree(budget,out/'raw'/'budget')
    shutil.copytree(sensitivity,out/'raw'/'sensitivity')
    for source, label in ((regression,'regression'),(final_tests,'final_tests')):
        target = out/'raw'/label
        target.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file():
                shutil.copy2(path,target/path.name)
            elif path.name not in ('pytest_work','pytest_temp'):
                shutil.copytree(path,target/path.name)
    for source in map(Path,extras):
        shutil.copytree(source,out/'raw'/'additional'/source.name,
                        ignore=shutil.ignore_patterns('work','pytest_work','pytest_temp',
                                                     '__pycache__','.pytest_cache'))
    if (documents/'quality_checks.json').is_file():
        shutil.copy2(documents/'quality_checks.json',out/'quality_checks.json')
    (out/'README.md').write_text(
        '# REMUS阶段2复审实验交付\n\n'
        '先读TEST_REPORT、FAILURE_ANALYSIS和EXPERIMENT_SUMMARY。raw保留本轮全部预算/敏感性'
        '数值轨迹、控制、观测、应急、候选诊断、CA误差、配置、命令日志与图片；'
        '失败记录没有删除。原数据目录同时留在项目results_stage2中。\n\n'
        'CPU共享负载下规划耗时不代表隔离实时能力。到达不等于安全验收。'
        'CA未来真值只用于执行后的离线分析，未反馈在线规划。没有ECP、PPLNM六自由度或电能节省声明。\n',
        encoding='utf-8')
    files = sorted(path for path in out.rglob('*') if path.is_file())
    inventory = {'inventory_scope':'payload files excluding INVENTORY.json itself',
                 'file_count':len(files),'uncompressed_bytes':sum(path.stat().st_size for path in files),
                 'files':[{'path':path.relative_to(out).as_posix(),'bytes':path.stat().st_size} for path in files],
                 'packaging':'ordinary ZIP; no extra digest or checksum validation'}
    (out/'INVENTORY.json').write_text(json.dumps(inventory,indent=2,ensure_ascii=False),encoding='utf-8')
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as stream:
        for path in sorted(out.rglob('*')):
            if path.is_file():
                stream.write(path,Path(out.name)/path.relative_to(out))
    return {'delivery_directory':str(out.resolve()),'archive':str(archive.resolve()),
            'archive_bytes':archive.stat().st_size,'experiment_count':len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('budget','sensitivity','regression','final-tests','documents','out','archive'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--extra',nargs='*',type=Path,default=[])
    args = parser.parse_args()
    print(json.dumps(assemble(args.budget,args.sensitivity,args.regression,args.final_tests,
                             args.documents,args.out,args.archive,args.extra),indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
