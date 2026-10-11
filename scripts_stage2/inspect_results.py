"""Recompute timing/log/actuator metrics and decode saved REMUS figures."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def table(path):
    return np.genfromtxt(path, delimiter=',', names=True, ndmin=1)


def inspect_case(path):
    summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
    cfg=json.loads((path/'config.json').read_text(encoding='utf-8'))
    states=table(path/'states.csv')
    controls=table(path/'controls.csv')
    planning=table(path/'planning.csv')
    errors=table(path/'prediction_errors.csv')
    plans=json.loads((path/'selected_plans.json').read_text(encoding='utf-8'))
    snapshots=json.loads((path/'obstacles.json').read_text(encoding='utf-8'))
    checks={}
    checks['finite_logs']=all(np.isfinite(data[name]).all() for data in (states,controls,planning,errors) for name in data.dtype.names)
    checks['physics_timing']=bool(np.allclose(np.diff(states['time_s']),cfg['physics_dt'],atol=1e-10,rtol=0))
    checks['control_interval_timing']=len(controls)==len(states)-1 and np.allclose(controls['time_s'],states['time_s'][:-1],atol=1e-10,rtol=0)
    checks['actuator_end_samples_align']=len(controls)==len(states)-1 and all(np.allclose(controls[name],states[name][1:],atol=1e-10,rtol=0) for name in ('actual_rudder_rad','actual_stern_rad','actual_rpm'))
    limits={'rudder':np.deg2rad(15),'stern':np.deg2rad(15),'rpm':1525.}
    checks['original_actuator_envelopes']=all(np.all(np.abs(controls['applied_'+key+('' if key=='rpm' else '_rad')])<=limit+1e-10) and np.all(np.abs(states['actual_'+key+('' if key=='rpm' else '_rad')])<=limit+1e-10) for key,limit in limits.items())
    checks['reference_envelopes']=bool(np.all((controls['ref_depth_m']>=0)&(controls['ref_depth_m']<=100)) and np.all((controls['ref_rpm']>=0)&(controls['ref_rpm']<=1525)))
    expected=np.arange(0,summary['duration_s']-1e-9,cfg['control_dt'])
    checks['planning_timing']=len(expected)==len(planning) and np.allclose(expected,planning['time_s'],atol=1e-10,rtol=0)
    checks['fixed_candidate_budget']=bool(np.all(planning['verified_candidates']<=cfg['candidate_budget']))
    checks['prediction_starts_at_actual_state']=all(np.allclose(np.asarray(plan['states'])[0,1:7],[states[name][round(plan['time_s']/cfg['physics_dt'])] for name in ('north_m','east_m','depth_m','roll_rad','pitch_rad','yaw_rad')],atol=1e-10,rtol=0) for plan in plans)
    checks['p95_recomputed']=bool(np.isclose(np.percentile(planning['planning_ms'],95),summary['planning_ms_p95'])) if len(planning) else True
    checks['prediction_rmse_recomputed']=bool(np.isclose(np.sqrt(np.mean(errors['position_error_m']**2)),summary['prediction_position_rmse_m'])) if len(errors) else True
    checks['arrival_and_collision_status_honest']=bool((not summary['reached_goal'] or summary['goal_distance_m']<=cfg['goal_tolerance_m']+1e-9) and ('collision' not in summary['status'] or summary['minimum_model_clearance_m']<=0))
    media=[]
    for file in sorted(path.iterdir()):
        if file.suffix.lower() in ('.png','.gif'):
            with Image.open(file) as picture:
                frames=getattr(picture,'n_frames',1)
                picture.seek(frames-1)
                picture.load()
                media.append({'name':file.name,'format':picture.format,'frames':frames,'size':picture.size})
    checks['images_decoded']=bool(media)
    terminal_present=bool(snapshots and abs(snapshots[-1]['time_s']-summary['duration_s'])<1e-9)
    # Older formal runs are preserved. Their display-only missing final snapshot
    # is reported separately; render_saved.py can repair it in a NEW directory.
    return {'case':path.name,'checks':{key:bool(value) for key,value in checks.items()},
            'all_numeric_media_checks_passed':all(checks.values()),'media':media,
            'terminal_truth_snapshot_present':terminal_present,
            'navigation_status':summary['status'],
            'derived_avoidance_acceptance_passed':summary['status']=='success' and summary['unsafe_emergency_forecast_count']==0 and summary['minimum_model_clearance_m']>=cfg['safety_margin_m']-1e-9,
            'unsafe_emergency_forecast_count':summary['unsafe_emergency_forecast_count']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        parser.error('inspection report already exists; choose a new filename')
    paths=sorted(p.parent for p in args.root.glob('*/summary.json') if (p.parent/'states.csv').exists())
    records=[inspect_case(path) for path in paths]
    report={'case_count':len(records),'all_passed':bool(records) and all(r['all_numeric_media_checks_passed'] for r in records),'records':records}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'case_count':len(records),'all_passed':report['all_passed']},indent=2))
    return 0 if report['all_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
