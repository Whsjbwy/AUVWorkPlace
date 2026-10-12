"""Registered A0-A5 REMUS risk validation with independent artifacts."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path


def main(argv=None):
    # Set before importing NumPy, BLAS, forecasts or the simulator.
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    from auv3d.stage5_experiments import (METHODS,PREDICTORS,COMBINATIONS,SPLITS,PHASES,
                                         DELAY_MODES,load_artifact_registry,run_study)
    from auv3d.stage5_protocol import manifest,protocol_record
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='task',required=True)
    registration = sub.add_parser('manifest')
    registration.add_argument('--phase',choices=PHASES,default='pilot')
    calibration = sub.add_parser('build-calibration')
    calibration.add_argument('--phase',choices=PHASES,default='pilot')
    calibration.add_argument('--predictors',nargs='+',choices=PREDICTORS,default=list(PREDICTORS))
    calibration.add_argument('--noise',nargs='+',type=float,default=[0.,.02,.1])
    calibration.add_argument('--out',required=True)
    calibration.add_argument('--obstacle-only',action='store_true',
        help='Use the fixed four obstacle families; always enabled for formal calibration')
    freeze = sub.add_parser('freeze-margin')
    freeze.add_argument('--root',nargs='+',required=True)
    freeze.add_argument('--out',required=True)
    analysis = sub.add_parser('analyze')
    analysis.add_argument('--root',nargs='+',required=True)
    analysis.add_argument('--out',required=True)
    analysis.add_argument('--anchors',nargs='+',type=float,default=[20.,40.,60.])
    analysis.add_argument('--shadow',action='store_true',help='Exact-controller same-state comparisons; schedule without concurrent CPU jobs')
    for task in ('run-pilot','run-study','tune-margin'):
        run = sub.add_parser(task)
        run.add_argument('--phase',choices=PHASES,default='pilot')
        run.add_argument('--out')
        run.add_argument('--split',nargs='+',choices=SPLITS,default=['test'] if task != 'tune-margin' else ['tune'])
        run.add_argument('--case',nargs='+',dest='case_ids')
        run.add_argument('--noise',type=float,default=.02)
        run.add_argument('--all-noise',action='store_true')
        run.add_argument('--limit',type=int)
        run.add_argument('--predictors',nargs='+',choices=PREDICTORS,default=['ca'])
        run.add_argument('--combination',nargs='+',choices=COMBINATIONS,default=['add'],dest='combinations')
        run.add_argument('--delay-mode',choices=DELAY_MODES,default='none',dest='planning_delay_mode')
        run.add_argument('--delay-s',type=float,default=0.,dest='planning_delay_s')
        run.add_argument('--workers',type=int,default=1)
        run.add_argument('--no-plot',action='store_true')
        run.add_argument('--animation',action='store_true')
        if task == 'tune-margin':
            run.add_argument('--margins',nargs='+',type=float,required=True)
        else:
            run.add_argument('--methods',nargs='+',choices=METHODS,default=['A0'])
            run.add_argument('--alpha',nargs='+',type=float,dest='alphas')
            run.add_argument('--artifacts',help='JSON predictor/noise calibration registry')
            run.add_argument('--tuned-margin',dest='tuned_margin_path')
    args = parser.parse_args(argv)
    if args.task == 'manifest':
        print(json.dumps(dict(protocol=protocol_record(args.phase),cases=manifest(args.phase)),indent=2,ensure_ascii=False))
        return 0
    if args.task == 'build-calibration':
        from auv3d.stage5_calibration import build_libraries,risk_calibration_rows
        from auv3d.stage4_experiments import write_json
        output = Path(args.out).resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('existing calibration suite is preserved')
        if len(set(args.noise)) != len(args.noise):
            raise ValueError('noise levels must be unique')
        rows = [row for noise in args.noise for row in manifest(args.phase,split='calibration',noise_std=noise)]
        registered_rows = rows
        obstacle_only = args.obstacle_only or args.phase == 'formal'
        if obstacle_only:
            rows = risk_calibration_rows(rows)
        result = build_libraries(rows,output,predictors=args.predictors)
        output.mkdir(parents=True,exist_ok=True)
        registry = dict(schema='stage5_artifact_registry_v1',phase=args.phase,
            calibration_family_scope='four predeclared obstacle families; all empty terrain controls excluded by family, never by seed/outcome' if obstacle_only else 'complete registered pool including explicit empty controls',
            registered_calibration_geometry_count=len({row['scene_id'] for row in registered_rows}),
            included_calibration_geometry_count=len({row['scene_id'] for row in rows}),
            calibration_libraries=[dict(predictor=row['predictor'],noise_std_m=row['noise_std_m'],path=str(Path(row['path']).resolve())) for row in result])
        write_json(output/'ARTIFACT_REGISTRY.json',registry)
        write_json(output/'CALIBRATION_REPORT.json',result)
        print(json.dumps(registry,indent=2,ensure_ascii=False))
        return 0
    if args.task == 'freeze-margin':
        from auv3d.stage5_analysis import freeze_fixed_margin
        print(json.dumps(freeze_fixed_margin(args.root,args.out),indent=2,ensure_ascii=False))
        return 0
    if args.task == 'analyze':
        from auv3d.stage5_analysis import analyze_studies
        print(json.dumps(analyze_studies(args.root,args.out,anchors=args.anchors,shadow=args.shadow),indent=2,ensure_ascii=False))
        return 0
    if args.task == 'run-pilot' and args.phase != 'pilot':
        raise ValueError('run-pilot cannot claim a formal study')
    output = args.out or f'results_stage5/{args.task}_{datetime.now():%Y%m%d_%H%M%S}'
    options = dict(phase=args.phase,splits=args.split,case_ids=args.case_ids,
        noise_std=None if args.all_noise else args.noise,limit=args.limit,
        predictors=args.predictors,combinations=args.combinations,
        planning_delay_mode=args.planning_delay_mode,planning_delay_s=args.planning_delay_s,
        workers=args.workers,plot=not args.no_plot,animate=args.animation)
    if args.task == 'tune-margin':
        options.update(methods=('A1',),fixed_margins=args.margins)
    else:
        options.update(methods=args.methods,alphas=args.alphas,
            calibration_paths=load_artifact_registry(args.artifacts) if args.artifacts else None,
            tuned_margin_path=args.tuned_margin_path)
    result = run_study(output,**options)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 2 if result['program_error_count'] else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError,RuntimeError,FileNotFoundError,FileExistsError) as error:
        print(f'ERROR: {error}')
        raise SystemExit(2)
