"""Run registered, paired CA risk pilots on the original REMUS six-DOF plant."""
from __future__ import annotations

import argparse
from datetime import datetime
import json


def main(argv=None):
    from auv3d.stage4_experiments import METHODS, SPLITS, run_study
    from auv3d.stage4_protocol import manifest, protocol_record
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='task', required=True)
    sub.add_parser('manifest', help='Print complete preregistered scenarios and split protocol')
    calibration = sub.add_parser('build-calibration', help='Build frozen calibration scene blocks from completed B0 episodes')
    calibration.add_argument('--baseline-root', required=True)
    calibration.add_argument('--out', required=True)
    analyze = sub.add_parser('analyze', help='Offline matched-clock forecast and same-controller risk diagnostics')
    analyze.add_argument('--root', nargs='+', required=True)
    analyze.add_argument('--library', required=True)
    analyze.add_argument('--out', required=True)
    analyze.add_argument('--baseline-root')
    analyze.add_argument('--alpha', nargs='+', type=float, default=[.1])
    analyze.add_argument('--anchors', nargs='+', type=float, default=[20., 40., 60.])
    analyze.add_argument('--shadow-case', nargs='+', dest='shadow_case_ids')
    for task in ('run-baseline', 'run-risk', 'run-paired'):
        run = sub.add_parser(task)
        run.add_argument('--out')
        run.add_argument('--split', nargs='+', choices=SPLITS)
        run.add_argument('--case', nargs='+', dest='case_ids')
        run.add_argument('--limit', type=int)
        run.add_argument('--workers', type=int, default=1)
        run.add_argument('--no-plot', action='store_true')
        run.add_argument('--animation', action='store_true')
        if task in ('run-risk', 'run-paired'):
            run.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
            run.add_argument('--alpha', '--alphas', dest='alphas', nargs='+', type=float, default=[.1])
            run.add_argument('--library', '--calibration', dest='calibration', required=True)
            run.add_argument('--baseline-root')
            if task == 'run-paired':
                run.set_defaults(methods=['B1', 'B2'])
    args = parser.parse_args(argv)
    if args.task == 'manifest':
        print(json.dumps({'protocol': protocol_record(), 'cases': manifest()}, indent=2,
                         ensure_ascii=False))
        return 0
    if args.task == 'build-calibration':
        from auv3d.stage4_calibration import build_library
        rows = [dict(row, method='B0') for row in manifest()]
        outcome = build_library(args.baseline_root, rows, args.out)
        print(json.dumps(outcome, indent=2, ensure_ascii=False, default=str))
        return 0
    if args.task == 'analyze':
        from auv3d.stage4_analysis import analyze_studies
        outcome = analyze_studies(args.root, args.library, args.out,
                                  baseline_root=args.baseline_root, alphas=args.alpha,
                                  anchors=args.anchors, shadow_case_ids=args.shadow_case_ids)
        print(json.dumps(outcome, indent=2, ensure_ascii=False))
        return 0
    output = args.out or f'results_stage4/{args.task}_{datetime.now():%Y%m%d_%H%M%S}'
    kwargs = (dict(methods=args.methods, alphas=args.alphas, calibration_path=args.calibration,
                   baseline_root=args.baseline_root)
              if args.task in ('run-risk', 'run-paired') else {})
    outcome = run_study(output, splits=args.split, case_ids=args.case_ids, limit=args.limit,
                        workers=args.workers, plot=not args.no_plot, animate=args.animation,
                        **kwargs)
    print(json.dumps(outcome, indent=2, ensure_ascii=False))
    return 2 if outcome['program_error_count'] else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f'ERROR: {error}')
        raise SystemExit(2)
