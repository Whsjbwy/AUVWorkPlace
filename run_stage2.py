"""Original REMUS six-DOF closed-loop reference-space DWA baseline."""
from dataclasses import replace
from datetime import datetime
import argparse
import json
from pathlib import Path

from auv3d.stage2_config import Stage2Config
from auv3d.scenarios import STAGE2_SCENARIOS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='task',required=True)
    run = sub.add_parser('simulate')
    run.add_argument('--backend',choices=['remus6dof','p25'],default='remus6dof')
    run.add_argument('--scenario',choices=STAGE2_SCENARIOS,default='crossing3d')
    run.add_argument('--config')
    run.add_argument('--out')
    run.add_argument('--no-animation',action='store_true')
    run.add_argument('--no-plot',action='store_true')
    for option in ('max-time','current-speed','current-direction-deg','sensor-range-m',
                   'observation-delay-s','obstacle-speed-scale','obstacle-turn-rate-deg-s',
                   'horizon-s','control-dt','warmup-s','command-delay-s',
                   'sensor-hfov-deg','sensor-vfov-deg'):
        run.add_argument('--'+option,type=float)
    run.add_argument('--candidate-budget',type=int)
    run.add_argument('--seed',type=int)
    response = sub.add_parser('responses')
    response.add_argument('--out')
    diagnostic = sub.add_parser('diagnose')
    diagnostic.add_argument('--out')
    args = parser.parse_args(argv)
    out = Path(args.out or f'results_stage2/{args.task}_{datetime.now():%Y%m%d_%H%M%S}')
    if out.exists() and any(out.iterdir()):
        parser.error('output directory must be new or empty; existing experiments are preserved')
    if args.task == 'simulate':
        cfg = Stage2Config.load(args.config) if args.config else Stage2Config()
        names = {'max_time':'max_time_s'}
        changes = {names.get(key,key):value for key,value in vars(args).items()
                   if key in ('max_time','current_speed','current_direction_deg','sensor_range_m',
                              'observation_delay_s','obstacle_speed_scale','obstacle_turn_rate_deg_s',
                              'horizon_s','control_dt','warmup_s','candidate_budget','seed',
                              'command_delay_s','sensor_hfov_deg','sensor_vfov_deg') and value is not None}
        cfg = replace(cfg,**changes)
        cfg.validate()
        if args.backend == 'remus6dof':
            from auv3d.stage2_simulation import run_remus_episode,save_remus_episode
            result = run_remus_episode(cfg,args.scenario)
            save_remus_episode(result,out,animate=not args.no_animation,plot=not args.no_plot)
        else:
            from auv3d.stage2_p25 import run_p25_episode
            from auv3d.simulation import save_episode
            result = run_p25_episode(cfg,args.scenario)
            save_episode(result,out,plot=not args.no_plot)
            (out/'events.json').write_text(json.dumps(result['events'],indent=2),encoding='utf-8')
        summary = result['summary']
    else:
        from auv3d.stage2_diagnostics import run_responses,run_tracking_diagnostic
        summary = run_responses(out) if args.task=='responses' else run_tracking_diagnostic(out)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print(f'Results: {out}')
    # Completion and success are deliberately separate in summary.json.
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,RuntimeError,FileExistsError,FileNotFoundError,FloatingPointError) as error:
        print(f'ERROR: {error}')
        raise SystemExit(2)
