"""Common REMUS six-DOF baseline: explicit memory, candidates and terrain switches."""
import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path

from auv3d.scenarios import STAGE2_SCENARIOS
from auv3d.stage2_config import Stage2Config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='task', required=True)
    run = sub.add_parser('simulate')
    run.add_argument('--backend', choices=['remus6dof'], default='remus6dof')
    run.add_argument('--scenario', choices=STAGE2_SCENARIOS, default='crossing3d')
    run.add_argument('--config')
    run.add_argument('--out')
    run.add_argument('--no-animation', action='store_true')
    run.add_argument('--no-plot', action='store_true')
    for name, choices in {
        'memory-policy': ['hard_expiry', 'retain_uncertain', 'conditional'],
        'candidate-scheme': ['baseline45', 'expanded'],
        'shortlist-policy': ['diverse', 'ranked', 'coverage'],
        'emergency-policy': ['coast', 'verified_backup'],
        'terrain-guidance': ['direct', 'coarse3d_astar'],
        'obstacle-predictor': ['ca', 'pplnm'],
        'pplnm-extension-mode': ['cv_after9', 'ca_after9', 'reject'],
    }.items():
        run.add_argument('--'+name, choices=choices)
    for name in ('max-time-s', 'horizon-s', 'control-dt', 'command-delay-s',
                 'track-max-age-s', 'memory-uncertainty-growth-mps',
                 'sensor-hfov-deg', 'sensor-vfov-deg', 'observation-period-s',
                 'observation-delay-s', 'current-speed', 'current-direction-deg',
                 'obstacle-speed-scale', 'obstacle-turn-rate-deg-s',
                 'prediction-history-dt-s'):
        run.add_argument('--'+name, type=float)
    for name in ('candidate-budget', 'seed'):
        run.add_argument('--'+name, type=int)
    run.add_argument('--pplnm-checkpoint')
    run.add_argument('--pplnm-calibration')
    run.add_argument('--pplnm-threads', type=int)
    args = parser.parse_args(argv)
    cfg = Stage2Config.load(args.config) if args.config else Stage2Config()
    ignored = {'task', 'backend', 'scenario', 'config', 'out', 'no_animation', 'no_plot'}
    cfg = replace(cfg, **{k: v for k, v in vars(args).items()
                         if k not in ignored and v is not None})
    cfg.validate()
    out = Path(args.out or f'results_stage3/{args.scenario}_{datetime.now():%Y%m%d_%H%M%S}')
    if out.exists() and any(out.iterdir()):
        parser.error('choose a new output directory; existing experiments are preserved')
    from auv3d.stage2_simulation import run_remus_episode, save_remus_episode
    result = run_remus_episode(cfg, args.scenario)
    save_remus_episode(result, out, animate=not args.no_animation, plot=not args.no_plot)
    print(json.dumps(result['summary'], ensure_ascii=False, indent=2))
    print(f'Results: {out.resolve()}')
    return 0  # Program completion is separate from navigation/safety outcomes.


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, FloatingPointError) as error:
        print(f'ERROR: {error}')
        raise SystemExit(2)
