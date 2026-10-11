"""Run the actual REMUS acceptance commands serially and preserve their outputs.

Use the stage 2 virtual environment to invoke this script. A new output root is
required: previous experiments are never overwritten. Task completion, Python
exit status and navigation success are separate recorded values.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CLIENT_ZONE = timezone(timedelta(hours=8))


def now():
    return datetime.now(CLIENT_ZONE).isoformat(timespec='seconds')


def acceptance_cases():
    cases = []
    for scenario in ('empty3d', 'static3d', 'crossing3d', 'head_on3d',
                     'multi_crossing3d', 'maneuvering3d', 'terrain3d',
                     'tracking_mismatch3d', 'paper_random3d'):
        cases.append(dict(name=scenario, scenario=scenario,
                          max_time=200 if scenario == 'paper_random3d' else 150,
                          animate=scenario == 'crossing3d', extra=[]))
    cases.extend([
        dict(name='crossing_current03_dir90', scenario='crossing3d', max_time=150,
             animate=False, extra=['--current-speed', '.3', '--current-direction-deg', '90']),
        dict(name='multi_crossing_speed15', scenario='multi_crossing3d', max_time=150,
             animate=False, extra=['--obstacle-speed-scale', '1.5']),
        dict(name='maneuvering_turn8_speed14', scenario='maneuvering3d', max_time=150,
             animate=False, extra=['--obstacle-turn-rate-deg-s', '8', '--obstacle-speed-scale', '1.4']),
        dict(name='crossing_range60_delay1', scenario='crossing3d', max_time=150,
             animate=False, extra=['--sensor-range-m', '60', '--observation-delay-s', '1']),
        dict(name='crossing_range35_delay2', scenario='crossing3d', max_time=150,
             animate=False, extra=['--sensor-range-m', '35', '--observation-delay-s', '2']),
    ])
    return cases


def run_case(case, out):
    case_out = out / case['name']
    command = [sys.executable, str(ROOT / 'run_stage2.py'), 'simulate',
               '--backend', 'remus6dof', '--scenario', case['scenario'],
               '--config', str(ROOT / 'configs' / 'stage2_development.json'),
               '--horizon-s', '24', '--control-dt', '2', '--candidate-budget', '9',
               '--max-time', str(case['max_time']), '--out', str(case_out)]
    if not case['animate']:
        command.append('--no-animation')
    command.extend(case['extra'])
    started = now()
    tic = time.perf_counter()
    log_path = out / 'logs' / (case['name'] + '.log')
    with log_path.open('x', encoding='utf-8') as log:
        log.write('COMMAND ' + subprocess.list2cmdline(command) + '\n')
        log.write('STARTED ' + started + '\n\n')
        log.flush()
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, 'MPLBACKEND': 'Agg',
                                        'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}, check=False)
        wall_time = time.perf_counter() - tic
        log.write('\nFINISHED ' + now() + '\n')
        log.write(f'EXIT_CODE {completed.returncode}\nWALL_CLOCK_S {wall_time:.6f}\n')
    summary_path = case_out / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else None
    row = {
        'name': case['name'], 'scenario': case['scenario'], 'command_argv': command,
        'command_powershell_display': subprocess.list2cmdline(command),
        'started_at': started, 'finished_at': now(), 'wall_clock_s': wall_time,
        'exit_code': completed.returncode, 'output_directory': str(case_out),
        'command_log': str(log_path), 'summary_available': summary is not None,
        'status': summary.get('status') if summary else 'process_error',
        'navigation_success': bool(summary and summary.get('reached_goal')),
        'summary': summary,
        'visualization_outputs': [str(path) for path in sorted(case_out.glob('*'))
                                  if path.suffix.lower() in ('.png', '.gif', '.html')],
    }
    if completed.returncode and summary:
        row['process_limitation'] = 'simulation summary exists but a later output step failed; inspect command log'
    return row


def analyze_existing(out):
    """Add a separate classification/media report without changing raw runs."""
    from PIL import Image

    manifest = json.loads((out / 'acceptance_results.json').read_text(encoding='utf-8'))
    if manifest.get('finished_at') is None:
        raise RuntimeError('acceptance batch is still running')
    analysis_path = out / 'acceptance_analysis.json'
    if analysis_path.exists():
        raise FileExistsError('acceptance analysis already exists; raw experiments are preserved')
    rows, media = [], []
    for record in manifest['records']:
        summary = record.get('summary') or {}
        arrived = bool(summary.get('reached_goal'))
        unsafe = int(summary.get('unsafe_emergency_forecast_count', 0))
        collision = 'collision' in summary.get('status', '')
        # Older early-batch summaries predate the explicit classification field.
        # Derive exactly the same criterion without changing their raw JSON.
        safety_pass = arrived and not collision and unsafe == 0
        rows.append({
            'name': record['name'], 'status': summary.get('status'),
            'navigation_success': arrived, 'avoidance_acceptance_passed': safety_pass,
            'unsafe_emergency_forecast_count': unsafe,
            'minimum_model_clearance_m': summary.get('minimum_model_clearance_m'),
            'duration_s': summary.get('duration_s'),
            'actual_close_interaction_count': summary.get('actual_close_interaction_count'),
            'maximum_simultaneous_close_interactions': summary.get('maximum_simultaneous_close_interactions'),
            'maximum_observed_obstacles': summary.get('maximum_observed_obstacles'),
            'maximum_tracked_obstacles': summary.get('maximum_tracked_obstacles'),
            'emergency_count': summary.get('emergency_count'),
            'planning_ms_p95': summary.get('planning_ms_p95'),
            'planning_exceeds_control_period_count': summary.get('planning_exceeds_control_period_count'),
            'prediction_position_rmse_m': summary.get('prediction_position_rmse_m'),
            'prediction_position_max_error_m': summary.get('prediction_position_max_error_m'),
            'process_exit_code': record['exit_code'],
        })
        for path_string in record['visualization_outputs']:
            path = Path(path_string)
            if path.suffix.lower() not in ('.png', '.gif'):
                continue
            with Image.open(path) as picture:
                media.append({'case': record['name'], 'path': str(path),
                              'format': picture.format, 'width': picture.width,
                              'height': picture.height, 'frames': getattr(picture, 'n_frames', 1)})
    report = {
        'created_at': now(), 'raw_manifest': str(out / 'acceptance_results.json'),
        'classification_criterion': 'goal reached, no actual collision, and zero unsafe emergency forecasts; this is not a real-world safety guarantee',
        'prediction_error_scope': 'selected reference held until replanning in the same deterministic original REMUS plant; zero numerical error is not robust execution-error validation',
        'case_count': len(rows), 'navigation_success_count': sum(row['navigation_success'] for row in rows),
        'avoidance_acceptance_passed_count': sum(row['avoidance_acceptance_passed'] for row in rows),
        'actual_collision_count': sum('collision' in (row['status'] or '') for row in rows),
        'process_failure_count': sum(row['process_exit_code'] != 0 for row in rows),
        'cases': rows, 'media_metadata': media,
    }
    with analysis_path.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f'Acceptance analysis: {analysis_path}', flush=True)
    return report


def normalize_existing_logs(out):
    """Create readable UTF-8 copies of earlier Windows command logs.

    The first acceptance batch used UTF-8 parent headers and Windows-default
    child stdout. Preserve those original byte streams and convert only copies.
    New subprocesses explicitly use UTF-8 and no longer have mixed encodings.
    """
    target_root = out / 'logs_utf8'
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError('UTF-8 log copies already exist; choose a new experiment root')
    target_root.mkdir(exist_ok=True)
    converted = []
    for source in sorted((out / 'logs').glob('*.log')):
        raw = source.read_bytes()
        header_separator = b'\r\n\r\n' if b'\r\n\r\n' in raw else b'\n\n'
        header, separator, child_output = raw.partition(header_separator)
        if not separator:
            raise ValueError(f'command log header is missing: {source}')
        try:
            body = child_output.decode('utf-8')
            original_encoding = 'utf-8'
        except UnicodeDecodeError:
            body = child_output.decode('gb18030')
            original_encoding = 'Windows child stdout GBK/GB18030, UTF-8 parent header'
        target = target_root / source.name
        with target.open('x', encoding='utf-8') as handle:
            handle.write(header.decode('utf-8').replace('\r\n', '\n') + '\n\n' + body.replace('\r\n', '\n'))
        converted.append({'original': str(source), 'utf8_copy': str(target),
                          'original_encoding': original_encoding})
    with (target_root / 'encoding_notes.json').open('x', encoding='utf-8') as handle:
        json.dump({'created_at': now(), 'raw_logs_modified': False, 'logs': converted},
                  handle, indent=2, ensure_ascii=False)
    print(f'UTF-8 log copies: {target_root}', flush=True)
    return converted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results_stage2/acceptance_20261011')
    parser.add_argument('--cases', nargs='+', help='Optional exact case names for a separate new experiment root')
    parser.add_argument('--analyze-existing', action='store_true',
                        help='Create a separate derived report for a completed batch; never change its raw outputs')
    parser.add_argument('--normalize-existing-logs', action='store_true',
                        help='Create UTF-8 copies of completed Windows command logs, retaining originals')
    args = parser.parse_args(argv)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out = out.resolve()
    if args.normalize_existing_logs:
        normalize_existing_logs(out)
        return 0
    if args.analyze_existing:
        analyze_existing(out)
        return 0
    if out.exists():
        parser.error('acceptance output root already exists; choose a new directory')
    cases = acceptance_cases()
    if args.cases:
        known = {case['name'] for case in cases}
        if not set(args.cases) <= known:
            parser.error('unknown acceptance case name')
        cases = [case for case in cases if case['name'] in args.cases]
    out.mkdir(parents=True, exist_ok=False)
    (out / 'logs').mkdir()
    manifest = {
        'started_at': now(), 'finished_at': None, 'python_executable': sys.executable,
        'python_version': sys.version, 'repository': str(ROOT),
        'execution': 'serial real Python subprocesses; process exit is distinct from reaching the goal',
        'configuration_contract': {'backend': 'remus6dof', 'physics_dt_s': .02,
                                   'control_dt_s': 2., 'horizon_s': 24., 'candidate_budget': 9,
                                   'original_hydrodynamic_parameters_modified': False},
        'case_count': len(cases), 'completed_count': 0, 'records': [],
    }
    manifest_path = out / 'acceptance_results.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] START {case['name']} {now()}", flush=True)
        row = run_case(case, out)
        manifest['records'].append(row)
        manifest['completed_count'] = index
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
        summary = row['summary'] or {}
        print(f"[{index}/{len(cases)}] DONE {case['name']} status={row['status']} "
              f"exit={row['exit_code']} gap={summary.get('minimum_model_clearance_m')} "
              f"p95_ms={summary.get('planning_ms_p95')} wall_s={row['wall_clock_s']:.2f}", flush=True)
    manifest['finished_at'] = now()
    manifest['success_count'] = sum(record['navigation_success'] for record in manifest['records'])
    manifest['process_failure_count'] = sum(record['exit_code'] != 0 for record in manifest['records'])
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Acceptance records: {manifest_path}', flush=True)
    analyze_existing(out)
    return 1 if manifest['process_failure_count'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
