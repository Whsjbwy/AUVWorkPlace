"""Run and record Windows/CPU acceptance checks without overwriting old results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    out = (args.out or ROOT / 'results_stage1' / f'acceptance_{stamp}').resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('output must be a new or empty directory')
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUTF8='1')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT), str(ROOT / 'PythonVehicleSimulator' / 'src')])
    records = []
    report = {'started_at': datetime.now().astimezone().isoformat(),
              'python': sys.version, 'platform': platform.platform(),
              'executable': sys.executable, 'output': str(out), 'checks': records}

    def record(name, arguments, cwd=ROOT, summary=None, expect_success=False):
        command = [sys.executable, *map(str, arguments)]
        logfile = out / f'{name}.log'
        print(f'RUN {name}: {subprocess.list2cmdline(command)}', flush=True)
        started = time.perf_counter()
        with logfile.open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = {'name': name, 'command': subprocess.list2cmdline(command), 'cwd': str(cwd),
               'exit_code': completed.returncode, 'wall_clock_s': time.perf_counter() - started,
               'log': str(logfile), 'passed': completed.returncode == 0}
        if summary and Path(summary).exists():
            row['summary'] = json.loads(Path(summary).read_text(encoding='utf-8'))
            if expect_success:
                row['passed'] &= row['summary'].get('status') == 'success'
        if name == 'pytest' and (out / 'pytest.xml').exists():
            suites = ET.parse(out / 'pytest.xml').getroot()
            row['junit'] = {key: sum(int(s.attrib.get(key, 0)) for s in suites.iter('testsuite'))
                            for key in ('tests', 'failures', 'errors', 'skipped')}
            row['passed'] &= row['junit']['skipped'] == 0
        records.append(row)
        report['all_passed'] = all(r['passed'] for r in records)
        (out / 'acceptance_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"{'PASS' if row['passed'] else 'FAIL'} {name} ({row['wall_clock_s']:.1f}s)", flush=True)

    pytest_work = out / 'pytest_work'
    pytest_work.mkdir()
    record('versions', ['-m', 'pip', 'list', '--format=json'])
    record('pip_check', ['-m', 'pip', 'check'])
    record('audit', [ROOT / 'run_stage1.py', 'audit', '--out', out / 'audit.json'], summary=out / 'audit.json')
    record('pytest', ['-m', 'pytest', ROOT / 'PythonVehicleSimulator' / 'tests', ROOT / 'tests_stage1',
                      '-q', '--junitxml', out / 'pytest.xml', '--basetemp', out / 'pytest_temp'], cwd=pytest_work)
    record('original_remus', [ROOT / 'run_remus.py', '--duration', '200', '--out', out / 'original_remus'],
           summary=out / 'original_remus' / 'summary.json')
    record('regression', [ROOT / 'run_stage1.py', 'regression', '--out', out / 'regression'],
           summary=out / 'regression' / 'summary.json')
    record('remus_characterize', [ROOT / 'run_stage1.py', 'remus-characterize', '--out', out / 'remus_characterize'],
           summary=out / 'remus_characterize' / 'summary.json')
    record('remus_route', [ROOT / 'run_stage1.py', 'remus-route', '--out', out / 'remus_route'],
           summary=out / 'remus_route' / 'summary.json', expect_success=True)
    for learned in (False, True):
        for scenario in ('empty3d', 'static3d', 'crossing3d', 'paper_random3d'):
            name = f"{'pplnm' if learned else 'ca'}_{scenario}"
            command = [ROOT / 'run_stage1.py', 'simulate', '--scenario', scenario, '--out', out / name]
            if learned:
                command += ['--checkpoint', ROOT / 'artifacts_stage1/pplnm_150/pplnm.pt',
                            '--calibration', ROOT / 'artifacts_stage1/pplnm_150/pplnm_calibration.json']
            record(name, command, summary=out / name / 'summary.json', expect_success=True)
    record('numeric_visual_checks', [ROOT / 'scripts_stage1' / 'inspect_results.py', out],
           summary=out / 'numeric_visual_checks.json')
    report['completed_at'] = datetime.now().astimezone().isoformat()
    (out / 'acceptance_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Report: {out / "acceptance_report.json"}', flush=True)
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
