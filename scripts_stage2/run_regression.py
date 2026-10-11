"""Record the retained stage 1 and new stage 2 regression checks on CPU."""
from __future__ import annotations

import argparse
from datetime import datetime
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
    parser.add_argument('--tests-only',action='store_true',help='versions, dependency check, audit and all pytest tests')
    args = parser.parse_args()
    out = (args.out or ROOT/'results_stage2'/f'regression_{datetime.now():%Y%m%d_%H%M%S}').resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('use a new or empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MPLBACKEND='Agg', PYTHONUTF8='1')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT),str(ROOT/'PythonVehicleSimulator'/'src')])
    records = []
    report = {'started_at':datetime.now().astimezone().isoformat(),
              'python':sys.version,'platform':platform.platform(),
              'executable':sys.executable,'checks':records}

    def check(name, arguments, cwd=ROOT, summary=None, task_success=False):
        command = [sys.executable, *map(str, arguments)]
        logfile = out/f'{name}.log'
        print(f'RUN {name}', flush=True)
        start = time.perf_counter()
        with logfile.open('w',encoding='utf-8') as stream:
            completed = subprocess.run(command,cwd=cwd,env=env,stdout=stream,stderr=subprocess.STDOUT)
        row = {'name':name,'command':subprocess.list2cmdline(command),'cwd':str(cwd),
               'exit_code':completed.returncode,'wall_clock_s':time.perf_counter()-start,
               'log':str(logfile),'passed':completed.returncode==0}
        if summary and Path(summary).exists():
            row['summary'] = json.loads(Path(summary).read_text(encoding='utf-8'))
            if task_success:
                row['passed'] &= row['summary'].get('status')=='success'
            if name=='responses':
                row['passed'] &= row['summary'].get('all_passed',False)
        if name=='pytest' and (out/'pytest.xml').exists():
            suites = ET.parse(out/'pytest.xml').getroot()
            row['junit'] = {key:sum(int(s.attrib.get(key,0)) for s in suites.iter('testsuite'))
                            for key in ('tests','failures','errors','skipped')}
            row['passed'] &= all(row['junit'][key]==0 for key in ('failures','errors','skipped'))
        records.append(row)
        report['all_passed'] = all(r['passed'] for r in records)
        (out/'regression_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print(f"{'PASS' if row['passed'] else 'FAIL'} {name}: {row['wall_clock_s']:.2f}s",flush=True)

    work = out/'pytest_work'
    work.mkdir()
    check('versions',['-m','pip','list','--format=json'])
    check('pip_check',['-m','pip','check'])
    check('audit',[ROOT/'run_stage1.py','audit','--out',out/'audit.json'],summary=out/'audit.json')
    check('pytest',['-m','pytest',ROOT/'PythonVehicleSimulator'/'tests',ROOT/'tests_stage1',
                    ROOT/'tests_stage2','-q','--junitxml',out/'pytest.xml','--basetemp',out/'pytest_temp'],cwd=work)
    if args.tests_only:
        report['completed_at'] = datetime.now().astimezone().isoformat()
        (out/'regression_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        return 0 if report['all_passed'] else 1
    check('original_remus',[ROOT/'run_remus.py','--duration','200','--out',out/'original_remus'],
          summary=out/'original_remus'/'summary.json')
    check('regression',[ROOT/'run_stage1.py','regression','--out',out/'regression'],
          summary=out/'regression'/'summary.json')
    check('responses',[ROOT/'run_stage2.py','responses','--out',out/'responses'],summary=out/'responses'/'summary.json')
    check('diagnose',[ROOT/'run_stage2.py','diagnose','--out',out/'diagnose'],summary=out/'diagnose'/'summary.json')
    for learned in (False,True):
        for scene in ('empty3d','static3d','crossing3d','paper_random3d'):
            name = f"stage1_{'pplnm' if learned else 'ca'}_{scene}"
            command = [ROOT/'run_stage1.py','simulate','--scenario',scene,'--out',out/name]
            if learned:
                command += ['--checkpoint',ROOT/'artifacts_stage1/pplnm_150/pplnm.pt',
                            '--calibration',ROOT/'artifacts_stage1/pplnm_150/pplnm_calibration.json']
            check(name,command,summary=out/name/'summary.json',task_success=True)
    report['completed_at'] = datetime.now().astimezone().isoformat()
    (out/'regression_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(out/'regression_report.json',flush=True)
    return 0 if report['all_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
