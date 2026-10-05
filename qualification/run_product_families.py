"""Product exports of all original frozen families; standard-library parent."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
project = args.project.resolve()
root = args.out.resolve()
root.mkdir(parents=True, exist_ok=False)
env = dict(os.environ)
env['PYTHONPATH'] = str(project / 'src') + os.pathsep + str(project)
report = {'scope': 'product exports of the four original frozen fixture families', 'families': []}

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

for family in range(1, 5):
    directory = root / f'family-{family}'
    directory.mkdir()
    records = []
    def run(command, display):
        result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=200)
        records.append({'command': display, 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
        (directory / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(display + ': ' + result.stdout + result.stderr)
        return json.loads(result.stdout)
    run([sys.executable, str(project / 'examples/three_run_archive.py'), 'source.h5', '--family', str(family)],
        f'python examples/three_run_archive.py source.h5 --family {family}')
    source_check = run([sys.executable, str(project / 'qualification/check_frozen.py'), 'source.h5', '--family', str(family), '--source'],
        f'python qualification/check_frozen.py source.h5 --family {family} --source')
    initial = sha(directory / 'source.h5')
    selection = ['--select', '/runs/selected'] + (['--select', '/runs/other'] if family == 2 else [])
    for arguments in (['plan', 'source.h5', *selection, '--out', 'plan.json'],
                      ['export', 'source.h5', '--plan', 'plan.json', '--out', 'selected.h5', '--report', 'export-report.json'],
                      ['verify', 'source.h5', 'selected.h5', '--plan', 'plan.json', '--report', 'verify-report.json'],
                      ['inspect', 'selected.h5', '--report', 'inspect-report.json']):
        run([sys.executable, '-m', 'h5carry', *arguments], 'h5carry ' + ' '.join(arguments))
    observed = run([sys.executable, str(project / 'qualification/check_frozen.py'), 'selected.h5', '--family', str(family)],
        f'python qualification/check_frozen.py selected.h5 --family {family}')
    assert initial == sha(directory / 'source.h5')
    runtime = json.loads((directory / 'verify-report.json').read_text())['runtime']
    record = {'family': family, 'status': 'passed', 'source': source_check,
              'manual_reopened_output_check': observed, 'source_sha256': initial,
              'source_unchanged': True, 'output_sha256': sha(directory / 'selected.h5'), 'runtime': runtime}
    report['families'].append(record)
    (root / 'frozen-results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': 'passed', 'families': len(report['families']), 'runtime': runtime}, sort_keys=True))
