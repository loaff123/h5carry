"""Original local end-to-end demo qualification driver; no native imports."""
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

def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

def invoke(directory, command, public_command, records):
    result = subprocess.run(command, cwd=directory, env=env, text=True, capture_output=True, timeout=200)
    record = {'command': public_command, 'exit_code': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    records.append(record)
    (directory / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError(public_command + ' failed: ' + result.stdout + result.stderr)
    return json.loads(result.stdout)

report = {'scope': 'frozen two-run family and separately supplemental demo', 'cases': []}
for supplemental in (False, True):
    label = 'supplemental-demo' if supplemental else 'frozen-two-run'
    directory = root / label
    directory.mkdir()
    commands = []
    generator = [sys.executable, str(project / 'examples/three_run_archive.py'), 'source.h5']
    generator += ['--demo'] if supplemental else ['--family', '2']
    invoke(directory, generator, 'python examples/three_run_archive.py source.h5 ' + ('--demo' if supplemental else '--family 2'), commands)
    source_hash = digest(directory / 'source.h5')
    for arguments in (
        ['plan', 'source.h5', '--select', '/runs/selected', '--select', '/runs/other', '--out', 'plan.json'],
        ['export', 'source.h5', '--plan', 'plan.json', '--out', 'selected.h5', '--report', 'export-report.json'],
        ['verify', 'source.h5', 'selected.h5', '--plan', 'plan.json', '--report', 'verify-report.json'],
        ['inspect', 'selected.h5', '--report', 'inspect-report.json'],
    ):
        invoke(directory, [sys.executable, '-m', 'h5carry', *arguments], 'h5carry ' + ' '.join(arguments), commands)
    source_unchanged = source_hash == digest(directory / 'source.h5')
    assert source_unchanged
    relocated = directory / 'relocated'
    relocated.mkdir()
    assert not list(relocated.iterdir())
    (directory / 'selected.h5').rename(relocated / 'selected.h5')
    holding = directory / 'source-hold'
    holding.mkdir()
    (directory / 'source.h5').rename(holding / 'source-held.bin')
    assert not (directory / 'source.h5').exists()
    extra = ['--require-demo'] if supplemental else []
    analysis = invoke(directory, [sys.executable, str(project / 'examples/analyze_selected.py'), 'relocated/selected.h5', *extra],
                      'python examples/analyze_selected.py relocated/selected.h5' + (' --require-demo' if supplemental else ''), commands)
    assert analysis['combined_total'] == 3346 and analysis['excluded_marker_absent']
    verify = json.loads((directory / 'verify-report.json').read_text())
    case = {'name': label, 'runtime': verify.get('runtime'), 'status': 'passed', 'source_sha256': source_hash,
            'source_unchanged': source_unchanged, 'original_source_path_absent_for_analysis': True,
            'source_bytes_held_aside_not_os_denied': True,
            'output_relocated_into_new_empty_directory': True, 'analysis': analysis,
            'output_sha256': digest(relocated / 'selected.h5')}
    (directory / 'analysis-result.json').write_text(json.dumps(analysis, indent=2) + '\n')
    report['cases'].append(case)
    (root / 'demo-results.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, sort_keys=True))
