"""Portable, fail-closed release qualification; standard-library parent only.

Run with a pinned source-profile interpreter. All HDF5 operations remain in the
existing bounded runners. Evidence includes the unmodified known-negative matrix,
not a rewritten success result. See README.md for installed-environment options.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
import zipfile

VERSION = '0.2.0a1'
EPOCH = 1791244800  # 2026-10-06 00:00:00 UTC; transport timestamps only.
PROFILES = {
    'baseline': {'python': '3.12.14', 'numpy': '2.3.5', 'h5py': '3.14.0', 'hdf5': '1.14.6'},
    'current': {'python': '3.12.14', 'numpy': '2.3.5', 'h5py': '3.16.0', 'hdf5': '2.0.0'},
}
EXPECTED_CASES = {
    'original': dict.fromkeys(('plain_rectangle', 'shared_coherent', 'repeated_scale_coherent',
                              'multiple_scales', 'aliases_coherent', 'empty_selection'), 'success')
                | dict.fromkeys(('shared_conflict', 'repeated_scale_conflict', 'aliases_conflict',
                                 'reference_inbound', 'nonpositional_rank', 'nonpositional_length',
                                 'compound'), 'expected_refusal')
                | {'reference_outbound': 'failed'},
    'variants': dict.fromkeys(('reference_outbound_retained_alias',
                              'reference_outbound_explicit_both_aliases'), 'success'),
    'resources': dict.fromkeys(('offset_crossing_chunks', 'wide_rows', 'rank32', 'empty_contiguous',
                               'empty_rank32', 'large_sparse_tiny_crop', 'maximum_payload',
                               'shared_128_consumers'), 'success')
                 | dict.fromkeys(('compressed_overcap', 'uncompressed_overcap'), 'expected_refusal'),
}
EXPECTED_COUNTS = {
    'original': dict(successes=6, policy_or_resource_refusals=7, failures=1, mutation_controls=3, language_controls=14),
    'variants': dict(successes=2, policy_or_resource_refusals=0, failures=0, mutation_controls=1, language_controls=0),
    'resources': dict(successes=8, policy_or_resource_refusals=2, failures=0, mutation_controls=0, language_controls=0),
}
MUTATIONS = {'original': {('value', 'shared_coherent'), ('alias', 'aliases_coherent'), ('scale', 'shared_coherent')},
             'variants': {('ref', 'reference_outbound_retained_alias')}, 'resources': set()}
LANGUAGE_CONTROLS = {'bool', 'negative', 'past_end', 'rank', 'step', 'float', 'reversed',
                     'missing_mapping', 'retained_whole_consumer', 'unknown_key', 'duplicate_path',
                     'duplicate_mapping', 'nonfinite', 'duplicate_json_key'}
NATIVE_CONTROLS = {'memory_bytes': 2 * 1024**3, 'cpu_seconds': 120, 'wall_seconds': 180,
                   'file_bytes': 1024**3, 'dynamic_plugins_disabled': True,
                   'resource_controls_are_not_a_sandbox': True}


def validate_snapshot_result(result, mode):
    """Accept the exact retained matrices, never arbitrary nonzero results."""
    assert result['counts'] == EXPECTED_COUNTS[mode], result['counts']
    rows = result['cases']
    assert len(rows) == len(EXPECTED_CASES[mode])
    assert {row['case']: row['status'] for row in rows} == EXPECTED_CASES[mode]
    assert result['status'] == ('failed' if mode == 'original' else 'passed')
    if mode == 'original':
        assert len(result['failures']) == 1
        failure = result['failures'][0]
        assert failure['case'] == 'reference_outbound'
        assert '05-independent-observer' in failure['error']
        assert 'exact retained namespace differs' in failure['error']
    else:
        assert result['failures'] == []
    assert len(result['mutations']) == len(MUTATIONS[mode])
    assert {(item['kind'], item['case']) for item in result['mutations']} == MUTATIONS[mode]
    assert all(item['status'] == 'detected' for item in result['mutations'])
    expected_language = LANGUAGE_CONTROLS if mode == 'original' else set()
    assert len(result['language_controls']) == len(expected_language)
    assert {item['name'] for item in result['language_controls']} == expected_language
    assert all(item['status'] == 'refused' for item in result['language_controls'])
    assert result['native_controls'] == NATIVE_CONTROLS
    assert result['production_unchanged_during_run'] is True
    assert result['production_sha256_start'] == result['production_sha256_end']
    for row in rows:
        assert row['source_unchanged'] is True
        assert row['source_sha256_before'] == row['source_sha256_after']
        if row['status'] == 'success':
            assert row['source_operation_path_unavailable_during_fresh_consumer'] is True
            assert row['no_stage_artifacts_after_success'] is True
        elif row['status'] == 'expected_refusal':
            assert row['no_plan_output_or_stage_after_refusal'] is True
    for record in result['frozen_sources'].values():
        assert record['unchanged'] is True
        assert record['sha256_before'] == record['sha256_after']


def validate_original_observer(record):
    assert record['tag'] == '05-independent-observer'
    assert record['exit_code'] == 2
    assert record['stdout'] == ''
    lines = record['stderr'].splitlines()
    assert len(lines) >= 2
    assert lines[-2] == 'AssertionError: exact retained namespace differs'
    assert ast.literal_eval(lines[-1]) == {
        'ok': False, 'error': {'code': 'ERROR', 'message': 'Native worker exited unexpectedly with status 1'}}


def validate_output_location(source, output):
    """Keep output out of trees copied or treated as immutable input evidence."""
    source, output = source.resolve(), output.resolve()
    if output.is_relative_to(source / 'qualification/runs'):
        return
    for name in ('src', 'tests', 'examples', 'docs', 'qualification', 'evidence', '.git'):
        if output.is_relative_to(source / name):
            raise ValueError('Output must be outside input trees, or under qualification/runs')


def historical_hashes(source):
    root = source / 'evidence'
    manifest = dict(line.split('  ', 1)[::-1] for line in (root / 'SHA256SUMS').read_text().splitlines())
    paths = sorted(path for path in root.rglob('*') if path.is_file() and path.suffix in ('.h5', '.bin'))
    assert sum(path.suffix == '.h5' for path in paths) == 112, 'Expected all 112 original HDF5 fixtures'
    assert sum(path.suffix == '.bin' for path in paths) == 12, 'Expected all 12 held original sources'
    paths += [root / 'original-workflows/baseline/product-families' / f'family-{number}' / 'plan.json'
              for number in range(1, 5)]
    hashes = {str(path.relative_to(root)): sha(path) for path in paths}
    assert all(manifest.get(name) == digest for name, digest in hashes.items()), 'Historical fixture differs from retained public manifest'
    return hashes


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def product_hashes(package_root):
    return {str(path.relative_to(package_root)): sha(path)
            for path in sorted((package_root / 'h5carry').rglob('*'))
            if path.is_file() and path.suffix in ('.py', '.json')}


def retained_inputs(project):
    """Fixture definitions, original tests, observers and public qualification inputs."""
    return {str(path.relative_to(project)): sha(path)
            for name in ('tests', 'examples', 'qualification')
            for path in sorted((project / name).rglob('*'))
            if path.is_file() and path.suffix in ('.py', '.json', '.md', '.txt')
            and 'runs' not in path.relative_to(project / name).parts}


def clean_environment(package_root=None, project=None):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('PYTHON', 'LD_', 'DYLD_', 'HDF5_', 'H5CARRY_'))}
    if package_root is not None:
        env['PYTHONPATH'] = os.pathsep.join(map(str, (package_root, project)))
    return env


def preamble(package_root, project):
    return ('import sys,pathlib,runpy; sys.path[:0]=' + repr([str(package_root), str(project)]) + '; '
            'import h5carry; assert pathlib.Path(h5carry.__file__).resolve() == pathlib.Path(' +
            repr(str(package_root / 'h5carry/__init__.py')) + '), "unexpected package origin"; '
            'assert h5carry.__version__ == ' + repr(VERSION) + '; ')


def module_command(python, package_root, project, module, *arguments):
    return [python, '-I', '-c', preamble(package_root, project) +
            'runpy.run_module(' + repr(module) + ', run_name="__main__")', *arguments]


def native_runtime():
    # Imported native libraries are deliberately confined to this supervised branch.
    from examples._bounded import require_native_limits
    require_native_limits()
    assert os.environ.get('H5CARRY_TEST_NATIVE_CHILD') == '1'
    import platform
    import resource
    from h5carry.supervisor import ProcessLimits
    limits = ProcessLimits().to_dict()
    assert limits == {'memory_bytes': 2 * 1024**3, 'cpu_seconds': 120, 'wall_seconds': 180,
                      'file_bytes': 1024**3, 'result_bytes': 8 * 1024**2,
                      'stdout_bytes': 64 * 1024, 'stderr_bytes': 64 * 1024}
    assert resource.getrlimit(resource.RLIMIT_AS) == (2 * 1024**3, 2 * 1024**3)
    assert resource.getrlimit(resource.RLIMIT_CPU) == (120, 121)
    assert resource.getrlimit(resource.RLIMIT_FSIZE) == (1024**3, 1024**3)
    import h5py
    import numpy
    print(json.dumps({'python': platform.python_version(), 'numpy': numpy.__version__,
                      'h5py': h5py.__version__, 'hdf5': h5py.version.hdf5_version,
                      'system': platform.system(), 'machine': platform.machine(),
                      'native_limits': limits}, sort_keys=True))


class Qualification:
    def __init__(self, args):
        self.args = args
        self.source = Path(__file__).resolve().parents[1]
        self.root = args.out.resolve()
        validate_output_location(self.source, self.root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.evidence = self.root / 'evidence'
        self.work = self.root / 'work'
        self.evidence.mkdir()
        self.work.mkdir()
        (self.evidence / 'logs').mkdir()
        self.expected_product = product_hashes(self.source / 'src')
        self.expected_inputs = retained_inputs(self.source)
        self.expected_historical = historical_hashes(self.source)
        self.report = {'version': VERSION, 'profile': args.profile, 'status': 'running',
                       'expected_runtime': PROFILES[args.profile], 'commands': [], 'surfaces': {},
                       'production_and_schemas': self.expected_product,
                       'retained_source_inputs': self.expected_inputs,
                       'historical_fixtures_and_plans': self.expected_historical,
                       'known_negative': 'Original reference_outbound observer remains failed; exact namespace discrepancy required',
                       'scope': 'Original benign local synthetic inputs, existing native limits; no hostile-file safety claim'}
        self.write_report()

    def write_report(self):
        save(self.evidence / 'qualification-result.json', self.report)

    def run(self, name, command, cwd, env=None, expected=0):
        started = time.monotonic()
        record = {'name': name, 'command': list(map(str, command)), 'cwd': str(cwd),
                  'expected_exit': expected}
        try:
            process = subprocess.run(record['command'], cwd=cwd, env=env or clean_environment(),
                                     capture_output=True, text=True, timeout=1200)
            record.update(exit_code=process.returncode, stdout=process.stdout, stderr=process.stderr)
        except subprocess.TimeoutExpired as exc:
            def text(value):
                return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else value or ''
            record.update(exit_code=None, stdout=text(exc.stdout), stderr=text(exc.stderr),
                          error='Orchestration deadline exceeded; no success claimed')
        record['elapsed_seconds'] = round(time.monotonic() - started, 6)
        record['passed'] = record['exit_code'] == expected
        save(self.evidence / 'logs' / (name + '.json'), record)
        (self.evidence / 'logs' / (name + '.log')).write_text(record['stdout'] + record['stderr'])
        self.report['commands'].append({key: value for key, value in record.items() if key not in ('stdout', 'stderr')})
        self.write_report()
        print(json.dumps({'command': name, 'exit_code': record['exit_code'], 'expected_exit': expected}), flush=True)
        if not record['passed']:
            raise RuntimeError(name + ': ' + record['stdout'][-2000:] + record['stderr'][-2000:])
        return record

    def copy_project(self, destination, *, source=False):
        destination.mkdir()
        for name in ('tests', 'examples', 'qualification') + (('src', 'docs') if source else ()):
            shutil.copytree(self.source / name, destination / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.egg-info', 'runs'))
        if source:
            for name in ('pyproject.toml', 'MANIFEST.in', 'LICENSE', 'README.md'):
                shutil.copyfile(self.source / name, destination / name)
        else:
            assert not (destination / 'src').exists()
        assert retained_inputs(destination) == self.expected_inputs

    def build(self):
        identities = []
        for number in (1, 2):
            project = self.work / f'build-{number}'
            self.copy_project(project, source=True)
            output = self.evidence / f'distributions-{number}'
            output.mkdir()
            env = clean_environment()
            env.update(SOURCE_DATE_EPOCH=str(EPOCH), PYTHONHASHSEED='0')
            self.run(f'build-{number}', [sys.executable, '-m', 'build', '--no-isolation', '--outdir', output], project, env)
            assert {item.name for item in output.iterdir()} == {
                f'h5carry-{VERSION}-py3-none-any.whl', f'h5carry-{VERSION}.tar.gz'}
            archive = output / f'h5carry-{VERSION}.tar.gz'
            raw = archive.read_bytes()
            (output / (archive.name + '.original-build-bytes')).write_bytes(raw)
            content = io.BytesIO()
            with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as original, \
                    tarfile.open(fileobj=content, mode='w', format=tarfile.PAX_FORMAT) as normalized:
                for entry in sorted(original.getmembers(), key=lambda item: item.name):
                    assert not entry.name.startswith('/') and '..' not in Path(entry.name).parts
                    assert entry.isfile() or entry.isdir()
                    entry.mtime = EPOCH
                    entry.uid = entry.gid = 0
                    entry.uname = entry.gname = ''
                    entry.pax_headers = {}
                    normalized.addfile(entry, original.extractfile(entry) if entry.isfile() else None)
            with archive.open('wb') as stream:
                with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=EPOCH, compresslevel=9) as zipped:
                    zipped.write(content.getvalue())
            identities.append({path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
                               for path in sorted(output.iterdir()) if path.suffix in ('.whl', '.gz')})
        assert identities[0] == identities[1], 'Repeated build artifact identities differ'
        self.report['builds'] = {'repeated_builds_identical': True, 'source_date_epoch': EPOCH,
                                'sdist_transport_metadata_normalized': True,
                                'original_build_bytes_retained': True, 'artifacts': identities[0]}
        self.write_report()
        output = self.evidence / 'distributions-1'
        wheel = output / f'h5carry-{VERSION}-py3-none-any.whl'
        with zipfile.ZipFile(wheel) as archive:
            hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                      if name.startswith('h5carry/') and Path(name).suffix in ('.py', '.json')}
        assert hashes == self.expected_product, 'Wheel product/schema differs from source'
        extracted = self.work / 'extracted'
        extracted.mkdir()
        sdist = output / f'h5carry-{VERSION}.tar.gz'
        with tarfile.open(sdist) as archive:
            archive.extractall(extracted, filter='data')
        project = extracted / f'h5carry-{VERSION}'
        assert list(extracted.iterdir()) == [project]
        assert product_hashes(project / 'src') == self.expected_product
        assert retained_inputs(project) == self.expected_inputs, 'sdist omitted or changed fixtures/tests/qualification inputs'
        return wheel, sdist, project

    def environment(self, kind, supplied):
        if supplied:
            # Do not resolve the interpreter symlink: its venv prefix is significant.
            python = supplied.absolute()
            assert python.is_file()
        else:
            env_root = self.work / ('env-' + kind)
            self.run(kind + '-venv', [sys.executable, '-m', 'venv', env_root], self.work)
            python = env_root / 'bin/python'
            self.run(kind + '-dependencies', [python, '-m', 'pip', 'install', '-c',
                     self.source / 'qualification' / f'constraints-{self.args.profile}.txt',
                     'build', 'setuptools', 'wheel', 'h5py', 'numpy'], self.work)
        record = self.run(kind + '-environment', [python, '-I', '-c',
            'import json,sys,sysconfig,pathlib; assert sys.prefix != sys.base_prefix; '
            'assert "include-system-site-packages = false" in (pathlib.Path(sys.prefix)/"pyvenv.cfg").read_text().lower(); '
            'print(json.dumps({"package_root":sysconfig.get_path("purelib"),"prefix":sys.prefix}))'], self.work)
        return python, Path(json.loads(record['stdout'])['package_root']).resolve()

    def legacy_v1(self, name, python, project, package_root, output, env):
        destination = output / 'legacy-v1'
        destination.mkdir()
        records = []
        for number in range(1, 5):
            family = self.source / 'evidence/original-workflows/baseline/product-families' / f'family-{number}'
            before = {part: sha(family / part) for part in ('source.h5', 'selected.h5', 'plan.json')}
            command = module_command(python, package_root, project, 'h5carry', 'verify',
                family / 'source.h5', family / 'selected.h5', '--plan', family / 'plan.json',
                '--report', destination / f'family-{number}-verify.json')
            self.run(name + f'-legacy-v1-{number}', command, project, env)
            verified = json.loads((destination / f'family-{number}-verify.json').read_text())
            assert verified['status'] == 'verified' and verified['coverage']['source_equality']
            assert verified['diagnostics'] == []
            assert verified['runtime'] == PROFILES[self.args.profile]
            assert before == {part: sha(family / part) for part in before}
            record = {'family': number, 'historical_base_profile': 'baseline', 'status': 'verified',
                      'source_equality': True, 'original_artifacts_unchanged': True, 'sha256': before}
            records.append(record)
        save(destination / 'results.json', records)
        assert historical_hashes(self.source) == self.expected_historical

    def surface(self, name, python, project, package_root, *, installed=False, frozen=None):
        output = self.evidence / name
        output.mkdir()
        env = clean_environment(package_root, project)
        assert product_hashes(package_root) == self.expected_product
        assert retained_inputs(project) == self.expected_inputs
        code = preamble(package_root, project) + (
            'import json; assert "h5py" not in sys.modules and "numpy" not in sys.modules; '
            'print(json.dumps({"origin":str(pathlib.Path(h5carry.__file__).resolve()),'
            '"version":h5carry.__version__,"native_free_import":True}))')
        identity = json.loads(self.run(name + '-origin', [python, '-I', '-c', code], project, env)['stdout'])
        if installed:
            assert not (project / 'src').exists()
            self.run(name + '-installed-identity', [python, '-I', project / 'qualification/check_install.py'], project, env)
        runtime = json.loads(self.run(name + '-runtime', module_command(python, package_root, project,
            'tests.native_runner', 'qualification.run_ci', '--native-runtime'), project, env)['stdout'])
        assert {key: runtime[key] for key in PROFILES[self.args.profile]} == PROFILES[self.args.profile], runtime
        assert runtime['system'] == 'Linux' and runtime['machine'] == 'x86_64', runtime
        tests = self.run(name + '-tests', module_command(python, package_root, project,
            'tests.native_runner', 'unittest', 'discover', '-s', 'tests', '-v'), project, env)
        combined = tests['stdout'] + tests['stderr']
        assert re.search(r'\nRan 208 tests in [0-9.]+s\n\nOK\s*$', combined), 'Expected exactly 208 tests, no skips or failures'
        matrices = {}
        for mode, flags in (('original', []), ('variants', ['--contract-variants']), ('resources', ['--resources'])):
            destination = output / mode
            command = [python, project / 'qualification/run_snapshots.py', '--project', project,
                       '--package-root', package_root, '--out', destination, *flags]
            frozen_root = frozen if mode == 'original' else (frozen or output / 'original') if mode == 'variants' else None
            if frozen_root:
                command.extend(['--frozen-root', frozen_root, '--frozen-manifest',
                                self.evidence / 'frozen-source-manifest.json', '--profile', self.args.profile])
            self.run(name + '-' + mode, command, project, env, expected=1 if mode == 'original' else 0)
            result = json.loads((destination / 'results.json').read_text())
            validate_snapshot_result(result, mode)
            assert result['candidate_package'] == str(package_root / 'h5carry/__init__.py')
            assert result['production_sha256_start'] == self.expected_product
            for row in result['cases']:
                assert sha(destination / row['case'] / 'source.h5') == row['source_sha256_before']
                if 'runtime' in row:
                    assert {key: row['runtime'][key] for key in PROFILES[self.args.profile]} == PROFILES[self.args.profile]
            if mode == 'original':
                validate_original_observer(json.loads((destination / 'reference_outbound/05-independent-observer.json').read_text()))
                if frozen is None:
                    save(self.evidence / 'frozen-source-manifest.json', [
                        {'case': row['case'], 'profile': self.args.profile, 'sha256': row['source_sha256_before']}
                        for row in result['cases']])
            if mode == 'resources':
                maximum = next(row for row in result['cases'] if row['case'] == 'maximum_payload')
                assert maximum['selected_payload_bytes'] == 268435456
            matrices[mode] = result['counts']
        self.run(name + '-baselines', [python, project / 'qualification/run_baselines.py', output / 'baselines'], project, env)
        baseline = json.loads((output / 'baselines/baseline-results.json').read_text())
        assert baseline['status'] == 'complete' and len(baseline['families']) == 4
        assert baseline['plain_source_unchanged'] and len(baseline['plain_controls']) == 2
        assert all(item['source_unchanged'] for item in baseline['families'])
        assert all(all(item['checks'].values()) for item in baseline['plain_controls'])
        self.run(name + '-frozen-families', [python, project / 'qualification/run_product_families.py',
                 '--project', project, '--out', output / 'frozen-families'], project, env)
        families = json.loads((output / 'frozen-families/frozen-results.json').read_text())['families']
        assert {row['family'] for row in families} == {1, 2, 3, 4} and len(families) == 4
        assert all(row['status'] == 'passed' and row['source_unchanged'] for row in families)
        self.run(name + '-demo', [python, project / 'qualification/run_demo.py',
                 '--project', project, '--out', output / 'demo'], project, env)
        demos = json.loads((output / 'demo/demo-results.json').read_text())['cases']
        assert {row['name'] for row in demos} == {'frozen-two-run', 'supplemental-demo'} and len(demos) == 2
        assert all(row['status'] == 'passed' and row['source_unchanged'] and
                   row['original_source_path_absent_for_analysis'] and row['analysis']['combined_total'] == 3346
                   for row in demos)
        self.legacy_v1(name, python, project, package_root, output, env)
        assert product_hashes(package_root) == self.expected_product
        assert retained_inputs(project) == self.expected_inputs
        self.report['surfaces'][name] = {'status': 'passed-with-retained-known-negative', 'tests': 208,
            'identity': identity, 'runtime': runtime, 'matrices': matrices,
            'frozen_families': 4, 'unchanged_legacy_v1_triples': 4, 'demo_journeys': 2, 'baselines': 4,
            'production_schemas_and_fixture_definitions_unchanged': True}
        self.write_report()

    def qualify(self):
        wheel, sdist, extracted = self.build()
        self.surface('source', Path(sys.executable), self.source, self.source / 'src')
        frozen = self.evidence / 'source/original'
        for kind, artifact, supplied in (('wheel', wheel, self.args.wheel_python),
                                         ('sdist', sdist, self.args.sdist_python)):
            python, package_root = self.environment(kind, supplied)
            project = self.work / ('installed-' + kind)
            self.copy_project(project)
            self.run(kind + '-install', [python, '-m', 'pip', 'install', '--force-reinstall', '--no-deps',
                     '--no-build-isolation', '--no-index', artifact], project)
            self.surface('installed-' + kind, python, project, package_root, installed=True, frozen=frozen)
            if kind == 'sdist':
                self.surface('extracted-sdist', python, extracted, extracted / 'src', frozen=frozen)
        for row in json.loads((self.evidence / 'frozen-source-manifest.json').read_text()):
            assert sha(frozen / row['case'] / 'source.h5') == row['sha256']
        assert product_hashes(self.source / 'src') == self.expected_product
        assert retained_inputs(self.source) == self.expected_inputs
        assert historical_hashes(self.source) == self.expected_historical
        self.report.update(status='passed-with-retained-known-negative',
                           all_112_historical_hdf5_and_12_held_sources_retained_unchanged=True,
                           all_original_snapshot_sources_retained_unchanged=True)
        self.write_report()


def main():
    if sys.flags.optimize:
        raise SystemExit('Qualification requires Python assertions enabled (no -O)')
    if sys.argv[1:] == ['--native-runtime']:
        native_runtime()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(PROFILES), required=True)
    parser.add_argument('--out', type=Path, required=True, help='New directory; evidence and disposable work are separated')
    parser.add_argument('--wheel-python', type=Path, help='Optional existing self-contained wheel venv, with pinned dependencies')
    parser.add_argument('--sdist-python', type=Path, help='Optional existing self-contained sdist venv, with pinned dependencies')
    args = parser.parse_args()
    qualification = Qualification(args)
    try:
        qualification.qualify()
    except BaseException as exc:
        qualification.report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        (qualification.evidence / 'failure.log').write_text(traceback.format_exc())
        qualification.write_report()
        raise
    finally:
        # Include complete raw negative results, native command records and original
        # synthetic fixtures. Do not include build workspaces or virtual environments.
        (qualification.evidence / 'SHA256SUMS').write_text(''.join(
            sha(path) + '  ' + str(path.relative_to(qualification.evidence)) + '\n'
            for path in sorted(qualification.evidence.rglob('*'))
            if path.is_file() and path.name != 'SHA256SUMS'))
    print(json.dumps({'profile': args.profile, 'status': qualification.report['status'],
                      'evidence': str(qualification.evidence)}, sort_keys=True))


if __name__ == '__main__':
    main()
