"""Bounded CLI qualification of the frozen original snapshot journeys.

The standard-library parent never opens HDF5. All fixture, observer and mutation
operations run through tests.native_runner; CLI native operations use the product
supervisor. Exact candidate import origin is asserted in every subprocess.

Portable replay:
  python qualification/run_snapshots.py --out /tmp/snapshot-replay
Frozen replay (files are only read and copied):
  python qualification/run_snapshots.py --frozen-root PATH --frozen-manifest PATH \
      --profile baseline --out /tmp/snapshot-frozen
Separate resource controls:
  python qualification/run_snapshots.py --resources --out /tmp/snapshot-resources
Every output directory is new so previous red logs remain intact.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def product_hashes(package_root):
    return {str(path.relative_to(package_root)): sha(path) for path in sorted((package_root/'h5carry').rglob('*'))
            if path.is_file() and path.suffix in ('.py', '.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--package-root', type=Path, help='Explicit directory containing the candidate h5carry package, for extracted-wheel or installed-artifact replay; default PROJECT/src')
    parser.add_argument('--frozen-root', type=Path)
    parser.add_argument('--frozen-manifest', type=Path)
    parser.add_argument('--profile', choices=('baseline', 'current'))
    parser.add_argument('--resources', action='store_true')
    parser.add_argument('--contract-variants', action='store_true')
    parser.add_argument('--only-case', action='append')
    args = parser.parse_args()
    project, root = args.project.resolve(), args.out.resolve()
    package_root = args.package_root.resolve() if args.package_root else project/'src'
    if not (package_root/'h5carry'/'__init__.py').is_file():
        parser.error('Explicit candidate package root has no h5carry package')
    if args.resources and args.contract_variants:
        parser.error('Contract variants and resources are separate runs')
    if args.resources and args.frozen_root:
        parser.error('Resource fixtures are separate from frozen original sources')
    if args.frozen_manifest and not (args.frozen_root and args.profile):
        parser.error('A frozen manifest requires --frozen-root and --profile')
    root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(project))
    from examples.snapshot_fixture import original_cases, resource_cases, contract_variant_cases
    cases = resource_cases() if args.resources else contract_variant_cases() if args.contract_variants else original_cases()
    if args.only_case:
        unknown = set(args.only_case) - {case['name'] for case in cases}
        if unknown:
            parser.error('Unknown cases: ' + ', '.join(sorted(unknown)))
        cases = [case for case in cases if case['name'] in args.only_case]
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('PYTHON', 'LD_', 'DYLD_', 'HDF5_', 'H5CARRY_'))}
    package = package_root/'h5carry'/'__init__.py'
    preamble = ('import sys, pathlib, runpy; sys.path[:0] = ' + repr([str(package_root), str(project)]) + '; '
                'import h5carry; assert pathlib.Path(h5carry.__file__).resolve() == pathlib.Path(' + repr(str(package)) + '), '
                '"unexpected candidate import origin"; ')
    cli = [sys.executable, '-I', '-c', preamble + 'from h5carry.cli import main; raise SystemExit(main())']
    native = [sys.executable, '-I', '-c', preamble + 'runpy.run_module("tests.native_runner", run_name="__main__")']
    report = {'scope': 'separate bounded resource controls' if args.resources else 'separate reference namespace contract variants' if args.contract_variants else 'frozen original 14 snapshot case identities',
              'case_definition': '2026-10-06 design PROBE-PLAN original coordinates, exact literal values and declared outcomes',
              'candidate_package': str(package), 'python_executable': sys.executable,
              'native_controls': {'memory_bytes': 2*1024**3, 'cpu_seconds': 120, 'wall_seconds': 180,
                                  'file_bytes': 1024**3, 'dynamic_plugins_disabled': True,
                                  'resource_controls_are_not_a_sandbox': True},
              'source_unavailable_scope': 'Original operation pathname is absent during a fresh-process relocated output-only consumer; the byte-identical held source remains, without OS filesystem isolation',
              'production_sha256_start': product_hashes(package_root), 'cases': [], 'mutations': [],
              'protection_controls': [], 'language_controls': [], 'failures': []}
    frozen_initial = {}
    manifest = {}
    if args.frozen_manifest:
        records = json.loads(args.frozen_manifest.read_text())
        manifest = {item['case']: item['sha256'] for item in records if item['profile'] == args.profile}
    if args.frozen_root:
        for case in cases:
            source = args.frozen_root.resolve()/case.get('source_case', case['name'])/'source.h5'
            frozen_initial[case['name']] = {'path': str(source), 'sha256_before': sha(source)}
            if manifest and frozen_initial[case['name']]['sha256_before'] != manifest[case.get('source_case', case['name'])]:
                raise AssertionError('Frozen source no longer equals pre-implementation manifest: ' + case['name'])
    report['frozen_sources'] = frozen_initial

    def save():
        (root/'results.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')

    def run(folder, tag, command):
        started = time.monotonic()
        try:
            result = subprocess.run(command, cwd=folder, env=environment, capture_output=True, text=True, timeout=600)
            record = {'tag': tag, 'command': command, 'exit_code': result.returncode, 'stdout': result.stdout,
                      'stderr': result.stderr, 'elapsed_seconds': round(time.monotonic()-started, 6)}
        except subprocess.TimeoutExpired as exc:
            record = {'tag': tag, 'command': command, 'exit_code': None, 'stdout': str(exc.stdout or ''),
                      'stderr': str(exc.stderr or ''), 'elapsed_seconds': round(time.monotonic()-started, 6),
                      'failure': 'outer timeout; no success claimed'}
        (folder/(tag+'.json')).write_text(json.dumps(record, indent=2)+'\n')
        try:
            record['value'] = json.loads(record['stdout'])
        except (ValueError, TypeError):
            record['value'] = None
        return record

    def success(record, status):
        if record['exit_code'] != 0 or not isinstance(record['value'], dict) or record['value'].get('status') != status:
            raise AssertionError(record['tag'] + ': expected ' + status + ', got ' + json.dumps(record))
        return record['value']

    def refusal(record, expected):
        value = record['value']
        if record['exit_code'] != 2 or not isinstance(value, dict) or value.get('status') != 'incomplete':
            raise AssertionError('Expected policy/resource refusal: ' + json.dumps(record))
        diagnostics = value.get('diagnostics', [])
        if len(diagnostics) != 1 or diagnostics[0].get('code') not in ('INVALID', 'UNSUPPORTED', 'RESOURCE'):
            raise AssertionError('Unexpected refusal diagnostic: ' + json.dumps(value))
        diagnostic = diagnostics[0]
        if expected == 'RESOURCE':
            if diagnostic['code'] != 'RESOURCE' or 'payload' not in diagnostic['message'].lower():
                raise AssertionError('Expected selected-payload refusal, got ' + json.dumps(value))
        elif expected not in diagnostic['message'].lower():
            raise AssertionError('Refusal category differs: ' + json.dumps(value))
        return value

    for case in cases:
        directory = root/case['name']; directory.mkdir()
        row = {'case': case['name'], 'expected_refusal': case['expected_refusal'], 'request': case['request'],
               'status': 'running'}
        for key in ('original_recipe_request', 'adapter_note', 'source_case', 'checker_case'):
            if key in case:
                row[key] = case[key]
        report['cases'].append(row); save()
        source, plan, output = directory/'source.h5', directory/'plan.json', directory/'snapshot.h5'
        request_file = directory/'selection.json'
        request_file.write_text(json.dumps(case['request'], indent=2)+'\n')
        if 'original_recipe_request' in case:
            (directory/'frozen-original-request.json').write_text(json.dumps(case['original_recipe_request'], indent=2)+'\n')
        try:
            if args.frozen_root:
                shutil.copyfile(args.frozen_root.resolve()/case.get('source_case', case['name'])/'source.h5', source)
                row['frozen_sha256_matches_source_copy'] = sha(source) == frozen_initial[case['name']]['sha256_before']
                if not row['frozen_sha256_matches_source_copy']:
                    raise AssertionError('Exact-byte source copy differs')
            else:
                fixture_args = ['examples.snapshot_fixture', str(source), '--case', case.get('source_case', case['name'])]
                if args.resources:
                    fixture_args.append('--resources')
                success(run(directory, '00-create-fixture', native+fixture_args), 'generated')
            row['source_sha256_before'] = sha(source)
            planned = run(directory, '01-cli-plan', cli+['plan', str(source), '--selection-json', str(request_file),
                          '--out', str(plan), *case.get('limits', [])])
            if case['expected_refusal']:
                row['refusal'] = refusal(planned, case['expected_refusal'])
                if plan.exists() or output.exists() or list(directory.glob('.h5carry-*')):
                    raise AssertionError('Refused plan created a plan/output/stage artifact')
                row['no_plan_output_or_stage_after_refusal'] = True
                row['status'] = 'expected_refusal'
            else:
                row['plan_result'] = success(planned, 'planned')
                decoded = json.loads(plan.read_text())
                if decoded['version'] != 2 or decoded['request'] != case['request']:
                    raise AssertionError('Plan did not retain exact typed original request/version')
                row['plan_version'] = 2
                row['selected_payload_bytes'] = decoded['graph']['payload_bytes']
                row['transformations'] = decoded['transformations']
                exported = success(run(directory, '02-cli-export', cli+['export', str(source), '--plan', str(plan),
                      '--out', str(output), '--report', str(directory/'export-report.json')]), 'verified')
                verified = success(run(directory, '03-cli-verify', cli+['verify', str(source), str(output), '--plan', str(plan),
                      '--report', str(directory/'verify-report.json')]), 'verified')
                inspected = success(run(directory, '04-cli-inspect', cli+['inspect', str(output),
                      '--report', str(directory/'inspect-report.json')]), 'inspected')
                if not verified['coverage']['source_equality'] or inspected['coverage']['source_equality']:
                    raise AssertionError('Source equality/inspect claim scopes differ')
                row['runtime'] = verified['runtime']
                row['product_reports'] = {'export': exported['status'], 'verify': verified['status'], 'inspect': inspected['status']}
                observer_args = ['qualification.check_snapshot', str(output), '--case', case.get('checker_case', case['name'])]
                if args.resources:
                    observer_args.append('--resources')
                row['independent_output_check'] = success(run(directory, '05-independent-observer', native+observer_args), 'verified')
                if case['name'] in ('shared_coherent', 'maximum_payload'):
                    before_output = sha(output)
                    result = run(directory, '06-no-clobber-output', cli+['export', str(source), '--plan', str(plan),
                                 '--out', str(output), '--report', str(directory/'must-not-publish.json')])
                    if result['exit_code'] != 2 or sha(output) != before_output or (directory/'must-not-publish.json').exists():
                        raise AssertionError('No-clobber failed')
                    protection = {'case': case['name'], 'existing_output_preserved': True}
                    for target_name, target in [('source', source), ('source-hardlink', directory/'source-hardlink.h5')]:
                        if target_name.endswith('hardlink'):
                            os.link(source, target)
                        result = run(directory, '07-protect-'+target_name, cli+['export', str(source), '--plan', str(plan),
                                     '--out', str(target), '--report', str(directory/('must-not-'+target_name+'.json'))])
                        if result['exit_code'] != 2 or sha(target) != row['source_sha256_before']:
                            raise AssertionError('Source destination protection failed')
                        if (directory/('must-not-'+target_name+'.json')).exists():
                            raise AssertionError('Source protection published a report')
                        protection[target_name+'_preserved'] = True
                    report['protection_controls'].append(protection)
                consumer_dir = directory/'consumer'; consumer_dir.mkdir()
                relocated, held = consumer_dir/'relocated.h5', directory/'held-source.bin'
                output.rename(relocated); source.rename(held)
                try:
                    observer_args[1] = str(relocated)
                    observer_args += ['--source-must-be-absent', str(source)]
                    row['relocated_source_unavailable_check'] = success(run(consumer_dir, '08-fresh-relocated-observer', native+observer_args), 'verified')
                    if source.exists() or sha(held) != row['source_sha256_before']:
                        raise AssertionError('Held source or unavailable-path assertion differs')
                    row['source_operation_path_unavailable_during_fresh_consumer'] = True
                    row['held_source_remains_available_without_os_isolation'] = True
                finally:
                    held.rename(source)
                row['output_sha256'] = sha(relocated)
                row['output_bytes'] = relocated.stat().st_size
                row['source_bytes'] = source.stat().st_size
                if list(directory.glob('.h5carry-*')):
                    raise AssertionError('Successful export left private stages behind')
                row['no_stage_artifacts_after_success'] = True
                row['status'] = 'success'
        except Exception as exc:
            row['status'] = 'failed'; row['error'] = type(exc).__name__ + ': ' + str(exc)
            report['failures'].append({'case': case['name'], 'error': row['error']})
            print(json.dumps({'case': case['name'], 'status': 'failed', 'error': row['error']}), flush=True)
        finally:
            if source.exists() and 'source_sha256_before' in row:
                row['source_sha256_after'] = sha(source)
                row['source_unchanged'] = row['source_sha256_after'] == row['source_sha256_before']
                if not row['source_unchanged']:
                    report['failures'].append({'case': case['name'], 'error': 'source changed'})
            save()
        print(json.dumps({'case': case['name'], 'status': row['status'], 'payload_bytes': row.get('selected_payload_bytes')}), flush=True)

    if not args.resources:
        language_source = root/'shared_coherent'/'source.h5'
        if language_source.exists():
            from examples.snapshot_fixture import box, request, maps, whole
            base = request([box('/A')], maps())
            definitions = [('bool', {'start': [True, 1]}), ('negative', {'start': [-1, 1]}),
                           ('past_end', {'stop': [7, 3]}), ('rank', {'start': [1]}),
                           ('step', {'step': [1, 1]}), ('float', {'start': [1., 1]}),
                           ('reversed', {'start': [4, 1], 'stop': [1, 3]})]
            requests = []
            for name, change in definitions:
                value = json.loads(json.dumps(base))
                value['objects'][0]['selection'].update(change)
                requests.append((name, value))
            value = json.loads(json.dumps(base)); value['scale_mappings'] = []
            requests.append(('missing_mapping', value))
            value = json.loads(json.dumps(base)); value['objects'].append(whole('/B'))
            requests.append(('retained_whole_consumer', value))
            value = json.loads(json.dumps(base)); value['unknown'] = 'unsupported'
            requests.append(('unknown_key', value))
            value = json.loads(json.dumps(base)); value['objects'].append(value['objects'][0])
            requests.append(('duplicate_path', value))
            value = json.loads(json.dumps(base)); value['scale_mappings'].append(value['scale_mappings'][0])
            requests.append(('duplicate_mapping', value))
            value = json.loads(json.dumps(base)); value['objects'][0]['selection']['start'][0] = float('nan')
            requests.append(('nonfinite', value))
            directory = root/'language-controls'; directory.mkdir()
            before = sha(language_source)
            for name, value in requests + [('duplicate_json_key', None)]:
                selection_file, plan_file = directory/(name+'.json'), directory/(name+'-plan.json')
                raw = json.dumps(value) if value is not None else json.dumps(base).replace(
                    '\"version\": 1', '\"version\": 1, \"version\": 1', 1)
                selection_file.write_text(raw+'\n')
                result = run(directory, name+'-refusal', cli+['plan', str(language_source),
                             '--selection-json', str(selection_file), '--out', str(plan_file)])
                try:
                    decoded = result['value']
                    if result['exit_code'] != 2 or not isinstance(decoded, dict) or decoded.get('status') != 'incomplete':
                        raise AssertionError('Invalid language request did not refuse')
                    if decoded['diagnostics'][0]['code'] != 'INVALID' or plan_file.exists() or list(directory.glob('.h5carry-*')):
                        raise AssertionError('Language request created a plan/stage or had wrong diagnostic')
                    report['language_controls'].append({'name': name, 'status': 'refused', 'diagnostics': decoded['diagnostics']})
                except Exception as exc:
                    report['failures'].append({'language': name, 'error': str(exc)})
                save()
            if sha(language_source) != before:
                report['failures'].append({'language': 'source-preservation', 'error': 'source changed'})
        for kind, case in [('value', 'shared_coherent'), ('alias', 'aliases_coherent'),
                           ('ref', 'reference_outbound'), ('scale', 'shared_coherent')]:
            if kind == 'ref' and args.contract_variants:
                case = 'reference_outbound_retained_alias'
            row = next((item for item in report['cases'] if item['case'] == case), None)
            if row is None or row['status'] != 'success':
                continue
            directory = root/('mutation-'+kind); directory.mkdir()
            target = directory/'mutated.h5'
            shutil.copyfile(root/case/'consumer'/'relocated.h5', target)
            try:
                success(run(directory, '01-mutate', native+['qualification.check_snapshot', str(target), '--case', row.get('checker_case', case),
                            '--mutate', kind]), 'mutated')
                record = run(directory, '02-independent-observer', native+['qualification.check_snapshot', str(target), '--case', row.get('checker_case', case)])
                if record['exit_code'] == 0 or 'AssertionError' not in record['stderr']:
                    raise AssertionError('Benign output mutation was not detected: ' + json.dumps(record))
                report['mutations'].append({'kind': kind, 'case': case, 'status': 'detected',
                                            'stderr_evidence': record['stderr']})
            except Exception as exc:
                report['failures'].append({'mutation': kind, 'error': str(exc)})
            save()
    for case, initial in frozen_initial.items():
        initial['sha256_after'] = sha(Path(initial['path']))
        initial['unchanged'] = initial['sha256_before'] == initial['sha256_after']
        if not initial['unchanged']:
            report['failures'].append({'frozen_source': case, 'error': 'immutable source changed'})
    report['production_sha256_end'] = product_hashes(package_root)
    report['production_unchanged_during_run'] = report['production_sha256_start'] == report['production_sha256_end']
    report['status'] = 'failed' if report['failures'] else 'passed'
    report['counts'] = {'successes': sum(item['status'] == 'success' for item in report['cases']),
                        'policy_or_resource_refusals': sum(item['status'] == 'expected_refusal' for item in report['cases']),
                        'failures': len(report['failures']), 'mutation_controls': len(report['mutations']),
                        'language_controls': len(report['language_controls'])}
    save()
    print(json.dumps({'status': report['status'], **report['counts'], 'results': str(root/'results.json')}, sort_keys=True))
    return 1 if report['failures'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
