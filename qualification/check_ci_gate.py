"""Standard-library regression checks for the hosted qualification gate itself.

These orchestration checks are separate from the 208 product tests.
"""
import copy
from pathlib import Path
import unittest

from run_ci import (EXPECTED_CASES, EXPECTED_COUNTS, LANGUAGE_CONTROLS,
                    MUTATIONS, NATIVE_CONTROLS, validate_snapshot_result,
                    validate_original_observer, validate_output_location)


def result(mode):
    cases = []
    for name, status in EXPECTED_CASES[mode].items():
        row = {'case': name, 'status': status, 'source_unchanged': True,
               'source_sha256_before': 'source', 'source_sha256_after': 'source'}
        if status == 'success':
            row.update(source_operation_path_unavailable_during_fresh_consumer=True,
                       no_stage_artifacts_after_success=True)
        elif status == 'expected_refusal':
            row['no_plan_output_or_stage_after_refusal'] = True
        cases.append(row)
    return {'counts': EXPECTED_COUNTS[mode].copy(), 'cases': cases,
            'failures': [{'case': 'reference_outbound', 'error': '05-independent-observer: exact retained namespace differs'}] if mode == 'original' else [],
            'status': 'failed' if mode == 'original' else 'passed',
            'mutations': [{'kind': kind, 'case': case, 'status': 'detected'} for kind, case in MUTATIONS[mode]],
            'language_controls': [{'name': name, 'status': 'refused'} for name in sorted(LANGUAGE_CONTROLS)] if mode == 'original' else [],
            'native_controls': NATIVE_CONTROLS.copy(),
            'production_unchanged_during_run': True,
            'production_sha256_start': {'h5carry/__init__.py': 'product'},
            'production_sha256_end': {'h5carry/__init__.py': 'product'},
            'frozen_sources': {}}


class GateChecks(unittest.TestCase):
    def test_exact_three_matrices_are_accepted(self):
        for mode in EXPECTED_CASES:
            validate_snapshot_result(result(mode), mode)

    def reject(self, change, mode='original'):
        record = result(mode)
        change(record)
        with self.assertRaises(AssertionError):
            validate_snapshot_result(record, mode)

    def test_wrong_count_is_rejected(self):
        self.reject(lambda r: r['counts'].update(successes=7))

    def test_duplicate_or_renamed_case_is_rejected(self):
        self.reject(lambda r: r['cases'][0].update(case='shared_coherent'))

    def test_wrong_known_failure_is_rejected(self):
        self.reject(lambda r: r['failures'][0].update(case='plain_rectangle'))

    def test_known_failure_cannot_be_counted_as_pass(self):
        self.reject(lambda r: r.update(status='passed'))

    def test_wrong_failure_stage_is_rejected(self):
        self.reject(lambda r: r['failures'][0].update(error='CLI export failed'))

    def test_product_or_source_changes_are_rejected(self):
        self.reject(lambda r: r['production_sha256_end'].update(other='changed'))
        self.reject(lambda r: r['cases'][0].update(source_sha256_after='changed'))

    def test_native_limit_change_is_rejected(self):
        self.reject(lambda r: r['native_controls'].update(memory_bytes=4 * 1024**3))

    def test_control_identity_change_is_rejected(self):
        self.reject(lambda r: r['mutations'][0].update(kind='ref'))
        self.reject(lambda r: r['language_controls'][0].update(name='duplicate_path'))

    def test_refusal_or_relocation_claim_missing_is_rejected(self):
        self.reject(lambda r: r['cases'][0].update(source_operation_path_unavailable_during_fresh_consumer=False))
        self.reject(lambda r: r['cases'][-1].update(no_plan_output_or_stage_after_refusal=False), 'resources')

    def test_only_exact_observer_negative_is_accepted(self):
        record = {'tag': '05-independent-observer', 'exit_code': 2, 'stdout': '',
                  'stderr': 'AssertionError: exact retained namespace differs\n' +
                            repr({'ok': False, 'error': {'code': 'ERROR',
                                 'message': 'Native worker exited unexpectedly with status 1'}}) + '\n'}
        validate_original_observer(record)
        for key, value in [('exit_code', 0), ('stderr', 'RESOURCE'), ('tag', '02-cli-export'),
                           ('stderr', record['stderr'] + "{'ok': False, 'error': {'code': 'RESOURCE', 'message': 'timeout'}}\n"),
                           ('stderr', record['stderr'].replace('status 1', 'status -11'))]:
            changed = copy.deepcopy(record)
            changed[key] = value
            with self.assertRaises(AssertionError):
                validate_original_observer(changed)

    def test_output_cannot_recursively_copy_inputs(self):
        source = Path('/tmp/qualification-source')
        for name in ('src', 'tests', 'examples', 'docs', 'qualification', 'evidence', '.git'):
            with self.assertRaises(ValueError):
                validate_output_location(source, source / name / 'new-run')
        validate_output_location(source, source / 'qualification/runs/new-run')
        validate_output_location(source, Path('/tmp/separate-evidence'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
