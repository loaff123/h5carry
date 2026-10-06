"""Native-library-free tests for the strict rectangular request contract."""
import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def request():
    return {'format': 'h5carry-selection', 'version': 1,
            'objects': [{'path': '/A', 'selection': {'kind': 'box', 'start': [1, 1], 'stop': [4, 3]}},
                        {'path': '/refs', 'selection': {'kind': 'whole'}}],
            'scale_mappings': [{'consumer': '/A', 'axis': 0, 'scale': '/time', 'mapping': 'index'}],
            'storage_policy': 'fixed-snapshot-v1'}


class SelectionTests(unittest.TestCase):
    def codec(self):
        try:
            return importlib.import_module('h5carry.selection')
        except ModuleNotFoundError:
            self.fail('strict typed selection codec is missing')

    def rejected(self, value, code='INVALID', limits=None):
        from h5carry.model import CarryError
        with self.assertRaises(CarryError) as caught:
            self.codec().validate_selection(value, limits)
        self.assertEqual(caught.exception.code, code)

    def test_canonicalizes_objects_and_mappings_without_mutating_request(self):
        value = request()
        value['objects'].reverse()
        value['scale_mappings'].append({'consumer': '/A', 'axis': 1, 'scale': '/channel', 'mapping': 'index'})
        value['scale_mappings'].reverse()
        before = copy.deepcopy(value)
        canonical = self.codec().validate_selection(value)
        self.assertEqual([o['path'] for o in canonical['objects']], ['/A', '/refs'])
        self.assertEqual([m['axis'] for m in canonical['scale_mappings']], [0, 1])
        self.assertEqual(value, before)
        canonical['objects'][0]['selection']['start'][0] = 0
        self.assertEqual(value, before)

    def test_whole_and_empty_rectangles_are_accepted(self):
        value = request()
        value['objects'][0]['selection']['stop'] = [1, 3]
        self.assertEqual(self.codec().validate_selection(value), value)
        value['objects'] = [{'path': '/', 'selection': {'kind': 'whole'}}]
        value['scale_mappings'] = []
        self.assertEqual(self.codec().validate_selection(value), value)

    def test_exact_keys_versions_and_field_types(self):
        variants = [None, [], {}]
        for key, replacement in [('version', True), ('version', 2), ('format', 'other'),
                                 ('objects', {}), ('objects', []), ('scale_mappings', {}),
                                 ('storage_policy', 'preserve')]:
            value = request(); value[key] = replacement; variants.append(value)
        value = request(); value['unknown'] = 1; variants.append(value)
        value = request(); value['objects'][0]['extra'] = 1; variants.append(value)
        value = request(); value['objects'][0]['selection']['step'] = [1, 1]; variants.append(value)
        value = request(); value['objects'][1]['selection']['start'] = []; variants.append(value)
        value = request(); value['scale_mappings'][0]['extra'] = 1; variants.append(value)
        for value in variants:
            with self.subTest(value=value): self.rejected(value)

    def test_exact_coordinates_rank_and_normalized_paths(self):
        for coordinate in [True, 1.0, -1, 2**63, '1', None]:
            value = request(); value['objects'][0]['selection']['start'][0] = coordinate
            with self.subTest(coordinate=coordinate): self.rejected(value)
        for start, stop in [([], []), ([1], [2, 3]), ([3], [2]), ([0]*33, [1]*33), ((1, 1), [4, 3])]:
            value = request(); value['objects'][0]['selection'].update(start=start, stop=stop)
            with self.subTest(start=start, stop=stop): self.rejected(value)
        value = request(); value['objects'][0]['selection'] = {'kind': 'box', 'start': [2**63-1], 'stop': [2**63-1]}
        self.assertEqual(self.codec().validate_selection(value), value)
        for path in ['relative', '/a/', '/a/../b', '//a', '/a\0', '/\ud800']:
            value = request(); value['objects'][0]['path'] = path
            with self.subTest(path=path): self.rejected(value)

    def test_duplicate_paths_and_mapping_assertions_are_rejected(self):
        value = request(); value['objects'].append(copy.deepcopy(value['objects'][0])); self.rejected(value)
        value = request(); value['scale_mappings'].append(copy.deepcopy(value['scale_mappings'][0])); self.rejected(value)
        for key, bad in [('axis', True), ('axis', -1), ('axis', 32), ('mapping', 'coordinate'), ('consumer', '/bad/'), ('scale', [])]:
            value = request(); value['scale_mappings'][0][key] = bad
            with self.subTest(key=key, bad=bad): self.rejected(value)

    def test_invocation_resource_limits_are_obeyed(self):
        from h5carry.model import Limits
        self.rejected(request(), 'RESOURCE', Limits(max_objects=1))
        value = request(); value['scale_mappings'] *= 2
        self.rejected(value, 'RESOURCE', Limits(max_edges=1))
        self.rejected(request(), 'RESOURCE', Limits(max_plan_bytes=50))
        self.rejected(request(), 'INVALID', Limits(max_rank=1))
        self.rejected(request(), 'RESOURCE', Limits(max_name_bytes=2))

    def test_strict_utf8_json_duplicate_keys_and_decode_bounds(self):
        from h5carry.model import CarryError, Limits
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'selection.json'
            path.write_text(json.dumps(request()))
            self.assertEqual(self.codec().load_selection(path), request())
            payloads = [b'{"format":"h5carry-selection","format":"h5carry-selection"}',
                        b'['*33+b']'*33, b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}',
                        b'{"a":'+b'9'*100+b'}', b'\xff', json.dumps(request()).encode('utf-16'),
                        b'['+b'[],'*500001+b'[]]']
            for payload in payloads:
                path.write_bytes(payload)
                with self.subTest(payload=payload[:30]), self.assertRaises(CarryError): self.codec().load_selection(path)
            path.write_text(json.dumps(request()))
            with self.assertRaises(CarryError) as caught: self.codec().load_selection(path, Limits(max_plan_bytes=50))
            self.assertEqual(caught.exception.code, 'RESOURCE')

    def test_nonregular_input_refuses_without_waiting_for_fifo_writer(self):
        self.codec()
        from h5carry.model import CarryError
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CarryError): self.codec().load_selection(directory)
            with self.assertRaises(CarryError): self.codec().load_selection(Path(directory)/'missing')
            if sys.platform == 'linux':
                path = Path(directory)/'fifo'; os.mkfifo(path)
                import h5carry
                package_root = Path(h5carry.__file__).resolve().parents[1]
                code = ('import sys\nfrom pathlib import Path\n'
                        'root = Path(sys.argv[1]).resolve()\nsys.path.insert(0, str(root))\n'
                        'import h5carry, h5carry.selection\n'
                        'assert Path(h5carry.__file__).resolve().parents[1] == root\n'
                        'assert Path(h5carry.selection.__file__).resolve() == root / "h5carry" / "selection.py"\n'
                        'from h5carry.selection import load_selection\nfrom h5carry.model import CarryError\n'
                        'try: load_selection(sys.argv[2])\nexcept CarryError as e: assert e.code == "INVALID"\n'
                        'else: raise AssertionError("FIFO accepted")')
                result = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(package_root), str(path)],
                                        capture_output=True, text=True, timeout=1)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_schema_resource_is_independent_of_test_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            from unittest.mock import patch
            relocated = str(Path(directory)/'relocated-tests'/'test_codec.py')
            with patch.dict(globals(), {'__file__': relocated}):
                self.test_schema_carries_exact_request_contract()

    def test_schema_carries_exact_request_contract(self):
        from importlib.resources import files
        resource = files('h5carry').joinpath('schemas', 'selection-v1.schema.json')
        self.assertTrue(resource.is_file(), 'packaged selection-v1 schema is missing')
        schema = json.loads(resource.read_text(encoding='utf-8'))
        self.assertEqual(set(schema['required']), set(request()))
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(schema['properties']['version']['const'], 1)


if __name__ == '__main__':
    unittest.main()
