"""Pure v2 envelope tests, with native transport replaced only at its boundary."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def rectangle_plan():
    from h5carry.model import Limits
    source_metadata = {'attributes': [], 'dtype': {'kind': 'fixed', 'str': '<f4', 'encoding': None},
                       'shape': [6, 4], 'creation': {'layout': 'chunked', 'chunks': [4, 3],
                       'maxshape': [None, 4], 'filters': [], 'fill': '00000000'},
                       'scale_name': None, 'labels': ['', ''], 'payload_sha256': None}
    output_metadata = copy.deepcopy(source_metadata)
    output_metadata.update(shape=[3, 2], payload_sha256='1'*64)
    output_metadata['creation'].update(chunks=[3, 2], maxshape=[3, 2])
    root = {'id': '/', 'kind': 'group', 'metadata': {'attributes': []}}
    selection = {'kind': 'box', 'start': [1, 1], 'stop': [4, 3]}
    return {'format': 'h5carry-plan', 'version': 2, 'source': {'sha256': '0'*64, 'size': 123},
            'request': {'format': 'h5carry-selection', 'version': 1,
                        'objects': [{'path': '/A', 'selection': copy.deepcopy(selection)}],
                        'scale_mappings': [], 'storage_policy': 'fixed-snapshot-v1'},
            'limits': Limits().to_dict(),
            'graph': {'objects': [root, {'id': '/A', 'kind': 'dataset', 'metadata': output_metadata}],
                      'links': [{'path': '/A', 'kind': 'hard', 'target': '/A'}],
                      'references': [], 'scales': [], 'payload_bytes': 24,
                      'reasons': {'/': ['ancestor'], '/A': ['selected rectangle']}},
            'source_objects': [copy.deepcopy(root), {'id': '/A', 'kind': 'dataset', 'metadata': source_metadata}],
            'selections': [{'id': '/', 'selection': {'kind': 'whole'}}, {'id': '/A', 'selection': selection}],
            'transformations': [{'id': '/A', 'source_shape': [6, 4], 'output_shape': [3, 2],
                                 'source_maxshape': [None, 4], 'output_maxshape': [3, 2],
                                 'source_chunks': [4, 3], 'output_chunks': [3, 2]}]}


def whole_plan(shape=None):
    p = rectangle_plan()
    shape = [6, 4] if shape is None else shape
    source = p['source_objects'][1]['metadata']
    source['shape'] = shape
    source['labels'] = ['']*len(shape)
    source['creation'].update(layout='contiguous', chunks=None, maxshape=shape)
    p['graph']['objects'][1]['metadata'] = copy.deepcopy(source)
    p['graph']['objects'][1]['metadata']['payload_sha256'] = '1'*64
    import math
    p['graph']['payload_bytes'] = math.prod(shape)*4
    p['request']['objects'][0]['selection'] = {'kind': 'whole'}
    p['selections'][1]['selection'] = {'kind': 'whole'}
    p['transformations'] = []
    return p


class PlanV2Tests(unittest.TestCase):
    def codec(self):
        try:
            return importlib.import_module('h5carry.plan_v2')
        except ModuleNotFoundError:
            self.fail('separate strict v2 plan codec is missing')

    def rejected(self, value, code='INVALID', limits=None):
        from h5carry.model import CarryError
        with self.assertRaises(CarryError) as caught: self.codec().validate_plan_v2(value, limits)
        self.assertEqual(caught.exception.code, code)

    def test_accepts_rectangular_and_unchanged_envelopes(self):
        for value in [rectangle_plan(), whole_plan(), whole_plan([]), whole_plan([0, 4])]:
            with self.subTest(value=value): self.assertEqual(self.codec().validate_plan_v2(value), value)

    def test_separate_v1_decoder_and_dispatch_preserve_v1(self):
        from h5carry.plan import validate_plan
        from h5carry.model import CarryError
        from tests.test_plan import empty_plan
        self.codec()
        from h5carry.plan import _validate_plan_v1
        self.assertEqual(validate_plan(empty_plan()), empty_plan())
        self.assertEqual(validate_plan(rectangle_plan()), rectangle_plan())
        with self.assertRaises(CarryError): _validate_plan_v1(rectangle_plan())
        value = empty_plan(); value['request'] = rectangle_plan()['request']
        with self.assertRaises(CarryError): validate_plan(value)

    def test_v2_roundtrip_uses_existing_strict_loader_and_no_clobber(self):
        self.codec()
        from h5carry.plan import load_plan, save_plan
        from h5carry.model import CarryError
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'plan.json'
            save_plan(rectangle_plan(), path)
            self.assertEqual(load_plan(path), rectangle_plan())
            with self.assertRaises(CarryError): save_plan(rectangle_plan(), path)

    def test_exact_v2_keys_versions_and_source_fingerprint(self):
        variants = [None, {}, []]
        for key, value in [('version', True), ('version', 1), ('format', 'other'),
                           ('source_objects', {}), ('selections', {}), ('transformations', {})]:
            p = rectangle_plan(); p[key] = value; variants.append(p)
        p = rectangle_plan(); p['unknown'] = 1; variants.append(p)
        p = rectangle_plan(); del p['source_objects']; variants.append(p)
        p = rectangle_plan(); p['source']['size'] = True; variants.append(p)
        p = rectangle_plan(); p['source']['path'] = 'private'; variants.append(p)
        p = rectangle_plan(); p['source']['sha256'] = 'A'*64; variants.append(p)
        for p in variants:
            with self.subTest(p=p): self.rejected(p)

    def test_all_source_and_selection_ids_are_sorted_unique_and_complete(self):
        variants = []
        for field in ['source_objects', 'selections']:
            p = rectangle_plan(); p[field].reverse(); variants.append(p)
            p = rectangle_plan(); p[field].append(copy.deepcopy(p[field][0])); variants.append(p)
            p = rectangle_plan(); p[field].pop(); variants.append(p)
            p = rectangle_plan(); p[field][1]['id'] = '/other'; variants.append(p)
            p = rectangle_plan(); p[field][1]['extra'] = 1; variants.append(p)
        p = rectangle_plan(); p['source_objects'][1]['kind'] = 'group'; variants.append(p)
        p = rectangle_plan(); p['source_objects'][1]['metadata']['payload_sha256'] = '2'*64; variants.append(p)
        for p in variants:
            with self.subTest(p=p): self.rejected(p)

    def test_source_extent_is_not_charged_as_selected_payload(self):
        p = rectangle_plan()
        p['source_objects'][1]['metadata']['shape'] = [2**40, 4]
        p['transformations'][0]['source_shape'] = [2**40, 4]
        p['limits']['max_payload_bytes'] = 24
        self.assertEqual(self.codec().validate_plan_v2(p), p)
        p['graph']['payload_bytes'] = 23
        self.rejected(p)
        p = rectangle_plan(); p['limits']['max_payload_bytes'] = 23
        self.rejected(p, 'RESOURCE')

    def test_crop_bounds_normalization_and_primitive_only_rules(self):
        variants = []
        for selection in [{'kind': 'box', 'start': [0], 'stop': [1]},
                          {'kind': 'box', 'start': [1, 1], 'stop': [7, 3]},
                          {'kind': 'box', 'start': [4, 1], 'stop': [3, 3]},
                          {'kind': 'box', 'start': [0, 0], 'stop': [6, 4]},
                          {'kind': 'box', 'start': [True, 1], 'stop': [4, 3]},
                          {'kind': 'box', 'start': [1, 1], 'stop': [4, 3], 'step': [1, 1]}]:
            p = rectangle_plan(); p['selections'][1]['selection'] = selection; variants.append(p)
        p = rectangle_plan(); p['selections'][0]['selection'] = {'kind': 'box', 'start': [0], 'stop': [1]}; variants.append(p)
        p = rectangle_plan(); p['source_objects'][1]['metadata']['shape'] = None; variants.append(p)
        p = rectangle_plan(); p['source_objects'][1]['metadata']['dtype'] = {'kind': 'reference', 'str': '|O', 'encoding': None}; variants.append(p)
        for p in variants:
            with self.subTest(p=p): self.rejected(p)

    def test_transformations_cover_only_proper_crops_and_exact_derived_storage(self):
        variants = []
        p = rectangle_plan(); p['transformations'] = []; variants.append(p)
        p = rectangle_plan(); p['transformations'] *= 2; variants.append(p)
        p = rectangle_plan(); p['transformations'][0]['id'] = '/'; variants.append(p)
        p = rectangle_plan(); p['transformations'][0]['extra'] = 1; variants.append(p)
        for field, bad in [('source_shape', [7, 4]), ('output_shape', [2, 3]), ('source_maxshape', [6, 4]),
                           ('output_maxshape', [None, 2]), ('source_chunks', [3, 3]), ('output_chunks', [4, 3])]:
            p = rectangle_plan(); p['transformations'][0][field] = bad; variants.append(p)
        for field, bad in [('shape', [2, 3]), ('labels', ['']), ('dtype', {'kind': 'fixed', 'str': '<f8', 'encoding': None})]:
            p = rectangle_plan(); p['graph']['objects'][1]['metadata'][field] = bad; variants.append(p)
        for field, bad in [('maxshape', [None, 2]), ('chunks', [4, 3]), ('layout', 'contiguous'), ('fill', 'ffffffff')]:
            p = rectangle_plan(); p['graph']['objects'][1]['metadata']['creation'][field] = bad; variants.append(p)
        p = whole_plan(); p['transformations'] = rectangle_plan()['transformations']; variants.append(p)
        p = whole_plan(); p['graph']['objects'][1]['metadata']['creation']['maxshape'] = [None, 4]; variants.append(p)
        for p in variants:
            with self.subTest(p=p): self.rejected(p)

    def test_empty_crop_retains_rank_fixed_maximum_and_clamped_chunks(self):
        p = rectangle_plan()
        p['request']['objects'][0]['selection']['stop'] = [1, 3]
        p['selections'][1]['selection']['stop'] = [1, 3]
        p['graph']['objects'][1]['metadata']['shape'] = [0, 2]
        p['graph']['objects'][1]['metadata']['creation'].update(maxshape=[0, 2], chunks=[1, 2])
        p['graph']['payload_bytes'] = 0
        p['transformations'][0].update(output_shape=[0, 2], output_maxshape=[0, 2], output_chunks=[1, 2])
        self.assertEqual(self.codec().validate_plan_v2(p), p)

    def test_invocation_ceilings_original_request_and_serialized_budget(self):
        from h5carry.model import Limits
        self.rejected(rectangle_plan(), 'RESOURCE', Limits(max_payload_bytes=23))
        p = rectangle_plan(); p['request']['objects'].append({'path': '/Z', 'selection': {'kind': 'whole'}})
        p['request']['objects'].reverse(); self.rejected(p)
        p = rectangle_plan(); p['limits']['max_plan_bytes'] = 100
        self.rejected(p, 'RESOURCE')
        p = rectangle_plan(); p['graph']['objects'][1]['metadata']['payload_sha256'] = None
        self.rejected(p)

    def test_make_plan_wraps_native_result_and_canonical_original_request(self):
        codec = self.codec()
        p = rectangle_plan()
        result = {key: p[key] for key in ['graph', 'source_objects', 'selections', 'transformations']}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.h5'; source.write_bytes(b'not parsed by the pure codec test')
            from h5carry.model import fingerprint
            p['source'] = fingerprint(source)
            with patch('h5carry.supervisor.run_native', return_value={'ok': True, 'result': result}) as native:
                actual = codec.make_selection_plan(source, p['request'])
            self.assertEqual(actual, p)
            self.assertEqual(native.call_args.args[0], {'operation': 'plan_v2', 'source': str(source),
                                                       'request': p['request'], 'limits': p['limits']})

    def test_make_plan_detects_source_mutation_and_propagates_native_refusal(self):
        codec = self.codec()
        from h5carry.model import CarryError
        p = rectangle_plan(); result = {key: p[key] for key in ['graph', 'source_objects', 'selections', 'transformations']}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.h5'; source.write_bytes(b'original')
            def mutate(_):
                source.write_bytes(b'changed')
                return {'ok': True, 'result': result}
            with patch('h5carry.supervisor.run_native', side_effect=mutate), self.assertRaises(CarryError) as caught:
                codec.make_selection_plan(source, p['request'])
            self.assertEqual(caught.exception.code, 'SOURCE_CHANGED')
            with patch('h5carry.supervisor.run_native', return_value={'ok': False, 'error': {'code': 'UNSUPPORTED', 'message': 'outside profile', 'path': '/A'}}):
                with self.assertRaises(CarryError) as caught: codec.make_selection_plan(source, p['request'])
            self.assertEqual(caught.exception.to_dict(), {'code': 'UNSUPPORTED', 'message': 'outside profile', 'path': '/A'})

    def test_transformation_shape_scalars_have_exact_integer_types(self):
        for field in ['source_shape', 'output_shape', 'source_maxshape', 'output_maxshape', 'source_chunks', 'output_chunks']:
            p = rectangle_plan(); index = 1 if field == 'source_maxshape' else 0
            p['transformations'][0][field][index] = float(p['transformations'][0][field][index])
            with self.subTest(field=field): self.rejected(p)
        p = rectangle_plan(); p['transformations'][0]['source_chunks'] = (4, 3)
        self.rejected(p)
        p = rectangle_plan(); p['transformations'][0]['output_shape'] = [True, 2]
        self.rejected(p)

    def test_v2_loader_requires_utf8_while_v1_decoder_stays_unchanged(self):
        self.codec()
        from h5carry.plan import load_plan
        from h5carry.model import CarryError
        from tests.test_plan import empty_plan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'plan.json'
            path.write_bytes(json.dumps(rectangle_plan()).encode('utf-16'))
            with self.assertRaises(CarryError) as caught: load_plan(path)
            self.assertEqual(caught.exception.code, 'INVALID')
            path.write_bytes(json.dumps(empty_plan()).encode('utf-16'))
            self.assertEqual(load_plan(path), empty_plan())

    def test_all_v2_creation_and_metadata_types_remain_strict(self):
        variants = []
        for field, bad in [('chunks', [0, 3]), ('maxshape', [None]), ('layout', 'compact'), ('fill', '00')]:
            p = rectangle_plan(); p['source_objects'][1]['metadata']['creation'][field] = bad; variants.append(p)
        for filters in [[{'id': 1, 'flags': True, 'values': [1]}],
                        [{'id': 1, 'flags': 0, 'values': [10]}],
                        [{'id': 2, 'flags': 0, 'values': [8]}],
                        [{'id': 3, 'flags': 0, 'values': [1]}],
                        [{'id': 1, 'flags': 0, 'values': [1]}]*2]:
            p = rectangle_plan(); p['source_objects'][1]['metadata']['creation']['filters'] = filters; variants.append(p)
        p = rectangle_plan(); p['source_objects'][1]['metadata']['labels'] = ['']; variants.append(p)
        p = rectangle_plan(); p['source_objects'][1]['metadata']['shape'] = [True, 4]; variants.append(p)
        for p in variants:
            with self.subTest(p=p): self.rejected(p)

    def test_null_and_reference_whole_descriptors_preserve_original_profile(self):
        p = whole_plan()
        source = p['source_objects'][1]['metadata']; output = p['graph']['objects'][1]['metadata']
        for metadata in [source, output]:
            metadata.update(shape=None, labels=[])
            metadata['creation']['maxshape'] = None
        p['graph']['payload_bytes'] = 0
        self.assertEqual(self.codec().validate_plan_v2(p), p)
        p = whole_plan([1])
        for metadata in [p['source_objects'][1]['metadata'], p['graph']['objects'][1]['metadata']]:
            metadata.update(dtype={'kind': 'reference', 'str': '|O', 'encoding': None}, payload_sha256=None)
            metadata['creation']['fill'] = None
        p['graph']['payload_bytes'] = 8
        p['graph']['references'] = [{'owner': '/A', 'attribute': None, 'index': 0, 'target': None}]
        self.assertEqual(self.codec().validate_plan_v2(p), p)

    def test_modules_remain_native_library_free(self):
        import h5carry, subprocess, sys
        package_root = Path(h5carry.__file__).resolve().parents[1]
        code = ('import sys\nfrom pathlib import Path\n'
                'root = Path(sys.argv[1]).resolve()\nsys.path.insert(0, str(root))\n'
                'import h5carry, h5carry.selection, h5carry.plan_v2\n'
                'assert Path(h5carry.__file__).resolve().parents[1] == root\n'
                'assert Path(h5carry.selection.__file__).resolve() == root / "h5carry" / "selection.py"\n'
                'assert Path(h5carry.plan_v2.__file__).resolve() == root / "h5carry" / "plan_v2.py"\n'
                'assert "h5py" not in sys.modules and "numpy" not in sys.modules')
        result = subprocess.run([sys.executable, '-I', '-S', '-c', code, str(package_root)],
                                capture_output=True, text=True, timeout=2)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_v2_graph_relationship_and_reason_lists_are_canonical(self):
        p = whole_plan()
        attributes = [{'name': name, 'dtype': {'kind': 'reference', 'str': '|O', 'encoding': None},
                       'shape': [], 'value': None} for name in ['a', 'b']]
        for root in [p['source_objects'][0], p['graph']['objects'][0]]:
            root['metadata']['attributes'] = copy.deepcopy(attributes)
        p['graph']['references'] = [{'owner': '/', 'attribute': name, 'index': 0, 'target': None} for name in ['a', 'b']]
        self.assertEqual(self.codec().validate_plan_v2(p), p)
        p['graph']['references'].reverse(); self.rejected(p)
        p = whole_plan()
        scale = whole_plan([6])['source_objects'][1]
        scale['id'] = '/time'; scale['metadata']['scale_name'] = 'time'
        p['source_objects'].append(scale)
        output_scale = copy.deepcopy(scale); output_scale['metadata']['payload_sha256'] = '1'*64
        p['graph']['objects'].append(output_scale)
        p['graph']['links'].append({'path': '/time', 'kind': 'hard', 'target': '/time'})
        p['graph']['reasons']['/time'] = ['scale dependency']
        p['graph']['payload_bytes'] += 24
        p['selections'].append({'id': '/time', 'selection': {'kind': 'whole'}})
        p['graph']['scales'] = [{'consumer': '/A', 'axis': axis, 'scale': '/time'} for axis in [0, 1]]
        self.assertEqual(self.codec().validate_plan_v2(p), p)
        p['graph']['scales'].reverse(); self.rejected(p)
        p = whole_plan(); p['graph']['reasons']['/'] = ['z', 'a']; self.rejected(p)
        p = whole_plan(); p['graph']['reasons']['/'] = ['ancestor', 'ancestor']; self.rejected(p)

    def test_schema_transformations_require_finite_fixed_output_maximum(self):
        from importlib.resources import files
        schema = json.loads(files('h5carry').joinpath('schemas', 'plan-v2.schema.json').read_text(encoding='utf-8'))
        properties = schema['properties']['transformations']['items']['properties']
        integer = {'type': 'integer', 'minimum': 0, 'maximum': 2**63-1}
        self.assertEqual(properties['output_maxshape'], {'type': 'array', 'items': integer,
                                                       'minItems': 1, 'maxItems': 32})
        self.assertEqual(properties['source_maxshape'], {'type': 'array',
                                                       'items': {'anyOf': [{'type': 'null'}, integer]},
                                                       'minItems': 1, 'maxItems': 32})

    def test_schema_resource_is_independent_of_test_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            from unittest.mock import patch
            relocated = str(Path(directory)/'relocated-tests'/'test_codec.py')
            with patch.dict(globals(), {'__file__': relocated}):
                self.test_schema_defines_exact_v2_envelope()

    def test_schema_defines_exact_v2_envelope(self):
        from importlib.resources import files
        resource = files('h5carry').joinpath('schemas', 'plan-v2.schema.json')
        self.assertTrue(resource.is_file(), 'packaged plan-v2 schema is missing')
        schema = json.loads(resource.read_text(encoding='utf-8'))
        self.assertEqual(set(schema['required']), set(rectangle_plan()))
        self.assertFalse(schema['additionalProperties'])
        self.assertEqual(schema['properties']['version']['const'], 2)


if __name__ == '__main__':
    unittest.main()
