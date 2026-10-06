"""Minimal v1 native-read/DCPL regression controls, generated in bounded children.

Removing the pre-read admission checks must call a forbidden native payload read;
removing independent DCPL/chunk-option equality must falsely verify the negative
controls. Original release fixtures and tests are deliberately left unchanged.
"""
import ctypes
import hashlib
import math
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from tests.test_profile import NativeFixtureMixin


class V1ReliabilityTests(NativeFixtureMixin, unittest.TestCase):
    def create(self, file, name='data', shape=(8,), chunked=False,
               reference=False, fill_time=None, populated=True, properties=None,
               user_fill=True):
        h5, np = self.h5, self.np
        dtype = h5.ref_dtype if reference else np.dtype('<i4')
        dcpl = h5.h5p.create(h5.h5p.DATASET_CREATE)
        if chunked:
            dcpl.set_chunk(tuple(max(1, min(4, n)) for n in shape))
        dcpl.set_fill_time(h5.h5d.FILL_TIME_IFSET if fill_time is None else fill_time)
        if not reference and user_fill:
            dcpl.set_fill_value(np.array(-7, dtype=dtype))
        if properties is not None:
            properties(dcpl)
        if shape is None:
            space = h5.h5s.create(h5.h5s.NULL)
        elif not shape:
            space = h5.h5s.create(h5.h5s.SCALAR)
        else:
            space = h5.h5s.create_simple(shape, shape)
        ds = h5.Dataset(h5.h5d.create(file.id, name.encode(),
                        h5.h5t.py_create(dtype, logical=True), space, dcpl=dcpl))
        count = 0 if shape is None else math.prod(shape)
        if populated and count:
            if reference:
                values = np.empty(shape, dtype=dtype)
                values.flat[:] = [h5.Reference()] * count
            else:
                values = np.arange(count, dtype=dtype).reshape(shape)
            ds[()] = values
        return ds

    def manual_plan(self, source, reference=False, populated=True):
        """Literal whole-object plan; NEVER values are never read to prepare it."""
        from h5carry.model import fingerprint
        with self.h5.File(source, 'r') as src:
            ds = src['data']
            root = self.p.describe_object(src['/'], self.limits, hash_payload=False)
            meta = self.p.describe_object(ds, self.limits, hash_payload=False)
            count = 0 if ds.shape is None else math.prod(ds.shape)
            if not reference:
                raw = (self.np.arange(count, dtype=ds.dtype).tobytes() if populated
                       else bytes(count * ds.dtype.itemsize))
                meta['payload_sha256'] = hashlib.sha256(raw).hexdigest()
            refs = [{'owner': '/data', 'attribute': None, 'index': i, 'target': None}
                    for i in range(count)] if reference else []
            graph = {'objects': [{'id': '/', 'kind': 'group', 'metadata': root},
                                 {'id': '/data', 'kind': 'dataset', 'metadata': meta}],
                     'links': [{'path': '/data', 'kind': 'hard', 'target': '/data'}],
                     'references': refs, 'scales': [],
                     'payload_bytes': count * ds.dtype.itemsize,
                     'reasons': {'/': ['manual fixture'], '/data': ['manual fixture']}}
        return {'format': 'h5carry-plan', 'version': 1, 'source': fingerprint(source),
                'selections': ['/data'], 'limits': self.limits.to_dict(), 'graph': graph}

    def forbidden_reads(self):
        """Instrument the boundary without performing undefined native reads."""
        original = self.h5.Dataset.__getitem__
        calls = []
        def observed(ds, selection, *args, **kwargs):
            if (ds.shape is not None and math.prod(ds.shape) and
                    ds.id.get_create_plist().get_fill_time() == self.h5.h5d.FILL_TIME_NEVER):
                calls.append((ds.name, selection))
                raise AssertionError('FILL_TIME_NEVER payload reached Dataset.__getitem__')
            return original(ds, selection, *args, **kwargs)
        return patch.object(self.h5.Dataset, '__getitem__', observed), calls

    def assert_never_error(self, exc):
        self.assertEqual(exc.code, 'UNSUPPORTED')
        self.assertIn('FILL_TIME_NEVER', exc.message)
        self.assertEqual(exc.path, '/data')

    def assert_never_report(self, report):
        self.assertEqual(report['status'], 'incomplete', report)
        self.assertFalse(report['coverage']['source_equality'], report)
        self.assertEqual(report['diagnostics'][0]['code'], 'UNSUPPORTED', report)
        self.assertIn('FILL_TIME_NEVER', report['diagnostics'][0]['message'])
        self.assertEqual(report['diagnostics'][0]['path'], '/data')

    def test_profile_payload_digest_refuses_sparse_populated_scalar_and_reference(self):
        for i, (shape, chunked, reference, populated) in enumerate([
            ((8,), False, False, False), ((8,), True, False, False),
            ((8,), True, False, True), ((), False, False, False),
            ((), False, False, True), ((8,), True, True, False),
            ((8,), True, True, True), ((), False, True, True)]):
            with self.subTest(shape=shape, chunked=chunked, reference=reference, populated=populated):
                ds = self.create(self.f, str(i), shape, chunked, reference,
                                 self.h5.h5d.FILL_TIME_NEVER, populated)
                instrument, calls = self.forbidden_reads()
                with instrument, self.assertRaises(self.error) as cm:
                    self.p.payload_digest(ds, self.limits)
                self.assertEqual(cm.exception.code, 'UNSUPPORTED')
                self.assertIn('FILL_TIME_NEVER', cm.exception.message)
                self.assertEqual(calls, [])

    def test_hashing_describe_refuses_but_metadata_only_describe_stays_permissive(self):
        ds = self.create(self.f, fill_time=self.h5.h5d.FILL_TIME_NEVER, populated=False)
        instrument, calls = self.forbidden_reads()
        with instrument:
            metadata = self.p.describe_object(ds, self.limits, hash_payload=False)
            self.assertIsNone(metadata['payload_sha256'])
            self.assertEqual(self.p.dataset_creation(ds), metadata['creation'])
            with self.assertRaises(self.error) as cm:
                self.p.describe_object(ds, self.limits)
        self.assert_never_error(cm.exception)
        self.assertEqual(calls, [])

    def test_scan_selected_never_refuses_before_fixed_or_reference_payload_read(self):
        from h5carry.scan import make_plan_native
        self.f.close()
        for i, (reference, populated, shape) in enumerate([
            (False, False, (8,)), (False, True, (8,)), (False, True, ()),
            (True, False, (8,)), (True, True, (8,)), (True, True, ())]):
            with self.subTest(reference=reference, populated=populated, shape=shape):
                source = str(Path(self.tmp.name) / f'scan-{i}.h5')
                with self.h5.File(source, 'w') as file:
                    self.create(file, shape=shape, reference=reference,
                                fill_time=self.h5.h5d.FILL_TIME_NEVER, populated=populated)
                instrument, calls = self.forbidden_reads()
                with instrument, self.assertRaises(self.error) as cm:
                    make_plan_native(source, ['/data'], self.limits)
                self.assert_never_error(cm.exception)
                self.assertEqual(calls, [])

    def test_unselected_never_reference_inventory_refuses_but_numeric_metadata_does_not(self):
        from h5carry.scan import make_plan_native
        self.f.close()
        for reference in (False, True):
            with self.subTest(reference=reference):
                source = str(Path(self.tmp.name) / f'unselected-{reference}.h5')
                with self.h5.File(source, 'w') as file:
                    self.create(file, reference=reference, fill_time=self.h5.h5d.FILL_TIME_NEVER)
                    self.create(file, 'selected')
                instrument, calls = self.forbidden_reads()
                with instrument:
                    if reference:
                        with self.assertRaises(self.error) as cm:
                            make_plan_native(source, ['/selected'], self.limits)
                        self.assert_never_error(cm.exception)
                    else:
                        graph = make_plan_native(source, ['/selected'], self.limits)
                        self.assertEqual([x['id'] for x in graph['objects']], ['/', '/selected'])
                self.assertEqual(calls, [])

    def test_writer_never_preflight_leaves_staging_untouched_for_fixed_and_references(self):
        from h5carry.model import fingerprint
        from h5carry.write import write_staging
        self.f.close()
        for reference in (False, True):
            with self.subTest(reference=reference):
                source = str(Path(self.tmp.name) / f'write-{reference}.h5')
                stage = Path(self.tmp.name) / f'stage-{reference}.h5'
                with self.h5.File(source, 'w') as file:
                    self.create(file, reference=reference, fill_time=self.h5.h5d.FILL_TIME_NEVER)
                plan = self.manual_plan(source, reference=reference)
                before = fingerprint(source)
                stage.write_bytes(b'caller-owned staging sentinel')
                instrument, calls = self.forbidden_reads()
                with instrument, self.assertRaises(self.error) as cm:
                    write_staging(source, plan['graph'], str(stage), self.limits)
                self.assert_never_error(cm.exception)
                self.assertEqual(calls, [])
                self.assertEqual(stage.read_bytes(), b'caller-owned staging sentinel')
                self.assertEqual(fingerprint(source), before)

    def test_source_verification_and_inspection_independently_refuse_never_reads(self):
        from h5carry.verify import inspect_output, verify_export
        self.f.close()
        for i, (reference, populated, shape) in enumerate([
            (False, False, (8,)), (False, True, (8,)), (False, True, ()),
            (True, False, (8,)), (True, True, (8,)), (True, True, ())]):
            with self.subTest(reference=reference, populated=populated, shape=shape):
                source = str(Path(self.tmp.name) / f'verify-{i}.h5')
                with self.h5.File(source, 'w') as file:
                    self.create(file, shape=shape, reference=reference,
                                fill_time=self.h5.h5d.FILL_TIME_NEVER, populated=populated)
                plan = self.manual_plan(source, reference=reference, populated=populated)
                instrument, calls = self.forbidden_reads()
                with instrument, patch('h5carry.profile.payload_digest', side_effect=AssertionError('shared digest')), \
                        patch('h5carry.profile.admit_payload_read', side_effect=AssertionError('shared read guard')):
                    self.assert_never_report(verify_export(source, source, plan, self.limits))
                    self.assert_never_report(inspect_output(source, self.limits))
                self.assertEqual(calls, [])

    def test_empty_and_null_never_pass_scan_write_verify_inspect_without_reads(self):
        from h5carry.model import fingerprint
        from h5carry.scan import make_plan_native
        from h5carry.write import write_staging
        from h5carry.verify import verify_export, inspect_output
        self.f.close()
        for i, (shape, chunked, reference) in enumerate([
            ((0, 3), False, False), ((3, 0), True, False), (None, False, False),
            ((0,), True, True), (None, False, True)]):
            with self.subTest(shape=shape, chunked=chunked, reference=reference):
                source = str(Path(self.tmp.name) / f'empty-{i}.h5')
                output = str(Path(self.tmp.name) / f'empty-out-{i}.h5')
                with self.h5.File(source, 'w') as file:
                    ds = self.create(file, shape=shape, chunked=chunked, reference=reference,
                                     fill_time=self.h5.h5d.FILL_TIME_NEVER)
                    self.assertEqual(self.p.payload_digest(ds, self.limits), hashlib.sha256(b'').hexdigest())
                with patch.object(self.h5.Dataset, '__getitem__', side_effect=AssertionError('empty payload read')):
                    graph = make_plan_native(source, ['/data'], self.limits)
                    write_staging(source, graph, output, self.limits)
                    plan = {'source': fingerprint(source), 'selections': ['/data'], 'graph': graph}
                    checked = verify_export(source, output, plan, self.limits)
                    self.assertEqual(checked['status'], 'verified', checked)
                    self.assertTrue(checked['coverage']['source_equality'])
                    self.assertEqual(checked['coverage']['data_blocks'], 0)
                    self.assertEqual(inspect_output(output, self.limits)['status'], 'inspected')

    def test_ordinary_fill_controls_preserve_native_creation_and_values(self):
        from h5carry.model import fingerprint
        from h5carry.scan import make_plan_native
        from h5carry.write import write_staging
        from h5carry.verify import verify_export
        self.f.close()
        for i, (chunked, fill_time, populated) in enumerate([
            (chunked, fill_time, populated) for chunked in (False, True)
            for fill_time in (self.h5.h5d.FILL_TIME_IFSET, self.h5.h5d.FILL_TIME_ALLOC)
            for populated in (False, True)]):
            with self.subTest(chunked=chunked, fill_time=fill_time, populated=populated):
                source = str(Path(self.tmp.name) / f'control-{i}.h5')
                output = str(Path(self.tmp.name) / f'control-out-{i}.h5')
                with self.h5.File(source, 'w') as file:
                    self.create(file, chunked=chunked, fill_time=fill_time, populated=populated)
                graph = make_plan_native(source, ['/data'], self.limits)
                write_staging(source, graph, output, self.limits)
                checked = verify_export(source, output, {'source': fingerprint(source),
                                        'selections': ['/data'], 'graph': graph}, self.limits)
                self.assertEqual(checked['status'], 'verified', checked)
                with self.h5.File(source, 'r') as src, self.h5.File(output, 'r') as dst:
                    self.assertTrue(src['data'].id.get_create_plist().equal(dst['data'].id.get_create_plist()))
                    self.assertTrue(src['data'].id.get_type().equal(dst['data'].id.get_type()))
                    self.assertEqual(dst['data'][()].tolist(), list(range(8)) if populated else [-7] * 8)

    def pair(self, tag, mutation, chunked=True, reference=False, shape=(8,), default_fill=False):
        from h5carry.plan import make_plan
        source = str(Path(self.tmp.name) / f'pair-{tag}-source.h5')
        output = str(Path(self.tmp.name) / f'pair-{tag}-output.h5')
        for path, mutate in ((source, None), (output, mutation)):
            with self.h5.File(path, 'w', libver='latest') as file:
                def properties(dcpl):
                    if mutate is not None:
                        mutate(dcpl)
                self.create(file, chunked=chunked, reference=reference, shape=shape,
                            properties=properties, user_fill=not default_fill)
        plan = make_plan(source, ['/data'])
        return source, output, plan

    def native_setter(self, name, typ, value):
        library = ctypes.CDLL(self.h5.h5p.__file__)
        fn = getattr(library, name)
        fn.argtypes = [ctypes.c_int64, typ]
        fn.restype = ctypes.c_int
        def mutate(dcpl):
            with self.h5._objects.phil:
                self.assertEqual(fn(dcpl.id, value), 0)
        return mutate

    def test_populated_output_dcpl_mutations_are_detected_despite_identical_descriptors(self):
        from h5carry.verify import verify_export
        h5 = self.h5
        mutations = {
            'fill-time': lambda p: p.set_fill_time(h5.h5d.FILL_TIME_ALLOC),
            'alloc-time': lambda p: p.set_alloc_time(h5.h5d.ALLOC_TIME_EARLY),
            'attr-order': lambda p: p.set_attr_creation_order(h5.h5p.CRT_ORDER_TRACKED | h5.h5p.CRT_ORDER_INDEXED),
            'attr-phase': lambda p: p.set_attr_phase_change(2, 1),
            'track-times': lambda p: p.set_obj_track_times(False),
        }
        self.f.close()
        for tag, mutation in mutations.items():
            with self.subTest(property=tag):
                source, output, plan = self.pair(tag, mutation)
                with self.h5.File(source, 'r') as src, self.h5.File(output, 'r') as dst:
                    self.assertEqual(self.p.describe_object(src['data'], self.limits),
                                     self.p.describe_object(dst['data'], self.limits))
                    self.assertFalse(src['data'].id.get_create_plist().equal(dst['data'].id.get_create_plist()))
                checked = verify_export(source, output, plan, self.limits)
                self.assertEqual(checked['status'], 'mismatch', checked)
                self.assertFalse(checked['coverage']['source_equality'])
                self.assertTrue(any('creation' in d['message'] and d.get('path') == '/data'
                                    for d in checked['diagnostics']), checked)

    def test_dcpl_comparison_includes_reference_empty_and_null_datasets(self):
        from h5carry.verify import verify_export
        self.f.close()
        for i, (reference, shape) in enumerate([(True, (8,)), (False, (0,)), (False, None), (True, None)]):
            with self.subTest(reference=reference, shape=shape):
                source, output, plan = self.pair(f'nonfixed-{i}', lambda p: p.set_attr_phase_change(2, 1),
                                               chunked=shape is not None, reference=reference, shape=shape)
                checked = verify_export(source, output, plan, self.limits)
                self.assertEqual(checked['status'], 'mismatch', checked)
                self.assertFalse(checked['coverage']['source_equality'])

    def test_default_versus_user_zero_fill_is_detected_despite_identical_descriptors(self):
        from h5carry.verify import verify_export
        self.f.close()
        source, output, plan = self.pair('fill-defined',
            lambda p: p.set_fill_value(self.np.array(0, dtype='<i4')), default_fill=True)
        with self.h5.File(source, 'r') as src, self.h5.File(output, 'r') as dst:
            a, b = src['data'].id.get_create_plist(), dst['data'].id.get_create_plist()
            self.assertNotEqual(a.fill_value_defined(), b.fill_value_defined())
            self.assertFalse(a.equal(b))
            self.assertEqual(self.p.describe_object(src['data'], self.limits),
                             self.p.describe_object(dst['data'], self.limits))
        checked = verify_export(source, output, plan, self.limits)
        self.assertEqual(checked['status'], 'mismatch', checked)
        self.assertFalse(checked['coverage']['source_equality'])

    def test_changed_fill_value_is_detected(self):
        from h5carry.verify import verify_export
        self.f.close()
        source, output, plan = self.pair('fill-value',
            lambda p: p.set_fill_value(self.np.array(-8, dtype='<i4')))
        checked = verify_export(source, output, plan, self.limits)
        self.assertEqual(checked['status'], 'mismatch', checked)
        self.assertFalse(checked['coverage']['source_equality'])

    def test_no_attributes_hint_is_live_only_not_a_reopened_mutation(self):
        from h5carry.verify import verify_export
        library = ctypes.CDLL(self.h5.h5p.__file__)
        getter = library.H5Pget_dset_no_attrs_hint
        getter.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_bool)]
        getter.restype = ctypes.c_int
        def hint(dcpl):
            value = ctypes.c_bool()
            with self.h5._objects.phil:
                self.assertEqual(getter(dcpl.id, ctypes.byref(value)), 0)
            return value.value
        mutate = self.native_setter('H5Pset_dset_no_attrs_hint', ctypes.c_bool, True)
        source = self.f.filename
        original = self.create(self.f, chunked=True)
        output = str(Path(self.tmp.name) / 'hint-output.h5')
        with self.h5.File(output, 'w', libver='latest') as dst:
            changed = self.create(dst, chunked=True, properties=mutate)
            self.assertFalse(hint(original.id.get_create_plist()))
            self.assertTrue(hint(changed.id.get_create_plist()))
            self.assertFalse(original.id.get_create_plist().equal(changed.id.get_create_plist()))
        self.f.close()
        with self.h5.File(source, 'r') as src, self.h5.File(output, 'r') as dst:
            self.assertFalse(hint(src['data'].id.get_create_plist()))
            self.assertFalse(hint(dst['data'].id.get_create_plist()))
            self.assertTrue(src['data'].id.get_create_plist().equal(dst['data'].id.get_create_plist()))
        plan = self.manual_plan(source)
        checked = verify_export(source, output, plan, self.limits)
        self.assertEqual(checked['status'], 'verified', checked)

    def test_low_level_datatype_check_does_not_depend_on_shared_descriptor_spelling(self):
        from h5carry.plan import make_plan
        from h5carry.verify import verify_export
        self.f.close()
        source = str(Path(self.tmp.name) / 'type-source.h5')
        output = str(Path(self.tmp.name) / 'type-output.h5')
        for path, dtype in ((source, '<i4'), (output, '>i4')):
            with self.h5.File(path, 'w') as file:
                file.create_dataset('data', data=self.np.zeros((8,), dtype=dtype), fillvalue=0)
        plan = make_plan(source, ['/data'])
        native_descriptor = self.p.dtype_descriptor
        def same_spelling(low_type, dtype, attribute=False):
            descriptor = native_descriptor(low_type, dtype, attribute=attribute)
            return dict(descriptor, str='<i4')
        # Exercise the promised direct TypeID comparison even if the shared
        # primitive's JSON spelling accidentally conceals this endian change.
        with patch('h5carry.verify.dtype_descriptor', same_spelling):
            checked = verify_export(source, output, plan, self.limits)
        self.assertEqual(checked['status'], 'mismatch', checked)
        self.assertFalse(checked['coverage']['source_equality'])
        self.assertTrue(any('low-level dataset datatype' in d['message']
                            for d in checked['diagnostics']), checked)

    def test_populated_output_never_fill_time_is_refused_before_read(self):
        from h5carry.verify import verify_export
        self.f.close()
        source, output, plan = self.pair('output-never', lambda p: p.set_fill_time(self.h5.h5d.FILL_TIME_NEVER))
        instrument, calls = self.forbidden_reads()
        with instrument:
            self.assert_never_report(verify_export(source, output, plan, self.limits))
        self.assertEqual(calls, [])

    def test_native_chunk_option_difference_is_detected_even_when_dcpl_equal(self):
        from h5carry.native_properties import chunk_options
        from h5carry.verify import verify_export
        self.f.close()
        source, output, plan = self.pair('chunk-options', self.native_setter('H5Pset_chunk_opts', ctypes.c_uint, 2))
        with self.h5.File(source, 'r') as src, self.h5.File(output, 'r') as dst:
            self.assertTrue(src['data'].id.get_create_plist().equal(dst['data'].id.get_create_plist()))
            self.assertEqual(chunk_options(src['data'].id.get_create_plist()), 0)
            self.assertEqual(chunk_options(dst['data'].id.get_create_plist()), 2)
            self.assertEqual(self.p.describe_object(src['data'], self.limits), self.p.describe_object(dst['data'], self.limits))
        checked = verify_export(source, output, plan, self.limits)
        self.assertEqual(checked['status'], 'mismatch', checked)
        self.assertFalse(checked['coverage']['source_equality'])
        self.assertTrue(any('chunk' in d['message'] and d.get('path') == '/data' for d in checked['diagnostics']), checked)

    def test_matching_nondefault_chunk_options_are_copied_and_verified(self):
        from h5carry.model import fingerprint
        from h5carry.native_properties import chunk_options
        from h5carry.scan import make_plan_native
        from h5carry.write import write_staging
        from h5carry.verify import verify_export
        source = self.f.filename
        self.create(self.f, chunked=True, properties=self.native_setter('H5Pset_chunk_opts', ctypes.c_uint, 2))
        self.f.close()
        output = str(Path(self.tmp.name) / 'matching-options.h5')
        graph = make_plan_native(source, ['/data'], self.limits)
        write_staging(source, graph, output, self.limits)
        with self.h5.File(output, 'r') as dst:
            self.assertEqual(chunk_options(dst['data'].id.get_create_plist()), 2)
        checked = verify_export(source, output, {'source': fingerprint(source), 'selections': ['/data'], 'graph': graph}, self.limits)
        self.assertEqual(checked['status'], 'verified', checked)

    def test_unavailable_or_unqualified_chunk_option_reader_cannot_verify(self):
        from h5carry.verify import verify_export
        self.f.close()
        source, output, plan = self.pair('chunk-options-unavailable', lambda p: None)
        failed_library = Mock()
        failed_library.H5Pget_chunk_opts.return_value = -1
        failures = {
            'missing-library': patch('ctypes.CDLL', side_effect=OSError('unavailable')),
            'missing-symbol': patch('ctypes.CDLL', return_value=object()),
            'failed-query': patch('ctypes.CDLL', return_value=failed_library),
            'unknown-hdf5': patch.object(self.h5.version, 'hdf5_version', 'unqualified'),
            'unknown-h5py': patch.object(self.h5, '__version__', 'unqualified'),
        }
        for failure, guard in failures.items():
            with self.subTest(failure=failure):
                with guard:
                    checked = verify_export(source, output, plan, self.limits)
                self.assertEqual(checked['status'], 'incomplete', checked)
                self.assertFalse(checked['coverage']['source_equality'])
                self.assertEqual(checked['diagnostics'][0]['code'], 'UNSUPPORTED')
                self.assertEqual(checked['diagnostics'][0]['path'], '/data')


if __name__ == '__main__':
    unittest.main()
