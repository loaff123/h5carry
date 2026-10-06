"""Output-only snapshot consumer with independent original-coordinate expectations.

Imports no product planner, scanner, writer, verifier, selection resolver or
transfer mapper. No plan or source is read. Run this trusted module under
``python -m tests.native_runner qualification.check_snapshot`` so limits and
plugin controls precede native imports. The source-path assertion tests path
independence; it is not filesystem isolation and a held source may still exist.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import sys


def require_native_limits():
    import resource
    if sys.platform != 'linux' or os.environ.get('H5CARRY_TEST_NATIVE_CHILD') != '1':
        raise RuntimeError('Use the bounded tests.native_runner launcher')
    for kind, maximum in ((resource.RLIMIT_AS, 2*1024**3), (resource.RLIMIT_CPU, 120),
                          (resource.RLIMIT_FSIZE, 1024**3)):
        soft, _ = resource.getrlimit(kind)
        if soft == resource.RLIM_INFINITY or not 0 < soft <= maximum:
            raise RuntimeError('Required native resource limits absent')
    if os.environ.get('HDF5_PLUGIN_PRELOAD') != '::':
        raise RuntimeError('Dynamic HDF5 plugins must be disabled')
    forbidden = ('h5carry.plan', 'h5carry.plan_v2', 'h5carry.scan', 'h5carry.scan_v2',
                 'h5carry.write', 'h5carry.write_v2', 'h5carry.verify', 'h5carry.verify_v2',
                 'h5carry.selection', 'h5carry.slicing')
    if any(name in sys.modules for name in forbidden):
        raise RuntimeError('External observer contaminated by product implementation imports')


class Observer:
    def __init__(self):
        self.checks = 0
        self.payload_bytes = 0
        self.maximum_read_bytes = 0

    def need(self, condition, message):
        self.checks += 1
        if not condition:
            raise AssertionError(message)

    def payload(self, dataset, selection, expected):
        import numpy as np
        actual = np.asarray(dataset[selection])
        self.payload_bytes += actual.nbytes
        self.maximum_read_bytes = max(self.maximum_read_bytes, actual.nbytes)
        self.need(actual.shape == expected.shape, 'payload shape differs at ' + dataset.name)
        self.need(actual.dtype == expected.dtype, 'payload dtype differs at ' + dataset.name)
        self.need(actual.tobytes(order='C') == expected.tobytes(order='C'), 'payload bytes differ at ' + dataset.name)

    def type(self, dataset, expected):
        self.need(dataset.id.get_type().equal(expected), 'exact low-level dtype differs at ' + dataset.name)

    def strings(self, obj, expected):
        import h5py
        for key, value in expected.items():
            self.need(obj.attrs[key] == value, 'ordinary attribute value differs at ' + obj.name + '@' + key)
            aid = obj.attrs.get_id(key)
            info = h5py.check_string_dtype(aid.dtype)
            self.need(aid.shape == () and info is not None and info.encoding == 'utf-8' and info.length is None,
                      'ordinary string attribute encoding/shape differs at ' + obj.name + '@' + key)

    def primitive(self, dataset, shape, dtype, chunks=None, filters=(), fill=None):
        import h5py
        self.need(dataset.shape == shape and dataset.maxshape == shape,
                  'fixed snapshot shape/maximum differs at ' + dataset.name)
        self.type(dataset, dtype)
        self.need(dataset.chunks == chunks, 'chunk transformation differs at ' + dataset.name)
        dcpl = dataset.id.get_create_plist()
        actual_filters = tuple((dcpl.get_filter(index)[0], dcpl.get_filter(index)[1], dcpl.get_filter(index)[2])
                               for index in range(dcpl.get_nfilters()))
        self.need(actual_filters == tuple(filters), 'exact filter pipeline/flags/parameters differ at ' + dataset.name)
        expected_layout = h5py.h5d.CHUNKED if chunks is not None else h5py.h5d.CONTIGUOUS
        self.need(dcpl.get_layout() == expected_layout, 'dataset layout differs at ' + dataset.name)
        self.need(not dataset.is_virtual and dataset.external is None, 'nonlocal dataset storage at ' + dataset.name)
        if fill is not None:
            self.need(dataset.fillvalue == fill, 'fill value differs at ' + dataset.name)

    def namespace(self, file, expected):
        import h5py
        self.need(set(file) == set(expected), 'exact retained namespace differs')
        for name in file:
            self.need(isinstance(file.get(name, getlink=True), h5py.HardLink), 'expected hard link at /' + name)
            self.need(isinstance(file[name], h5py.Dataset), 'expected dataset at /' + name)
        self.need(all(file[name].attrs.get('marker') != 'excluded' for name in file), 'excluded marker leaked')

    def scales(self, file, consumers, attachments, labels, values):
        import h5py
        import numpy as np
        expected = Counter()
        for name in consumers:
            d = file[name]
            self.need([axis.label for axis in d.dims] == labels[name], 'dimension labels differ at /' + name)
            for axis, names in attachments[name].items():
                ids = Counter(int(h5py.h5o.get_info(file[target].id).addr) for target in names)
                high = Counter(int(h5py.h5o.get_info(s.id).addr) for s in d.dims[axis].values())
                raw = Counter(int(h5py.h5o.get_info(file[r].id).addr) for r in d.attrs['DIMENSION_LIST'][axis])
                self.need(high == raw == ids, 'raw/high-level forward scale relations differ at /' + name)
                for target in names:
                    expected[(target, int(h5py.h5o.get_info(d.id).addr), axis)] += 1
        actual = Counter()
        for name, expected_values in values.items():
            d = file[name]
            self.need(d.is_scale and d.attrs['CLASS'] == b'DIMENSION_SCALE' and d.attrs['NAME'] == b'axis',
                      'scale convention/name differs at /' + name)
            self.primitive(d, (len(expected_values),), h5py.h5t.STD_I32BE)
            if expected_values:
                self.payload(d, slice(None), np.array(expected_values, dtype='>i4'))
            else:
                self.need(d.size == 0, 'empty scale not empty')
            self.need(set(d.attrs) == {'CLASS', 'NAME', 'REFERENCE_LIST'}, 'scale attribute namespace differs at /' + name)
            for row in d.attrs['REFERENCE_LIST']:
                actual[(name, int(h5py.h5o.get_info(file[row['dataset']].id).addr), int(row['dimension']))] += 1
        self.need(actual == expected, 'exact retained reverse scale consumer/axis pairs differ')


def check_original(path, case):
    require_native_limits()
    import h5py
    import numpy as np
    obs = Observer()
    with h5py.File(path, 'r', rdcc_nbytes=0) as f:
        square = case == 'repeated_scale_coherent'
        empty = case == 'empty_selection'
        plain = case == 'plain_rectangle'
        outbound = case in ('reference_outbound', 'reference_outbound_retained_alias')
        gain_name = 'gain_alias' if case == 'reference_outbound_retained_alias' else 'gain'
        consumers = ['Q'] if square else ['A', 'B'] if case == 'shared_coherent' else ['A']
        names = set(consumers)
        if not plain:
            names.update(['time'] if square else ['time', 'channel'])
        if case == 'aliases_coherent':
            names.add('A_alias')
        if case == 'multiple_scales':
            names.add('alternate')
        if outbound:
            names.update(['gain_alias', 'refs'] if gain_name == 'gain_alias' else ['gain', 'gain_alias', 'refs'])
        obs.namespace(f, names)
        obs.need(set(f.attrs) == {'experiment'}, 'root attribute namespace differs')
        obs.strings(f, {'experiment': 'original benign slice probe'})
        literals = {'A': [[101, 102], [201, 202], [301, 302]],
                    'B': [[1101, 1102], [1201, 1202], [1301, 1302]],
                    'Q': [[101, 102, 103], [201, 202, 203], [301, 302, 303]]}
        filters = ((2, 1, (4,)), (1, 1, (4,)), (3, 0, ()))
        for name in consumers:
            d = f[name]
            shape = (0, 2) if empty else (3, 3) if square else (3, 2)
            obs.primitive(d, shape, h5py.h5t.STD_I32BE,
                          None if square else (1, 2) if empty else (3, 2),
                          () if square else filters, None if square else -1)
            if not empty:
                obs.payload(d, (), np.array(literals[name], dtype='>i4'))
            else:
                obs.need(d.size == 0, 'empty consumer not empty')
            normal = set() if square else {'unit', 'marker'}
            if outbound:
                normal.add('calibration_ref')
            reserved = set() if plain else {'DIMENSION_LIST'} | (set() if square else {'DIMENSION_LABELS'})
            obs.need(set(d.attrs) == normal | reserved, 'consumer attribute namespace differs at /' + name)
            if not square:
                obs.strings(d, {'unit': 'counts', 'marker': name})
        if not plain:
            attachments = {name: {0: ['time'] + (['alternate'] if case == 'multiple_scales' else []),
                                  1: ['time'] if square else ['channel']} for name in consumers}
            labels = {name: ['', ''] if square else ['time', 'channel'] for name in consumers}
            values = {'time': [] if empty else [10, 20, 30]}
            if not square:
                values['channel'] = [500, 600]
            if case == 'multiple_scales':
                values['alternate'] = [110, 120, 130]
            obs.scales(f, consumers, attachments, labels, values)
        else:
            obs.need([len(dimension) for dimension in f['A'].dims] == [0, 0], 'unexpected plain scales')
        distinct = consumers + ([] if plain else ['time'] if square else ['time', 'channel'])
        if case == 'multiple_scales':
            distinct.append('alternate')
        if outbound:
            distinct.extend([gain_name, 'refs'])
        ids = [int(h5py.h5o.get_info(f[name].id).addr) for name in distinct]
        obs.need(len(ids) == len(set(ids)), 'distinct required dataset identities conflated')
        if case == 'aliases_coherent':
            obs.need(f['A'].id == f['A_alias'].id, 'retained alias identity split')
        if outbound:
            gain = f[gain_name]
            obs.primitive(gain, (4,), h5py.h5t.IEEE_F64LE)
            obs.payload(gain, (), np.array([2., 3., 5., 7.], dtype='<f8'))
            obs.need(set(gain.attrs) == {'unit'}, 'gain attribute namespace differs')
            obs.strings(gain, {'unit': 'dimensionless'})
            if gain_name == 'gain':
                obs.need(gain.id == f['gain_alias'].id, 'whole reference target alias split')
            aid = f['A'].attrs.get_id('calibration_ref')
            obs.need(aid.shape == () and aid.get_type().equal(h5py.h5t.STD_REF_OBJ), 'ordinary ref attr type/shape changed')
            obs.need(f[f['A'].attrs['calibration_ref']].id == gain.id, 'normal attribute reference target differs')
            refs = f['refs']
            obs.need(refs.shape == refs.maxshape == (2,) and refs.chunks is None and not refs.attrs,
                     'whole ref dataset descriptor differs')
            obs.type(refs, h5py.h5t.STD_REF_OBJ)
            actual = refs[()]
            obs.need(bool(actual[0]) and not actual[1], 'nonnull/null reference pattern differs')
            obs.need(f[actual[0]].id == gain.id, 'reference dataset target differs')
    return obs


def check_resource(path, case):
    require_native_limits()
    import h5py
    import numpy as np
    obs = Observer()
    with h5py.File(path, 'r', rdcc_nbytes=0) as f:
        many = case == 'shared_128_consumers'
        names = {'C%03d' % index for index in range(128)} | {'time', 'channel'} if many else {'A'}
        obs.namespace(f, names)
        obs.need(not f.attrs, 'unexpected resource root attributes')
        if many:
            consumers = ['C%03d' % index for index in range(128)]
            for index, name in enumerate(consumers):
                d = f[name]
                obs.primitive(d, (8, 4), h5py.h5t.STD_I32BE)
                expected = np.array([[10000*index+100*i+j for j in range(2, 6)] for i in range(3, 11)], dtype='>i4')
                obs.payload(d, (), expected)
                obs.need(set(d.attrs) == {'DIMENSION_LIST'}, 'unexpected shared consumer attribute')
            obs.scales(f, consumers, {name: {0: ['time'], 1: ['channel']} for name in consumers},
                       {name: ['', ''] for name in consumers},
                       {'time': [30, 40, 50, 60, 70, 80, 90, 100], 'channel': [200, 300, 400, 500]})
            return obs
        d = f['A']
        obs.need(not d.attrs, 'unexpected resource dataset attribute')
        if case == 'offset_crossing_chunks':
            obs.primitive(d, (5, 14), h5py.h5t.STD_I32BE, (2, 4), ((1, 1, (4,)),))
            obs.payload(d, (), np.array([[100*i+j for j in range(3, 17)] for i in range(1, 6)], dtype='>i4'))
        elif case == 'wide_rows':
            obs.primitive(d, (2, 399876), h5py.h5t.STD_I64BE, (1, 256))
            for row in range(2):
                for col in range(0, 399876, 1024):
                    end = min(col+1024, 399876)
                    expected = (1000000*(row+1) + np.arange(123+col, 123+end)).astype('>i8')
                    obs.payload(d, (row, slice(col, end)), expected)
        elif case == 'rank32':
            shape = (1,)*30 + (2, 3)
            obs.primitive(d, shape, h5py.h5t.STD_I32BE)
            obs.payload(d, (), np.array([5, 6, 7, 9, 10, 11], dtype='>i4').reshape(shape))
        elif case in ('empty_contiguous', 'empty_rank32'):
            shape = (0, 2) if case == 'empty_contiguous' else (0,) + (1,)*31
            obs.primitive(d, shape, h5py.h5t.STD_I32BE)
            obs.need(d.size == 0, 'empty resource not empty')
        elif case == 'large_sparse_tiny_crop':
            obs.primitive(d, (1, 2), h5py.h5t.STD_U8LE, (1, 2), fill=7)
            obs.payload(d, (), np.array([[7, 7]], dtype='u1'))
        elif case == 'maximum_payload':
            obs.primitive(d, (65536, 1024), h5py.h5t.STD_U32LE, (32, 256), ((1, 1, (4,)),), fill=7)
            expected = np.full((32, 1024), 7, dtype='u4')
            for row in range(0, 65536, 32):
                obs.payload(d, (slice(row, row+32), slice(None)), expected)
            obs.need(obs.payload_bytes == 256*1024**2, 'maximum selected-payload count differs')
        else:
            raise ValueError('Unknown successful resource case')
    return obs


def mutate(path, kind):
    require_native_limits()
    import h5py
    with h5py.File(path, 'r+') as f:
        if kind == 'value':
            f['A'][0, 0] = 999
        elif kind == 'alias':
            del f['A_alias']; f.create_dataset('A_alias', data=f['A'][()])
        elif kind == 'ref':
            f['refs'][0] = f['A'].ref
        elif kind == 'scale':
            f['B'].dims[0].detach_scale(f['time'])
        else:
            raise ValueError('Unknown mutation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--resources', action='store_true')
    parser.add_argument('--source-must-be-absent', type=Path)
    parser.add_argument('--mutate', choices=('value', 'alias', 'ref', 'scale'))
    args = parser.parse_args()
    if args.source_must_be_absent is not None and args.source_must_be_absent.exists():
        raise AssertionError('Original operation source pathname is still available')
    if args.mutate:
        mutate(args.input, args.mutate)
        print(json.dumps({'status': 'mutated', 'kind': args.mutate}))
        return
    obs = (check_resource if args.resources else check_original)(args.input, args.case)
    import h5py
    import numpy as np
    print(json.dumps({'status': 'verified', 'case': args.case, 'checks': obs.checks,
        'payload_bytes_observed': obs.payload_bytes, 'maximum_logical_observer_read': obs.maximum_read_bytes,
        'fresh_process_pid': os.getpid(), 'source_path_unavailable': args.source_must_be_absent is not None,
        'output_only_independent_observer': True,
        'runtime': {'python': platform.python_version(), 'h5py': h5py.__version__,
                    'hdf5': h5py.version.hdf5_version, 'numpy': np.__version__}}, sort_keys=True))


if __name__ == '__main__':
    main()
