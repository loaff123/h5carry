"""Original rectangular-snapshot fixture definitions and portable reproducer.

These 14 case identities, source coordinates, formula values and outcomes were
frozen in the 2026-10-06 design probe. The typed adapter changes only syntax;
redundant attachment assertions through /A_alias are written once for the shared
/A identity. This is original fixture code, never a copying implementation.
Run native fixture creation with tests.native_runner, not by direct import.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def box(path, start=(1, 1), stop=(4, 3)):
    return {'path': path, 'selection': {'kind': 'box', 'start': list(start), 'stop': list(stop)}}


def whole(path):
    return {'path': path, 'selection': {'kind': 'whole'}}


def request(objects, mappings=()):
    return {'format': 'h5carry-selection', 'version': 1, 'objects': sorted(objects, key=lambda item: item['path']),
            'scale_mappings': sorted(mappings, key=lambda item: (item['consumer'], item['axis'], item['scale'])), 'storage_policy': 'fixed-snapshot-v1'}


def maps(consumers=('/A',), scales=('/time', '/channel')):
    return [{'consumer': consumer, 'axis': axis, 'scale': scale, 'mapping': 'index'}
            for consumer in consumers for axis, scale in enumerate(scales)]


def original_cases():
    qmaps = maps(('/Q',), ('/time', '/time'))
    multiple = [{'consumer': '/A', 'axis': 0, 'scale': scale, 'mapping': 'index'} for scale in ('/time', '/alternate')] + maps()[1:]
    entries = [
        ('plain_rectangle', [box('/A')], [], None),
        ('shared_coherent', [box('/A'), box('/B')], maps(('/A', '/B')), None),
        ('shared_conflict', [box('/A'), box('/B', (2, 1), (5, 3))], maps(('/A', '/B')), 'conflict'),
        ('repeated_scale_coherent', [box('/Q', (1, 1), (4, 4))], qmaps, None),
        ('repeated_scale_conflict', [box('/Q', (1, 2), (4, 5))], qmaps, 'conflict'),
        ('multiple_scales', [box('/A')], multiple, None),
        ('aliases_coherent', [box('/A'), box('/A_alias')], maps(), None),
        ('aliases_conflict', [box('/A'), box('/A_alias', (2, 1), (5, 3))], maps(), 'conflict'),
        ('reference_outbound', [box('/A'), whole('/refs'), whole('/gain_alias')], maps(), None),
        ('reference_inbound', [box('/A'), whole('/refs')], maps(), 'conflict'),
        ('empty_selection', [box('/A', (2, 1), (2, 3))], maps(), None),
        ('nonpositional_rank', [box('/A')], maps(scales=('/weird', '/channel')), 'rank-one'),
        ('nonpositional_length', [box('/A')], maps(scales=('/weird', '/channel')), 'rank-one'),
        ('compound', [box('/records', (0,), (1,))], [], 'compound'),
    ]
    result = []
    for name, objects, mappings, refusal in entries:
        old_mappings = mappings
        if name.startswith('aliases_'):
            old_mappings = maps(('/A', '/A_alias'))
        old = {'selections': [{'path': entry['path'], 'start': entry['selection']['start'],
                               'stop': entry['selection']['stop']}
                              for entry in objects if entry['selection']['kind'] == 'box'],
               'whole': [entry['path'] for entry in objects if entry['selection']['kind'] == 'whole'],
               'scale_mappings': old_mappings}
        result.append({'name': name, 'request': request(objects, mappings),
                       'original_recipe_request': old, 'expected_refusal': refusal,
                       'adapter_note': 'Deduplicate identical scale assertions through the known /A_alias identity'
                           if name.startswith('aliases_') else 'Syntax-only conversion to strict typed request'})
    return result



def contract_variant_cases():
    """Two separately named controls; never rewrite the frozen prototype case."""
    original = next(case for case in original_cases() if case['name'] == 'reference_outbound')
    strengthened = json.loads(json.dumps(original['request']))
    strengthened['objects'].append(whole('/gain'))
    strengthened['objects'].sort(key=lambda item: item['path'])
    return [
        {'name': 'reference_outbound_retained_alias', 'source_case': 'reference_outbound',
         'checker_case': 'reference_outbound_retained_alias', 'request': original['request'],
         'expected_refusal': None, 'original_recipe_request': original['original_recipe_request'],
         'adapter_note': 'Same request; separately corrected expectation follows established retained-path closure: /gain_alias is sufficient for whole gain identity, refs remap to it'},
        {'name': 'reference_outbound_explicit_both_aliases', 'source_case': 'reference_outbound',
         'checker_case': 'reference_outbound', 'request': strengthened, 'expected_refusal': None,
         'original_recipe_request': original['original_recipe_request'],
         'adapter_note': 'Strengthened request explicitly adds whole /gain; original prototype checker remains unchanged and checks both retained aliases'},
    ]


def resource_cases():
    one = (1,) * 30
    entries = [
        ('offset_crossing_chunks', [box('/A', (1, 3), (6, 17))], None, ['--chunk-bytes', '32']),
        ('wide_rows', [box('/A', (1, 123), (3, 399999))], None, ['--chunk-bytes', '8192']),
        ('rank32', [box('/A', (0,) * 30 + (1, 1), one + (3, 4))], None, []),
        ('empty_contiguous', [box('/A', (2, 1), (2, 3))], None, []),
        ('empty_rank32', [box('/A', (1,) + (0,) * 31, (1,) * 32)], None, []),
        ('large_sparse_tiny_crop', [box('/A', (2**35 - 1, 1), (2**35, 3))], None, []),
        ('maximum_payload', [box('/A', (1, 0), (65537, 1024))], None, []),
        ('shared_128_consumers', [box('/C%03d' % index, (3, 2), (11, 6)) for index in range(128)], None, []),
        ('compressed_overcap', [box('/A', (1, 0), (8194, 8192))], 'RESOURCE', []),
        ('uncompressed_overcap', [box('/A', (1, 0), (8194, 8192))], 'RESOURCE', []),
    ]
    result = []
    for name, objects, refusal, limits in entries:
        mappings = maps(tuple(entry['path'] for entry in objects)) if name == 'shared_128_consumers' else []
        result.append({'name': name, 'request': request(objects, mappings), 'expected_refusal': refusal,
                       'limits': limits})
    return result


def require_native_limits():
    """Check controls before NumPy/h5py imports and any native file operation."""
    import resource
    import sys
    if sys.platform != 'linux' or os.environ.get('H5CARRY_TEST_NATIVE_CHILD') != '1':
        raise RuntimeError('Run fixture/check modules through tests.native_runner')
    for kind, maximum in ((resource.RLIMIT_AS, 2 * 1024**3), (resource.RLIMIT_CPU, 120),
                          (resource.RLIMIT_FSIZE, 1024**3)):
        soft, _ = resource.getrlimit(kind)
        if soft == resource.RLIM_INFINITY or not 0 < soft <= maximum:
            raise RuntimeError('Required native process limits missing')
    if os.environ.get('HDF5_PLUGIN_PRELOAD') != '::':
        raise RuntimeError('Dynamic HDF5 plugins are not disabled')


def create_original(path, case):
    require_native_limits()
    import h5py
    import numpy as np
    with h5py.File(path, 'w') as f:
        f.attrs['experiment'] = 'original benign slice probe'
        time = f.create_dataset('time', data=np.array([0, 10, 20, 30, 40, 50], dtype='>i4'))
        time.make_scale('axis')
        channel = f.create_dataset('channel', data=np.array([400, 500, 600, 700], dtype='>i4'))
        channel.make_scale('axis')
        gain = f.create_dataset('gain', data=np.array([2., 3., 5., 7.]))
        gain.attrs['unit'] = 'dimensionless'
        for name, offset in [('A', 0), ('B', 1000), ('C', 9000)]:
            d = f.create_dataset(name, data=np.array([[offset + 100*i + j for j in range(4)] for i in range(6)],
                dtype='>i4'), chunks=(4, 4), compression='gzip', compression_opts=4, shuffle=True,
                fletcher32=True, fillvalue=-1)
            d.attrs['unit'] = 'counts'; d.attrs['marker'] = 'excluded' if name == 'C' else name
            if case != 'plain_rectangle':
                d.dims[0].attach_scale(time); d.dims[1].attach_scale(channel)
                d.dims[0].label = 'time'; d.dims[1].label = 'channel'
        f['A_alias'] = f['A']
        if case.startswith('repeated_scale'):
            q = f.create_dataset('Q', data=np.array([[100*i + j for j in range(6)] for i in range(6)], dtype='>i4'))
            q.dims[0].attach_scale(time); q.dims[1].attach_scale(time)
        if case == 'multiple_scales':
            alt = f.create_dataset('alternate', data=np.array([100, 110, 120, 130, 140, 150], dtype='>i4'))
            alt.make_scale('axis'); f['A'].dims[0].attach_scale(alt)
        if case == 'reference_outbound':
            f['gain_alias'] = gain; f['A'].attrs['calibration_ref'] = gain.ref
            refs = f.create_dataset('refs', (2,), dtype=h5py.ref_dtype)
            refs[0] = gain.ref; refs[1] = h5py.Reference()
        if case == 'reference_inbound':
            refs = f.create_dataset('refs', (1,), dtype=h5py.ref_dtype); refs[0] = f['A'].ref
        if case.startswith('nonpositional'):
            f['A'].dims[0].detach_scale(time)
            shape = (2, 3) if case.endswith('rank') else (5,)
            weird = f.create_dataset('weird', data=np.arange(np.prod(shape), dtype='>i4').reshape(shape))
            weird.make_scale('not positional'); f['A'].dims[0].attach_scale(weird)
        if case == 'compound':
            f.create_dataset('records', data=np.array([(1, 2.), (3, 4.)], dtype=[('i', '<i4'), ('x', '<f8')]))


def create_resource(path, case):
    require_native_limits()
    import h5py
    import numpy as np
    with h5py.File(path, 'w') as f:
        if case == 'offset_crossing_chunks':
            f.create_dataset('A', data=np.array([[100*i+j for j in range(19)] for i in range(7)], dtype='>i4'),
                             chunks=(2, 4), compression='gzip', compression_opts=4)
        elif case == 'wide_rows':
            d = f.create_dataset('A', (3, 400001), dtype='>i8', chunks=(1, 256))
            for row in range(3):
                for col in range(0, 400001, 4096):
                    end = min(col+4096, 400001)
                    d[row, col:end] = 1000000*row + np.arange(col, end, dtype='>i8')
        elif case == 'rank32':
            f.create_dataset('A', data=np.arange(12, dtype='>i4').reshape((1,)*30 + (3, 4)))
        elif case == 'empty_contiguous':
            f.create_dataset('A', data=np.arange(20, dtype='>i4').reshape(4, 5))
        elif case == 'empty_rank32':
            f.create_dataset('A', data=np.arange(3, dtype='>i4').reshape((3,) + (1,)*31))
        elif case == 'large_sparse_tiny_crop':
            f.create_dataset('A', (2**35, 4), dtype='u1', chunks=(1, 4), fillvalue=7)
        elif case == 'maximum_payload':
            f.create_dataset('A', (65537, 1024), dtype='u4', chunks=(32, 256),
                             compression='gzip', compression_opts=4, fillvalue=7)
        elif case == 'shared_128_consumers':
            time = f.create_dataset('time', data=np.array([10*i for i in range(16)], dtype='>i4')); time.make_scale('axis')
            channel = f.create_dataset('channel', data=np.array([100*i for i in range(8)], dtype='>i4')); channel.make_scale('axis')
            for index in range(128):
                d = f.create_dataset('C%03d' % index,
                    data=np.array([[10000*index+100*i+j for j in range(8)] for i in range(16)], dtype='>i4'))
                d.dims[0].attach_scale(time); d.dims[1].attach_scale(channel)
        elif case in ('compressed_overcap', 'uncompressed_overcap'):
            options = {'compression': 'gzip', 'compression_opts': 4} if case == 'compressed_overcap' else {}
            f.create_dataset('A', (8194, 8192), dtype='u4', chunks=(32, 256), fillvalue=7, **options)
        else:
            raise ValueError('Unknown resource case: ' + case)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--resources', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to replace fixture')
    known = resource_cases() if args.resources else original_cases()
    if args.case not in {item['name'] for item in known}:
        raise ValueError('Unknown fixture case')
    (create_resource if args.resources else create_original)(args.output, args.case)
    print(json.dumps({'status': 'generated', 'case': args.case, 'source_bytes': args.output.stat().st_size}))


if __name__ == '__main__':
    main()
