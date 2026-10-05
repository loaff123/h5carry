"""Manual reopened-output assertions for the four original frozen fixture families.

This observer imports no H5Carry planner, scanner, writer, verifier, or plan file.
Expected paths, values, relations and exclusions are specified independently.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples._bounded import launch, require_native_limits


def check_selected(path, family):
    require_native_limits()
    import h5py
    import numpy as np
    from examples.analyze_selected import marker_scan

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    with h5py.File(path, 'r') as archive:
        selected_runs = ('selected', 'other') if family == 2 else ('selected',)
        groups = {'/', '/axes', '/calibration', '/runs'} | {'/runs/' + name for name in selected_runs}
        arrays = {'/axes/time': np.asarray([0., .1, .2, .3]),
                  '/axes/channel': np.asarray([1, 2, 3], dtype='i2'),
                  '/calibration/gain': np.asarray([1.5, 2.5, 3.5])}
        for name in selected_runs:
            arrays['/runs/' + name + '/intensity'] = np.arange(12, dtype='i4').reshape(4, 3) + (100 if name == 'other' else 0)
        if family == 4:
            arrays.update({'/axes/alternate': np.arange(4, dtype='u2'),
                           '/runs/selected/empty': np.empty((0, 3), dtype='f8'),
                           '/runs/selected/scalar': np.asarray(-0., dtype='f8'),
                           '/runs/selected/square': np.eye(4)})
        expected_paths = groups | set(arrays) | {'/runs/selected/intensity_alias', '/runs/selected/gain'}
        if family == 3:
            expected_paths.add('/runs/selected/targets')
        observed = {'/'}
        pending = [archive['/']]
        while pending:
            group = pending.pop()
            for name in group:
                full = group.name.rstrip('/') + '/' + name
                observed.add(full)
                link = group.get(name, getlink=True)
                require(isinstance(link, (h5py.HardLink, h5py.SoftLink)), 'unknown or external link')
                if isinstance(link, h5py.HardLink) and isinstance(group[name], h5py.Group):
                    pending.append(group[name])
        require(observed == expected_paths, 'selected output namespace differs from manual expectation')
        for name in groups:
            require(isinstance(archive[name], h5py.Group), 'expected group at ' + name)
        for name, expected in arrays.items():
            dataset = archive[name]
            require(isinstance(dataset, h5py.Dataset), 'expected dataset at ' + name)
            require(dataset.dtype == expected.dtype and dataset.shape == expected.shape, 'dtype or shape changed at ' + name)
            require(np.asarray(dataset[()]).tobytes() == expected.tobytes(), 'logical bytes changed at ' + name)
            if name.endswith('/intensity'):
                require(dataset.chunks == (2, 3) and dataset.compression == 'gzip' and dataset.compression_opts == 4, 'intensity storage changed')
            else:
                # h5py auto-chunks the original zero-length dataset.
                require(dataset.compression is None, 'unexpected compression at ' + name)
            require(not dataset.shuffle and not dataset.fletcher32 and dataset.scaleoffset is None, 'unexpected filter at ' + name)
            require(dataset.external is None and not dataset.is_virtual, 'nonlocal output storage')
        first = archive['/runs/selected/intensity']
        require(first.id == archive['/runs/selected/intensity_alias'].id, 'selected alias split')
        distinct = [int(h5py.h5o.get_info(archive[name].id).addr) for name in sorted(arrays)]
        require(len(set(distinct)) == len(distinct), 'distinct expected datasets conflated')
        soft = archive['/runs/selected'].get('gain', getlink=True)
        require(isinstance(soft, h5py.SoftLink) and soft.path == '/calibration/gain', 'soft target text changed')
        gain = archive['/calibration/gain']
        require(archive['/runs/selected/gain'].id == gain.id, 'soft calibration identity changed')
        reserved = {'CLASS', 'NAME', 'REFERENCE_LIST', 'DIMENSION_LIST', 'DIMENSION_LABELS'}
        for name in groups | set(arrays):
            obj = archive[name]
            ordinary = set(obj.attrs) - reserved if isinstance(obj, h5py.Dataset) else set(obj.attrs)
            expected = {'unit'} if name == '/calibration/gain' else ({'marker'} if name.endswith('/intensity') else set())
            if family == 3 and name == '/runs/selected':
                expected = {'calibration_ref'}
            require(ordinary == expected, 'ordinary attribute names changed at ' + name)
            if 'marker' in ordinary:
                require(obj.attrs['marker'] == name.split('/')[2], 'marker changed')
            if 'unit' in ordinary:
                require(obj.attrs['unit'] == 'dimensionless', 'unit changed')
        if family == 3:
            refs = archive['/runs/selected/targets']
            require(h5py.check_dtype(ref=refs.dtype) is h5py.Reference and refs.shape == (2,), 'reference dataset type/shape changed')
            values = refs[()]
            require(bool(values[0]) and archive[values[0]].id == gain.id and not values[1], 'reference targets changed')
            require(archive[archive['/runs/selected'].attrs['calibration_ref']].id == gain.id, 'reference attribute target changed')
            require('/calibration/gain_alias' not in archive, 'unselected calibration alias leaked')
        scales = ['/axes/time', '/axes/channel'] + (['/axes/alternate'] if family == 4 else [])
        expected_edges = Counter()
        for name in selected_runs:
            dataset = archive['/runs/' + name + '/intensity']
            require([axis.label for axis in dataset.dims] == ['time', ''], 'intensity labels changed')
            for axis, targets in ((0, ['/axes/time'] + (['/axes/alternate'] if family == 4 else [])), (1, ['/axes/channel'])):
                target_ids = Counter(int(h5py.h5o.get_info(archive[target].id).addr) for target in targets)
                high = Counter(int(h5py.h5o.get_info(item.id).addr) for item in dataset.dims[axis].values())
                raw = Counter(int(h5py.h5o.get_info(archive[ref].id).addr) for ref in dataset.attrs['DIMENSION_LIST'][axis])
                require(high == raw == target_ids, 'forward scale set changed')
                for target in targets:
                    expected_edges[(target, int(h5py.h5o.get_info(dataset.id).addr), axis)] += 1
        if family == 4:
            square = archive['/runs/selected/square']
            require([axis.label for axis in square.dims] == ['', ''], 'square labels changed')
            for axis in (0, 1):
                high = list(square.dims[axis].values())
                raw = square.attrs['DIMENSION_LIST'][axis]
                require(len(high) == len(raw) == 1 and high[0].id == archive['/axes/time'].id and archive[raw[0]].id == archive['/axes/time'].id, 'shared square-axis scale changed')
                expected_edges[('/axes/time', int(h5py.h5o.get_info(square.id).addr), axis)] += 1
        actual_reverse = Counter()
        for name in scales:
            scale = archive[name]
            require(scale.is_scale and scale.attrs['NAME'] == b'coordinate', 'scale identity/name changed')
            for row in scale.attrs.get('REFERENCE_LIST', []):
                target = archive[row['dataset']]
                actual_reverse[(name, int(h5py.h5o.get_info(target.id).addr), int(row['dimension']))] += 1
        require(actual_reverse == expected_edges, 'reverse consumers differ from retained forward edges')
        marker = marker_scan(archive)
        require(marker['complete'] and not marker['excluded_found'], 'excluded marker present or marker scan incomplete')
        return {'status': 'verified', 'family': family, 'file': Path(path).name,
                'namespace_matches': True, 'logical_values_and_types_match': True,
                'ordinary_attributes_match': True, 'selected_aliases_match': True,
                'explicit_references_match': True, 'forward_reverse_scales_match': True,
                'excluded_marker_absent': True, 'objects_checked': marker['objects_checked'],
                'scale_edges': sum(expected_edges.values()),
                'independent_of_product_implementation_and_plan': True}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('input', type=Path)
    result.add_argument('--family', type=int, choices=(1, 2, 3, 4), required=True)
    result.add_argument('--source', action='store_true', help='check the complete original source instead')
    return result


def worker_main(arguments):
    args = parser().parse_args(arguments)
    if args.source:
        from qualification.run_baselines import check_source
        report = check_source(args.input, args.family)
        report.update(status='verified', file=args.input.name, family=args.family)
    else:
        report = check_selected(args.input, args.family)
    print(json.dumps(report, sort_keys=True))
    return 0


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    parser().parse_args(arguments)
    return launch('qualification.check_frozen', arguments)


if __name__ == '__main__':
    raise SystemExit(main())
