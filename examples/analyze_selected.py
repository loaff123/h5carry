"""Analyze a relocated original two-run export without opening its source.

The calculation multiplies each of the three intensity channels by its gain
and sums every sample: 173 for selected, 3173 for other, and 3346 combined.
This consumer does not import the H5Carry planner, writer, or verifier.
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


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def marker_scan(archive):
    """Visit distinct named and reference-reachable objects, including anonymous ones."""
    import h5py
    import numpy as np

    def references(value):
        if isinstance(value, h5py.RegionReference):
            raise ValueError('region references are outside the original example')
        if isinstance(value, h5py.Reference):
            if value:
                yield value
        elif isinstance(value, np.ndarray):
            if value.dtype.hasobject or value.dtype.fields:
                for item in value.flat:
                    yield from references(item)
        elif isinstance(value, np.void) and value.dtype.fields:
            for field in value.dtype.names:
                yield from references(value[field])
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from references(item)

    pending = [archive['/']]
    visited = set()
    excluded_found = []
    unresolved = []
    while pending:
        obj = pending.pop()
        token = int(h5py.h5o.get_info(obj.id).addr)
        if token in visited:
            continue
        visited.add(token)
        if len(visited) > 1000:
            raise ValueError('original example object budget exceeded')
        values = [obj.attrs[name] for name in obj.attrs]
        if isinstance(obj, h5py.Group):
            for name in obj:
                link = obj.get(name, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    raise ValueError('external links are not allowed in this example')
                try:
                    pending.append(obj[name])
                except (KeyError, ValueError) as exc:
                    unresolved.append(type(exc).__name__ + ': unresolved link')
        elif isinstance(obj, h5py.Dataset):
            marker = obj.attrs.get('marker')
            if isinstance(marker, bytes):
                marker = marker.decode('utf-8')
            if marker == 'excluded':
                excluded_found.append(obj.name or '<anonymous>')
            if h5py.check_dtype(ref=obj.dtype) is not None:
                values.append(obj[()])
            elif obj.shape == (4, 3) and obj.dtype.kind in 'iuf':
                if np.array_equal(obj[()], np.arange(12).reshape(4, 3) + 200):
                    excluded_found.append(obj.name or '<anonymous>')
        for value in values:
            for reference in references(value):
                try:
                    target = archive[reference]
                    if target is None:
                        raise ValueError('null target')
                    pending.append(target)
                except (KeyError, ValueError, RuntimeError):
                    unresolved.append('unresolved object reference')
    return {'objects_checked': len(visited), 'excluded_found': sorted(set(excluded_found)),
            'unresolved': unresolved, 'complete': not unresolved}


def analyze(path, require_demo=False):
    require_native_limits()
    import h5py
    import numpy as np

    with h5py.File(path, 'r') as archive:
        scan = marker_scan(archive)
        _require(not scan['excluded_found'], 'excluded marker-bearing data is present')
        _require(scan['complete'], 'not every retained link/reference resolves')
        _require('excluded' not in archive['/runs'], 'excluded run is present')
        _require(set(archive['/runs']) == {'selected', 'other'}, 'unexpected run names')
        for scale_path, expected_values, label in (('/axes/time', [0.0, 0.1, 0.2, 0.3], 'time'),
                                                   ('/axes/channel', [1, 2, 3], 'channel')):
            scale = archive[scale_path]
            _require(np.array_equal(scale[()], expected_values), label + ' coordinate values changed')
            _require(scale.is_scale and scale.attrs['NAME'] == b'coordinate', label + ' scale designation changed')
        gain = archive['/calibration/gain']
        _require(np.array_equal(gain[()], [1.5, 2.5, 3.5]), 'calibration gains changed')
        _require(archive['/runs/selected/gain'].id == gain.id, 'calibration soft link has wrong identity')
        link = archive['/runs/selected'].get('gain', getlink=True)
        _require(isinstance(link, h5py.SoftLink) and link.path == '/calibration/gain', 'calibration soft-link text changed')
        intensity = archive['/runs/selected/intensity']
        _require(intensity.id == archive['/runs/selected/intensity_alias'].id, 'selected alias split')
        totals = {}
        references_checked = 0
        expected_reverse = {'/axes/time': Counter(), '/axes/channel': Counter()}
        for name, offset in (('selected', 0), ('other', 100)):
            run = archive['/runs/' + name]
            data = run['intensity']
            _require(np.array_equal(data[()], np.arange(12).reshape(4, 3) + offset), name + ' intensity changed')
            _require(data.dims[0].label == 'time', name + ' time label changed')
            for axis, scale_path in ((0, '/axes/time'), (1, '/axes/channel')):
                scales = list(data.dims[axis].values())
                _require(len(scales) == 1 and scales[0].id == archive[scale_path].id, name + ' shared scale identity changed')
                raw = data.attrs['DIMENSION_LIST'][axis]
                _require(len(raw) == 1 and archive[raw[0]].id == archive[scale_path].id, name + ' raw forward scale changed')
                expected_reverse[scale_path][(int(h5py.h5o.get_info(data.id).addr), axis)] += 1
            if 'targets' in run:
                targets = run['targets'][()]
                _require(targets.shape == (2,) and bool(targets[0]) and not targets[1], name + ' reference/null contract changed')
                _require(archive[targets[0]].id == gain.id, name + ' reference dataset has wrong calibration')
                _require(archive[run.attrs['calibration_ref']].id == gain.id, name + ' reference attribute has wrong calibration')
                references_checked += 3
            elif require_demo:
                raise ValueError(name + ' supplemental reference dataset is missing')
            totals[name] = float(np.sum(data[()].astype('f8') * gain[()]))
        for scale_path, expected in expected_reverse.items():
            reverse = Counter()
            scale = archive[scale_path]
            for item in scale.attrs.get('REFERENCE_LIST', []):
                target = archive[item['dataset']]
                reverse[(int(h5py.h5o.get_info(target.id).addr), int(item['dimension']))] += 1
            _require(reverse == expected, scale_path + ' reverse consumers changed')
        if require_demo:
            _require(archive.attrs['title'] == 'Original three-run measurement archive', 'root title changed')
            _require(archive['/runs'].attrs['instrument'] == 'simulated intensity detector', 'ancestor metadata changed')
            _require(archive[archive['/runs'].attrs['calibration_ref']].id == gain.id, 'ancestor calibration reference changed')
            references_checked += 1
        _require(totals == {'selected': 173.0, 'other': 3173.0}, 'calibrated totals changed')
        return {'status': 'verified', 'file': Path(path).name, 'calibrated_totals': totals,
                'combined_total': sum(totals.values()), 'selected_alias_preserved': True,
                'soft_calibration_identity_preserved': True, 'shared_scales_preserved': True,
                'forward_reverse_scales_match': True, 'explicit_reference_checks': references_checked,
                'excluded_marker_absent': True, 'objects_checked': scan['objects_checked'],
                'source_opened': False, 'supplemental_demo': require_demo}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('input', type=Path)
    result.add_argument('--require-demo', action='store_true')
    return result


def worker_main(arguments):
    args = parser().parse_args(arguments)
    print(json.dumps(analyze(args.input, args.require_demo), sort_keys=True))
    return 0


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    parser().parse_args(arguments)
    return launch('examples.analyze_selected', arguments)


if __name__ == '__main__':
    raise SystemExit(main())
