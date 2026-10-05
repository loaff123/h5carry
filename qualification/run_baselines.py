"""Reproduce original benign h5py API baselines and credit preserved behavior.

This compares only four frozen fixture families plus a plain-copy control.
It does not run h5copy/h5diff executables, infer a general copying defect, or
fetch or open third-party HDF5 inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples._bounded import launch, require_native_limits
from examples.three_run_archive import create_family, create_plain_control


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_source(path, family):
    """Reopen source and check hand-specified scientific and relation expectations."""
    require_native_limits()
    import h5py
    import numpy as np

    with h5py.File(path, 'r') as archive:
        expected_names = {'/', '/axes', '/axes/time', '/axes/channel', '/calibration',
                          '/calibration/gain', '/runs', '/runs/selected', '/runs/other', '/runs/excluded',
                          '/runs/selected/intensity', '/runs/other/intensity', '/runs/excluded/intensity',
                          '/runs/selected/intensity_alias', '/runs/selected/gain'}
        if family == 3:
            expected_names.update(('/calibration/gain_alias', '/runs/selected/targets'))
        if family == 4:
            expected_names.update(('/runs/selected/empty', '/runs/selected/scalar',
                                   '/runs/selected/square', '/axes/alternate'))
        names = {'/'}
        pending = [archive['/']]
        while pending:
            group = pending.pop()
            for name in group:
                path_name = group.name.rstrip('/') + '/' + name
                names.add(path_name)
                link = group.get(name, getlink=True)
                _require(not isinstance(link, h5py.ExternalLink), 'unexpected external source link')
                if isinstance(link, h5py.HardLink) and isinstance(group[name], h5py.Group):
                    pending.append(group[name])
        _require(names == expected_names, 'frozen source namespace changed')
        time = archive['/axes/time']
        channel = archive['/axes/channel']
        gain = archive['/calibration/gain']
        _require(time.dtype == np.dtype('f8') and np.array_equal(time[()], [0.0, 0.1, 0.2, 0.3]), 'time values changed')
        _require(channel.dtype == np.dtype('i2') and np.array_equal(channel[()], [1, 2, 3]), 'channel values changed')
        _require(gain.dtype == np.dtype('f8') and np.array_equal(gain[()], [1.5, 2.5, 3.5]), 'gain values changed')
        _require(gain.attrs['unit'] == 'dimensionless', 'gain unit changed')
        selected = archive['/runs/selected']
        _require(selected['intensity'].id == selected['intensity_alias'].id, 'source alias identity changed')
        _require(isinstance(selected.get('gain', getlink=True), h5py.SoftLink), 'source soft link changed')
        _require(selected.get('gain', getlink=True).path == '/calibration/gain', 'source soft target changed')
        _require(selected['gain'].id == gain.id, 'source soft target identity changed')
        expected = Counter()
        actual_forward = Counter()
        for run, offset in (('selected', 0), ('other', 100), ('excluded', 200)):
            dataset = archive['/runs/' + run + '/intensity']
            _require(dataset.dtype == np.dtype('i4') and np.array_equal(dataset[()], np.arange(12).reshape(4, 3) + offset), run + ' values changed')
            _require(dataset.attrs['marker'] == run, run + ' marker changed')
            _require(dataset.chunks == (2, 3) and dataset.compression == 'gzip' and dataset.compression_opts == 4, run + ' storage changed')
            _require(dataset.dims[0].label == 'time' and dataset.dims[1].label == '', run + ' labels changed')
            for axis, paths in ((0, ['/axes/time'] + (['/axes/alternate'] if family == 4 and run == 'selected' else [])),
                                (1, ['/axes/channel'])):
                dataset_id = int(h5py.h5o.get_info(dataset.id).addr)
                expected_ids = Counter(int(h5py.h5o.get_info(archive[p].id).addr) for p in paths)
                high = Counter(int(h5py.h5o.get_info(scale.id).addr) for scale in dataset.dims[axis].values())
                raw = Counter(int(h5py.h5o.get_info(archive[ref].id).addr) for ref in dataset.attrs['DIMENSION_LIST'][axis])
                _require(high == raw == expected_ids, run + ' forward scale targets changed')
                for p in paths:
                    expected[(p, dataset_id, axis)] += 1
                    actual_forward[(p, dataset_id, axis)] += 1
        if family == 3:
            _require(gain.id == archive['/calibration/gain_alias'].id, 'calibration source alias changed')
            refs = selected['targets'][()]
            _require(refs.shape == (2,) and bool(refs[0]) and not refs[1], 'source nonnull/null references changed')
            _require(archive[refs[0]].id == gain.id and archive[selected.attrs['calibration_ref']].id == gain.id, 'source reference target changed')
        scales = ['/axes/time', '/axes/channel']
        if family == 4:
            _require(selected['empty'].shape == (0, 3) and selected['empty'].dtype == np.dtype('f8'), 'empty source changed')
            _require(selected['scalar'].shape == () and np.asarray(selected['scalar'][()]).tobytes() == np.asarray(-0.0, dtype='f8').tobytes(), 'negative zero changed')
            _require(np.array_equal(selected['square'][()], np.eye(4)), 'square values changed')
            _require(np.array_equal(archive['/axes/alternate'][()], np.arange(4, dtype='u2')), 'alternate scale changed')
            scales.append('/axes/alternate')
            square = selected['square']
            square_id = int(h5py.h5o.get_info(square.id).addr)
            for axis in (0, 1):
                actual = list(square.dims[axis].values())
                raw = square.attrs['DIMENSION_LIST'][axis]
                _require(len(actual) == len(raw) == 1 and actual[0].id == time.id and archive[raw[0]].id == time.id, 'square scale relation changed')
                expected[('/axes/time', square_id, axis)] += 1
                actual_forward[('/axes/time', square_id, axis)] += 1
        reverse = Counter()
        for p in scales:
            scale = archive[p]
            _require(scale.is_scale and scale.attrs['NAME'] == b'coordinate', 'source scale designation changed')
            for record in scale.attrs.get('REFERENCE_LIST', []):
                target = archive[record['dataset']]
                reverse[(p, int(h5py.h5o.get_info(target.id).addr), int(record['dimension']))] += 1
        _require(reverse == actual_forward == expected, 'source forward/reverse scale relations disagree')
        return {'valid': True, 'namespace_matches': True, 'ordinary_values_preserved': True,
                'source_aliases_resolve': True, 'source_references_resolve': True,
                'forward_reverse_scales_match': True, 'scale_edges': sum(expected.values()),
                'reopened_read_only': True}


def _observe_copy(path, family, expanded):
    import h5py
    import numpy as np
    from examples.analyze_selected import marker_scan

    with h5py.File(path, 'r') as archive:
        runs = ('selected', 'other') if family == 2 else ('selected',)
        values_ok = True
        for name in runs:
            offset = 100 if name == 'other' else 0
            values_ok &= np.array_equal(archive['/runs/' + name + '/intensity'][()], np.arange(12).reshape(4, 3) + offset)
        if family == 4:
            values_ok &= archive['/runs/selected/empty'].shape == (0, 3)
            values_ok &= np.asarray(archive['/runs/selected/scalar'][()]).tobytes() == np.asarray(-0.0).tobytes()
            values_ok &= np.array_equal(archive['/runs/selected/square'][()], np.eye(4))
        errors = []
        observations = []
        def inspect_scales(name, obj):
            if isinstance(obj, h5py.Dataset) and not obj.is_scale:
                try:
                    paths = [[scale.name for scale in axis.values()] for axis in obj.dims]
                    observations.append({'dataset': '/' + name, 'scales': paths})
                except (RuntimeError, KeyError, ValueError) as exc:
                    errors.append({'dataset': '/' + name, 'type': type(exc).__name__, 'message': str(exc)})
        archive.visititems(inspect_scales)
        selected = archive['/runs/selected']
        result = {'method': 'expanded' if expanded else 'ordinary', 'copy_created': True,
                  'ordinary_values_preserved': bool(values_ok),
                  'intensity_alias_preserved': selected['intensity'].id == selected['intensity_alias'].id,
                  'scale_iteration_errors': errors, 'scale_iteration_observations': observations,
                  'all_named_scale_iterations_succeeded': not errors}
        try:
            calibration = selected['gain']
            result['calibration_link_resolves_to_expected_values'] = bool(np.array_equal(calibration[()], [1.5, 2.5, 3.5]))
        except (KeyError, ValueError):
            result['calibration_link_resolves_to_expected_values'] = False
        if family == 3:
            refs = selected['targets'][()]
            first_ok = bool(refs[0]) and np.array_equal(archive[refs[0]][()], [1.5, 2.5, 3.5])
            attribute = selected.attrs['calibration_ref']
            attribute_ok = bool(attribute) and np.array_equal(archive[attribute][()], [1.5, 2.5, 3.5])
            result['explicit_references'] = {'dataset_calibration_resolves': bool(first_ok),
                                            'null_reference_preserved': not bool(refs[1]),
                                            'attribute_calibration_resolves': bool(attribute_ok),
                                            'dataset_first_is_null': not bool(refs[0]),
                                            'attribute_is_null': not bool(attribute)}
        else:
            result['explicit_references'] = {'applicable': False}
        result['excluded_marker_scan'] = marker_scan(archive)
        return result


def _plain_checks(path):
    import h5py
    import numpy as np
    with h5py.File(path, 'r') as archive:
        values = archive['/data/values']
        return {'values': bool(np.array_equal(values[()], np.arange(12).reshape(4, 3))),
                'dtype': values.dtype == np.dtype('>i4'),
                'dataset_attribute': values.attrs['unit'] == 'counts',
                'group_attribute': archive['/data'].attrs['kind'] == 'plain control',
                'hard_alias': values.id == archive['/data/alias'].id,
                'soft_target_values': bool(np.array_equal(archive['/data/soft'][()], values[()])),
                'chunk_filter': values.chunks == (2, 3) and values.compression == 'gzip'}


def run_baselines(destination):
    require_native_limits()
    import h5py
    import numpy as np

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    report = {'status': 'complete', 'scope': 'four original frozen families and a separate plain-copy control',
              'comparison_api': 'h5py.Group.copy',
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                          'h5py': h5py.__version__, 'hdf5': h5py.version.hdf5_version,
                          'platform': platform.system()},
              'process_limits': {'address_space_bytes': 2 * 1024 ** 3, 'cpu_seconds': 120,
                                 'wall_seconds': 180, 'file_size_bytes': 1024 ** 3,
                                 'dynamic_plugins_disabled': True},
              'families': [], 'plain_controls': [],
              'supplementary_cli_tools': {name: {'available': shutil.which(name) is not None, 'run': False}
                                          for name in ('h5copy', 'h5diff')}}
    for family in range(1, 5):
        source = destination / f'family-{family}-source.h5'
        selections = create_family(source, family)
        before = _digest(source)
        source_check = check_source(source, family)
        entry = {'family': family, 'source_file': source.name, 'source_sha256': before,
                 'selections': selections, 'source': source_check, 'copies': []}
        for expanded in (False, True):
            name = 'expanded' if expanded else 'ordinary'
            output = destination / f'family-{family}-{name}.h5'
            with h5py.File(source, 'r') as archive, h5py.File(output, 'x') as copied:
                for selection in selections:
                    parent, _, leaf = selection.rpartition('/')
                    group = copied.require_group(parent)
                    archive.copy(selection, group, name=leaf, expand_soft=expanded,
                                 expand_external=expanded, expand_refs=expanded)
            observation = _observe_copy(output, family, expanded)
            observation['output_file'] = output.name
            observation['output_sha256'] = _digest(output)
            entry['copies'].append(observation)
        entry['source_rechecked_after_copy'] = check_source(source, family)
        entry['source_unchanged'] = before == _digest(source)
        _require(entry['source_unchanged'], 'baseline copy changed its source')
        report['families'].append(entry)
    source = destination / 'plain-source.h5'
    create_plain_control(source)
    initial = _digest(source)
    report['plain_source_checks'] = _plain_checks(source)
    _require(all(report['plain_source_checks'].values()), 'plain source validation failed')
    for expanded in (False, True):
        name = 'expanded' if expanded else 'ordinary'
        output = destination / f'plain-{name}.h5'
        with h5py.File(source, 'r') as archive, h5py.File(output, 'x') as copied:
            archive.copy('/data', copied, expand_soft=expanded,
                         expand_external=expanded, expand_refs=expanded)
        report['plain_controls'].append({'method': name, 'checks': _plain_checks(output),
                                         'output_file': output.name, 'output_sha256': _digest(output)})
    report['plain_source_unchanged'] = initial == _digest(source)
    report['limitations'] = [
        'Only the stated original fixtures and exact runtime are measured.',
        'Copy successes and failed scale iterations are reported separately.',
        'Expanded soft links may become hard links; successful plain controls do not claim soft-link text retention.',
        'No excluded marker found is not a proof of general reverse-reference expansion behavior.',
        'No speed, novelty, universal compatibility, parser-security, or current-upstream regression claim.',
        'h5copy and h5diff command-line tools were not run; these are h5py API observations.',
    ]
    (destination / 'baseline-results.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('output_directory', type=Path)
    return result


def worker_main(arguments):
    args = parser().parse_args(arguments)
    report = run_baselines(args.output_directory)
    summary = {'status': report['status'], 'runtime': report['runtime'],
               'families': len(report['families']), 'copies': 8,
               'copies_with_scale_iteration_errors': sum(bool(c['scale_iteration_errors']) for f in report['families'] for c in f['copies']),
               'plain_controls_passed': all(all(item['checks'].values()) for item in report['plain_controls']),
               'report': 'baseline-results.json'}
    print(json.dumps(summary, sort_keys=True))
    return 0


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    parser().parse_args(arguments)
    return launch('qualification.run_baselines', arguments)


if __name__ == '__main__':
    raise SystemExit(main())
