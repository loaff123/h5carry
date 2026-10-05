"""Generate original benign measurement fixtures; never downloads HDF5 data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples._bounded import launch, require_native_limits


def create_family(destination, family=2):
    """Write one unchanged frozen fixture family, inside a bounded worker.

    The original data/paths are deliberately tiny and deterministic. Family 2
    selects two runs, family 3 adds explicit references, and family 4 adds
    multiple scales, repeated scale names, a scalar and an empty dataset.
    """
    require_native_limits()
    if family not in (1, 2, 3, 4):
        raise ValueError('family must be 1, 2, 3, or 4')
    import h5py
    import numpy as np

    with h5py.File(destination, 'x') as archive:
        axes = archive.create_group('axes')
        time = axes.create_dataset('time', data=np.asarray((0.0, 0.1, 0.2, 0.3)))
        channel = axes.create_dataset('channel', data=np.asarray((1, 2, 3), dtype='i2'))
        time.make_scale('coordinate')
        channel.make_scale('coordinate')
        calibration = archive.create_group('calibration')
        gain = calibration.create_dataset('gain', data=np.asarray((1.5, 2.5, 3.5)))
        gain.attrs['unit'] = 'dimensionless'
        runs = archive.create_group('runs')
        for name, offset in (('selected', 0), ('other', 100), ('excluded', 200)):
            run = runs.create_group(name)
            intensity = run.create_dataset('intensity', data=np.arange(12, dtype='i4').reshape(4, 3) + offset,
                                           chunks=(2, 3), compression='gzip')
            intensity.attrs['marker'] = name
            intensity.dims[0].label = 'time'
            intensity.dims[0].attach_scale(time)
            intensity.dims[1].attach_scale(channel)
        selected = runs['selected']
        selected['intensity_alias'] = selected['intensity']
        selected['gain'] = h5py.SoftLink('/calibration/gain')
        if family == 3:
            calibration['gain_alias'] = gain
            targets = selected.create_dataset('targets', (2,), dtype=h5py.ref_dtype)
            targets[...] = [gain.ref, h5py.Reference()]
            selected.attrs['calibration_ref'] = gain.ref
        if family == 4:
            selected.create_dataset('empty', (0, 3), dtype='f8')
            selected.create_dataset('scalar', data=np.float64(-0.0))
            alternate = axes.create_dataset('alternate', data=np.arange(4, dtype='u2'))
            alternate.make_scale('coordinate')
            selected['intensity'].dims[0].attach_scale(alternate)
            square = selected.create_dataset('square', data=np.eye(4))
            for axis in square.dims:
                axis.attach_scale(time)
    selections = ['/runs/selected']
    if family == 2:
        selections.append('/runs/other')
    return selections


def create_demo(destination):
    """Supplemental sharing example; not one of the four frozen families."""
    selections = create_family(destination, 2)
    import h5py
    with h5py.File(destination, 'r+') as archive:
        gain = archive['/calibration/gain']
        archive.attrs['title'] = 'Original three-run measurement archive'
        archive['/runs'].attrs['instrument'] = 'simulated intensity detector'
        archive['/runs'].attrs['calibration_ref'] = gain.ref
        for name in ('selected', 'other'):
            run = archive['/runs/' + name]
            targets = run.create_dataset('targets', (2,), dtype=h5py.ref_dtype,
                                         chunks=(2,), compression='gzip', compression_opts=7,
                                         maxshape=(None,))
            targets[...] = [gain.ref, h5py.Reference()]
            run.attrs['calibration_ref'] = gain.ref
    return selections


def create_plain_control(destination):
    """Original plain case that ordinary and expanded h5py copying handles."""
    require_native_limits()
    import h5py
    import numpy as np
    with h5py.File(destination, 'x') as archive:
        group = archive.create_group('data')
        group.attrs['kind'] = 'plain control'
        values = group.create_dataset('values', data=np.arange(12, dtype='>i4').reshape(4, 3),
                                      chunks=(2, 3), compression='gzip')
        values.attrs['unit'] = 'counts'
        group['alias'] = values
        group['soft'] = h5py.SoftLink('/data/values')
    return ['/data']


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('output', type=Path)
    result.add_argument('--family', type=int, choices=(1, 2, 3, 4), default=2)
    result.add_argument('--demo', action='store_true', help='add explicit reference and ancestor metadata to family 2')
    return result


def worker_main(arguments):
    args = parser().parse_args(arguments)
    if args.demo and args.family != 2:
        raise ValueError('--demo extends family 2 only')
    selections = create_demo(args.output) if args.demo else create_family(args.output, args.family)
    print(json.dumps({'status': 'generated', 'file': args.output.name,
                      'fixture': 'supplemental-demo' if args.demo else f'frozen-family-{args.family}',
                      'selections': selections, 'expected_calibrated_totals': [173, 3173] if args.family == 2 else [173]}))
    return 0


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    parser().parse_args(arguments)
    return launch('examples.three_run_archive', arguments)


if __name__ == '__main__':
    raise SystemExit(main())
