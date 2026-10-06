"""Planner/writer rectangular transfers; verifier has separate arithmetic."""
from __future__ import annotations
import math
from .model import CarryError


def iter_transfers(start, stop, itemsize, cap, chunks=None):
    """C-order, rank-preserving row tiles with O(rank) iterator state.

    Each chunked tile touches one original native chunk. The accounting bound
    is decoded chunk exposure for this read, not total native memory usage.
    """
    if any(a == b for a, b in zip(start, stop)):
        return
    if itemsize > cap or (chunks and math.prod(chunks) * itemsize > cap):
        raise CarryError('RESOURCE', 'Native chunk or element exceeds chunk_bytes before payload read')
    if not start:
        yield (), ()
        return
    current = list(start)
    capacity = cap // itemsize
    while True:
        end = min(stop[-1], current[-1] + capacity)
        if chunks:
            end = min(end, (current[-1] // chunks[-1] + 1) * chunks[-1])
        left = tuple(slice(value, value + 1) for value in current[:-1]) + (slice(current[-1], end),)
        right = tuple(slice(part.start - origin, part.stop - origin) for part, origin in zip(left, start))
        yield left, right
        current[-1] = end
        axis = len(current) - 1
        while current[axis] == stop[axis]:
            current[axis] = start[axis]
            axis -= 1
            if axis < 0:
                return
            current[axis] += 1


def guard_read(dataset, limits):
    if dataset.shape is None or 0 in dataset.shape:
        return
    dcpl = dataset.id.get_create_plist()
    # NEVER permits undefined unallocated bytes, not a reproducible snapshot.
    import h5py
    if dcpl.get_fill_time() == h5py.h5d.FILL_TIME_NEVER:
        raise CarryError('UNSUPPORTED', 'Typed snapshots cannot read FILL_TIME_NEVER datasets', dataset.name)
    if dataset.chunks and math.prod(dataset.chunks) * dataset.dtype.itemsize > limits.chunk_bytes:
        raise CarryError('RESOURCE', 'Uncompressed native chunk exceeds chunk_bytes before payload read', dataset.name)


def transfers(dataset, selection, limits):
    if dataset.shape is None:
        return
    if selection['kind'] == 'whole':
        start, stop = [0] * dataset.ndim, list(dataset.shape)
    else:
        start, stop = selection['start'], selection['stop']
    if any(a == b for a, b in zip(start, stop)):
        return
    guard_read(dataset, limits)
    yield from iter_transfers(start, stop, dataset.dtype.itemsize, limits.chunk_bytes, dataset.chunks)
