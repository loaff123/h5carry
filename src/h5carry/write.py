"""Allocate, copy, remap, and rebuild a declared native HDF5 graph."""
from __future__ import annotations

from collections import defaultdict
import math

import h5py
import numpy as np

from .model import CarryError, validate_path
from . import profile


def _hard_object(file, path, limits):
    """Never let a plan path cause implicit external/soft-link traversal."""
    validate_path(path, limits)
    obj = file['/']
    if path == '/':
        return obj
    for name in path.strip('/').split('/'):
        if not isinstance(obj, h5py.Group):
            raise CarryError('INVALID', 'Plan object path crosses a dataset', path)
        link = obj.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise CarryError('UNSUPPORTED', 'Plan object path is not a source hard path', path)
        obj = obj[name]
    return obj


def write_staging(source, graph, staging, limits):
    """Write only the caller-owned staging file; verification is separate."""
    with h5py.File(source, 'r') as src:
        records = {item['id']: item for item in graph['objects']}
        source_objects = {}
        actual_payload_bytes = 0
        # The plan is untrusted. Account the actual admitted source dataspaces
        # before opening the staging output, allocating objects, or copying any
        # dataset values. A forged graph total or shape cannot raise this budget.
        for path in sorted(records):
            original = _hard_object(src, path, limits)
            profile.describe_object(original, limits, hash_payload=False)
            expected_group = records[path]['kind'] == 'group'
            if expected_group != isinstance(original, h5py.Group):
                raise CarryError('INVALID', 'Plan/source object kind mismatch', path)
            if isinstance(original, h5py.Dataset):
                count = 0 if original.shape is None else math.prod(original.shape)
                actual_payload_bytes += count * original.dtype.itemsize
                if actual_payload_bytes > limits.max_payload_bytes:
                    raise CarryError('RESOURCE', 'Actual source payload exceeds max_payload_bytes before staging', path)
            source_objects[path] = original
        with h5py.File(staging, 'w', track_order=True) as out:
            destination = {'/': out['/']}
            for path in sorted(records, key=lambda value: (value.count('/'), value)):
                record = records[path]
                original = source_objects[path]
                if record['kind'] == 'group':
                    if not isinstance(original, h5py.Group):
                        raise CarryError('INVALID', 'Plan/source object kind mismatch', path)
                    if path != '/':
                        destination[path] = out.create_group(path, track_order=True)
                else:
                    if not isinstance(original, h5py.Dataset):
                        raise CarryError('INVALID', 'Plan/source object kind mismatch', path)
                    # Exact qualified type, dataspace, and creation list. Reference
                    # fill has already been required to be null; no raw ref leaks.
                    ident = h5py.h5d.create(out.id, path.encode('utf-8'), original.id.get_type(),
                                           original.id.get_space(), dcpl=original.id.get_create_plist().copy())
                    destination[path] = h5py.Dataset(ident)
            for link in graph['links']:
                if link['kind'] == 'hard' and link['path'] != link['target']:
                    out[link['path']] = destination[link['target']]
            for link in graph['links']:
                if link['kind'] == 'soft':
                    out[link['path']] = h5py.SoftLink(link['target'])
            refs = defaultdict(dict)
            for edge in graph['references']:
                refs[(edge['owner'], edge['attribute'])][edge['index']] = edge['target']

            def reference_array(owner, attribute, shape):
                result = np.empty(shape, dtype=h5py.ref_dtype)
                mapping = refs[(owner, attribute)]
                for index in range(result.size):
                    if index not in mapping:
                        raise CarryError('INVALID', 'Missing planned attribute reference', owner)
                    target = mapping[index]
                    result.flat[index] = h5py.Reference() if target is None else destination[target].ref
                return result

            for path in sorted(records):
                original, copied = source_objects[path], destination[path]
                if isinstance(original, h5py.Dataset):
                    offset = 0
                    for selection in profile.iter_blocks(original.shape, original.dtype.itemsize, limits.chunk_bytes):
                        if h5py.check_dtype(ref=original.dtype) is h5py.Reference:
                            # Construct bounded blocks in plan index order, never copy
                            # file-local reference bytes from the original file.
                            block_shape = tuple(s.stop - s.start for s in selection) if selection else ()
                            block = np.empty(block_shape, dtype=h5py.ref_dtype)
                            mapping = refs[(path, None)]
                            for index in range(block.size):
                                if offset not in mapping:
                                    raise CarryError('INVALID', 'Missing planned dataset reference', path)
                                target = mapping[offset]
                                block.flat[index] = h5py.Reference() if target is None else destination[target].ref
                                offset += 1
                            copied[selection] = block
                        else:
                            copied[selection] = original[selection]
                for attr in records[path]['metadata']['attributes']:
                    name = attr['name']
                    aid = original.attrs.get_id(name)
                    if aid.shape is None:
                        data = h5py.Empty(aid.dtype)
                    elif h5py.check_dtype(ref=aid.dtype) is h5py.Reference:
                        data = reference_array(path, name, aid.shape)
                    else:
                        data = original.attrs[name]
                    copied.attrs.create(name, data, dtype=aid.dtype)
            for path in sorted(records):
                meta = records[path]['metadata']
                if records[path]['kind'] == 'dataset' and meta['scale_name'] is not None:
                    destination[path].make_scale(meta['scale_name'])
            for edge in graph['scales']:
                destination[edge['consumer']].dims[edge['axis']].attach_scale(destination[edge['scale']])
            for path in sorted(records):
                if records[path]['kind'] == 'dataset':
                    for axis, label in enumerate(records[path]['metadata']['labels']):
                        if label:
                            destination[path].dims[axis].label = label
            out.flush()
