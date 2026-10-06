"""Typed snapshot reconstruction, after source-bound decision rederivation."""
from __future__ import annotations

from collections import defaultdict
import math

import h5py
import numpy as np

from .model import CarryError, validate_path, fingerprint, canonical_json
from .slicing import transfers
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


def write_staging(source, plan, staging, limits):
    """Write only the caller-owned staging file; verification is separate."""
    from .scan_v2 import make_plan_native
    if fingerprint(source) != plan['source']:
        raise CarryError('SOURCE_CHANGED', 'Source differs from typed plan')
    derived = make_plan_native(source, plan['request'], limits)
    for field in ('graph','source_objects','selections','transformations'):
        if canonical_json(derived[field]) != canonical_json(plan[field]):
            raise CarryError('MISMATCH', 'Typed plan differs from freshly derived source: ' + field)
    if fingerprint(source) != plan['source']:
        raise CarryError('SOURCE_CHANGED', 'Source changed before typed staging')
    graph = derived['graph']
    selections = {item['id']:item['selection'] for item in derived['selections']}
    with h5py.File(source, 'r', rdcc_nbytes=0) as src:
        records = {item['id']:item for item in graph['objects']}
        source_objects = {path:_hard_object(src,path,limits) for path in records}
        with h5py.File(staging, 'w', track_order=True, rdcc_nbytes=0) as out:
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
                    dcpl = original.id.get_create_plist().copy()
                    space = original.id.get_space()
                    if selections[path]['kind'] == 'box':
                        shape = tuple(record['metadata']['shape'])
                        space = h5py.h5s.create_simple(shape,shape)
                        if original.chunks:
                            dcpl.set_chunk(tuple(record['metadata']['creation']['chunks']))
                    ident = h5py.h5d.create(out.id, path.encode('utf-8'), original.id.get_type(),
                                           space, dcpl=dcpl)
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
                    for selection, destination_selection in transfers(original,selections[path],limits):
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
                            copied[destination_selection] = block
                        else:
                            copied[destination_selection] = original[selection]
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
