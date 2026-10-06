"""Strict native HDF5 profile and canonical primitive encodings.

Import only in a resource-limited native worker. This module owns no graph walk.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import h5py
import numpy as np

from .model import CarryError

RESERVED = frozenset({'CLASS', 'NAME', 'DIMENSION_LIST', 'REFERENCE_LIST', 'DIMENSION_LABELS'})


def unsupported(message, path=None):
    raise CarryError('UNSUPPORTED', message, path)


def bounded_name(name, limits, path=None):
    try:
        length = len(name.encode('utf-8', 'strict'))
    except (UnicodeError, AttributeError):
        unsupported('Names must be valid UTF-8 text', path)
    if length > limits.max_name_bytes:
        raise CarryError('RESOURCE', 'Name exceeds max_name_bytes', path)


def _shape(shape, limits, path):
    if shape is not None and len(shape) > limits.max_rank:
        raise CarryError('RESOURCE', 'Dataspace rank exceeds max_rank', path)
    return None if shape is None else list(shape)


def dtype_descriptor(low_type, dtype, attribute=False):
    """Admit exact canonical low-level encodings, never dtype spelling alone."""
    dtype = np.dtype(dtype)
    if low_type.committed():
        unsupported('Committed/named datatype')
    ref = h5py.check_dtype(ref=dtype)
    if ref is not None:
        if ref is not h5py.Reference or not low_type.equal(h5py.h5t.STD_REF_OBJ):
            unsupported('Region or noncanonical reference datatype')
        return {'kind': 'reference', 'str': dtype.str, 'encoding': None}
    text = h5py.check_string_dtype(dtype)
    if text is not None and text.length is None:
        if not attribute:
            unsupported('Variable-length dataset text')
        if text.encoding not in ('ascii', 'utf-8') or not low_type.equal(h5py.h5t.py_create(dtype, logical=True)):
            unsupported('Noncanonical variable-length text encoding')
        return {'kind': 'text', 'str': dtype.str, 'encoding': text.encoding}
    admitted = ((dtype.kind in 'iu' and dtype.itemsize in (1, 2, 4, 8)) or
                (dtype.kind == 'b' and dtype.itemsize == 1) or
                (dtype.kind == 'f' and dtype.itemsize in (4, 8)) or
                (dtype.kind == 'c' and dtype.itemsize in (8, 16)) or
                (dtype.kind == 'S' and dtype.itemsize > 0))
    if not admitted or dtype.fields or dtype.subdtype or (h5py.check_enum_dtype(dtype) is not None and dtype.kind != 'b'):
        unsupported('Unsupported compound, enum, opaque, vlen, or primitive datatype')
    canonical = h5py.h5t.py_create(dtype, logical=True)
    if not low_type.equal(canonical):
        unsupported('Noncanonical low-level datatype encoding (precision, offset, padding, order, or mapping)')
    return {'kind': 'fixed', 'str': dtype.str, 'encoding': None}


def iter_blocks(shape, itemsize, chunk_bytes):
    """Yield C-order slabs with at most chunk_bytes of fixed-width values."""
    if shape is None or any(n == 0 for n in shape):
        return
    if itemsize > chunk_bytes:
        raise CarryError('RESOURCE', 'One element exceeds chunk_bytes')
    if not shape:
        yield ()
        return
    capacity = max(1, chunk_bytes // itemsize)
    # Full trailing dimensions keep the concatenated slabs in C order.
    tile = [1] * len(shape)
    for axis in range(len(shape) - 1, -1, -1):
        tile[axis] = min(shape[axis], capacity)
        capacity //= tile[axis]
        if tile[axis] != shape[axis]:
            break
    for starts in itertools.product(*(range(0, n, t) for n, t in zip(shape, tile))):
        yield tuple(slice(start, min(start + width, size)) for start, width, size in zip(starts, tile, shape))


def admit_payload_read(dataset):
    """Refuse undefined native values before any whole-dataset payload read.

    Metadata-only description remains usable, including for an empty typed crop
    of a nonempty source. Allocated storage cannot prove full initialization.
    """
    if (dataset.shape is not None and math.prod(dataset.shape) and
            dataset.id.get_create_plist().get_fill_time() == h5py.h5d.FILL_TIME_NEVER):
        unsupported('Nonempty FILL_TIME_NEVER payload reads are unqualified', dataset.name)


def payload_digest(dataset, limits):
    admit_payload_read(dataset)
    digest = hashlib.sha256()
    for selection in iter_blocks(dataset.shape, dataset.dtype.itemsize, limits.chunk_bytes):
        values = np.asarray(dataset[selection], dtype=dataset.dtype)
        digest.update(values.tobytes(order='C'))
    return digest.hexdigest()


def attribute_descriptor(obj, name, limits):
    bounded_name(name, limits, obj.name)
    aid = obj.attrs.get_id(name)
    path = obj.name + '@' + name
    try:
        dtype = dtype_descriptor(aid.get_type(), aid.dtype, attribute=True)
    except (TypeError, ValueError) as exc:
        unsupported('Attribute datatype has no qualified NumPy representation: ' + str(exc), path)
    except CarryError as exc:
        if exc.path is None:
            exc.path = path
        raise
    shape = _shape(aid.shape, limits, path)
    count = 0 if shape is None else math.prod(shape)
    if count * aid.dtype.itemsize > limits.max_attribute_bytes or (count and aid.get_storage_size() > limits.max_attribute_bytes):
        raise CarryError('RESOURCE', 'Attribute exceeds max_attribute_bytes', path)
    value = None
    if shape is not None and dtype['kind'] == 'fixed':
        value = np.asarray(obj.attrs[name], dtype=aid.dtype).tobytes(order='C').hex()
    elif shape is not None and dtype['kind'] == 'text':
        encoded = []
        used = 0
        for item in np.asarray(obj.attrs[name], dtype=object).reshape(-1):
            try:
                raw = item.encode(dtype['encoding'], 'strict') if isinstance(item, str) else bytes(item)
                # Validate bytes rather than silently changing their encoding.
                raw.decode(dtype['encoding'], 'strict')
            except UnicodeError:
                unsupported('Invalid variable-length attribute text encoding', path)
            used += len(raw)
            if used > limits.max_attribute_bytes:
                raise CarryError('RESOURCE', 'Variable-length attribute exceeds max_attribute_bytes', path)
            encoded.append(raw.hex())
        value = encoded
    return {'name': name, 'dtype': dtype, 'shape': shape, 'value': value}


def dataset_creation(dataset):
    dcpl = dataset.id.get_create_plist()
    if dcpl.get_external_count():
        unsupported('External raw dataset storage', dataset.name)
    layout_id = dcpl.get_layout()
    if layout_id not in (h5py.h5d.CONTIGUOUS, h5py.h5d.CHUNKED):
        unsupported('Virtual, compact, or unknown dataset layout', dataset.name)
    filters = []
    seen = set()
    for index in range(dcpl.get_nfilters()):
        ident, flags, values, _ = dcpl.get_filter(index)
        values = list(values)
        if ident not in (1, 2, 3) or ident in seen:
            unsupported('Unsupported or duplicate dataset filter', dataset.name)
        if flags not in (0, 1):
            unsupported('Unsupported dataset filter flags', dataset.name)
        if ((ident == 1 and (len(values) != 1 or not 0 <= values[0] <= 9)) or
                (ident == 2 and values != [dataset.dtype.itemsize]) or
                (ident == 3 and values)):
            unsupported('Noncanonical built-in filter parameters', dataset.name)
        seen.add(ident)
        filters.append({'id': ident, 'flags': flags, 'values': values})
    if h5py.check_dtype(ref=dataset.dtype) is not None:
        fill = dataset.fillvalue
        if fill is not None and bool(fill):
            unsupported('Non-null object-reference fill value', dataset.name)
        fill_hex = None
    else:
        fill_hex = np.asarray(dataset.fillvalue, dtype=dataset.dtype).tobytes().hex()
    return {'layout': 'chunked' if layout_id == h5py.h5d.CHUNKED else 'contiguous',
            'chunks': None if dataset.chunks is None else list(dataset.chunks),
            'maxshape': None if dataset.maxshape is None else list(dataset.maxshape),
            'filters': filters, 'fill': fill_hex}


def _reserved_aid(dataset, name, limits):
    aid = dataset.attrs.get_id(name)
    count = 0 if aid.shape is None else math.prod(aid.shape)
    if count * aid.dtype.itemsize > limits.max_attribute_bytes or (count and aid.get_storage_size() > limits.max_attribute_bytes):
        raise CarryError('RESOURCE', 'Reserved attribute exceeds max_attribute_bytes', dataset.name + '@' + name)
    if aid.get_type().committed():
        unsupported('Committed reserved attribute type', dataset.name)
    return aid


def _scale_text(dataset, name, limits):
    aid = _reserved_aid(dataset, name, limits)
    typ = aid.get_type()
    if (aid.shape != () or typ.get_class() != h5py.h5t.STRING or typ.is_variable_str() or
            typ.get_cset() != h5py.h5t.CSET_ASCII or typ.get_strpad() != h5py.h5t.STR_NULLTERM):
        unsupported('Malformed dimension-scale string metadata', dataset.name + '@' + name)
    raw = bytes(dataset.attrs[name])
    if typ.get_size() != len(raw) + 1:
        unsupported('Noncanonical dimension-scale string size', dataset.name + '@' + name)
    try:
        return raw.decode('utf-8', 'strict')
    except UnicodeError:
        unsupported('Invalid UTF-8 dimension-scale name', dataset.name + '@' + name)


def reserved_metadata(dataset, limits):
    """Validate reserved encodings only; callers independently decode edges."""
    present = RESERVED.intersection(dataset.attrs)
    if not isinstance(dataset, h5py.Dataset):
        if present:
            unsupported('Reserved dimension-scale attributes on group', dataset.name)
        return {}
    rank = len(dataset.shape) if dataset.shape is not None else 0
    scale_name = None
    if 'CLASS' in present:
        if _scale_text(dataset, 'CLASS', limits) != 'DIMENSION_SCALE' or 'NAME' not in present:
            unsupported('Unknown CLASS or missing scale NAME', dataset.name)
        scale_name = _scale_text(dataset, 'NAME', limits)
    elif 'NAME' in present or 'REFERENCE_LIST' in present:
        unsupported('Reserved scale metadata without DIMENSION_SCALE CLASS', dataset.name)
    if scale_name is not None and 'DIMENSION_LIST' in present:
        unsupported('A dimension scale cannot itself be a scale consumer', dataset.name)
    labels = [''] * rank
    if 'DIMENSION_LABELS' in present:
        aid = _reserved_aid(dataset, 'DIMENSION_LABELS', limits)
        expected = h5py.string_dtype('ascii')
        if aid.shape != (rank,) or not aid.get_type().equal(h5py.h5t.py_create(expected, logical=True)):
            unsupported('Malformed DIMENSION_LABELS datatype or shape', dataset.name)
        used = 0
        labels = []
        for item in dataset.attrs['DIMENSION_LABELS']:
            try:
                value = item.decode('utf-8', 'strict') if isinstance(item, bytes) else str(item)
                used += len(value.encode('utf-8', 'strict'))
            except UnicodeError:
                unsupported('Invalid dimension-label text encoding', dataset.name + '@DIMENSION_LABELS')
            if used > limits.max_attribute_bytes:
                raise CarryError('RESOURCE', 'Dimension labels exceed max_attribute_bytes', dataset.name)
            labels.append(value)
    if 'DIMENSION_LIST' in present:
        aid = _reserved_aid(dataset, 'DIMENSION_LIST', limits)
        expected = h5py.vlen_dtype(h5py.ref_dtype)
        if aid.shape != (rank,) or not aid.get_type().equal(h5py.h5t.py_create(expected, logical=True)):
            unsupported('Malformed DIMENSION_LIST datatype or shape', dataset.name)
        actual_bytes = sum(refs.nbytes for refs in dataset.attrs['DIMENSION_LIST'])
        if actual_bytes > limits.max_attribute_bytes:
            raise CarryError('RESOURCE', 'DIMENSION_LIST reference payload exceeds max_attribute_bytes', dataset.name)
    if 'REFERENCE_LIST' in present:
        aid = _reserved_aid(dataset, 'REFERENCE_LIST', limits)
        expected = np.dtype([('dataset', h5py.ref_dtype), ('dimension', 'u4')], align=True)
        if aid.shape is None or len(aid.shape) != 1 or not aid.get_type().equal(h5py.h5t.py_create(expected, logical=True)):
            unsupported('Malformed REFERENCE_LIST datatype or shape', dataset.name)
    return {'scale_name': scale_name, 'labels': labels}


def describe_object(obj, limits, hash_payload=True):
    result = {'attributes': []}
    reserved = reserved_metadata(obj, limits)
    names = list(obj.attrs)
    for name in names:
        bounded_name(name, limits, obj.name)
    for name in sorted(names):
        if name not in RESERVED:
            result['attributes'].append(attribute_descriptor(obj, name, limits))
    if isinstance(obj, h5py.Dataset):
        try:
            descriptor = dtype_descriptor(obj.id.get_type(), obj.dtype)
        except (TypeError, ValueError) as exc:
            unsupported('Datatype has no qualified NumPy representation: ' + str(exc), obj.name)
        except CarryError as exc:
            if exc.path is None:
                exc.path = obj.name
            raise
        if obj.dtype.itemsize > limits.chunk_bytes:
            raise CarryError('RESOURCE', 'One dataset element exceeds chunk_bytes', obj.name)
        result.update(dtype=descriptor, shape=_shape(obj.shape, limits, obj.name),
                      creation=dataset_creation(obj), **reserved)
        result['payload_sha256'] = (payload_digest(obj, limits) if hash_payload and descriptor['kind'] == 'fixed' else None)
    return result
