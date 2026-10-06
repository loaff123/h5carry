"""Strict, bounded, native-library-free rectangular selection request codec.

Validation here checks the language only. Source extents, identities, attachment
truth and dependency conflicts are checked by the source planner and verifier.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .model import CarryError, Limits, canonical_json, open_regular, validate_path
from .plan import _bad, _integer, _json_preflight, _keys, _unique_pairs


def _selection(value, limits):
    """Validate and copy one whole/box language value, without source inference."""
    if type(value) is not dict or type(value.get('kind')) is not str:
        _bad('selection must be a whole or box object')
    if value['kind'] == 'whole':
        _keys(value, ('kind',), 'whole selection')
        return {'kind': 'whole'}
    if value['kind'] != 'box':
        _bad('unsupported selection kind')
    _keys(value, ('kind', 'start', 'stop'), 'box selection')
    start, stop = value['start'], value['stop']
    if (type(start) is not list or type(stop) is not list or not start or
            len(start) != len(stop) or len(start) > limits.max_rank):
        _bad('box bounds must be nonempty equal-rank bounded lists')
    for first, last in zip(start, stop):
        _integer(first)
        _integer(last)
        if first > last:
            _bad('box start must not exceed its exclusive stop')
    return {'kind': 'box', 'start': start.copy(), 'stop': stop.copy()}


def validate_selection(request: dict, limits: Limits | None = None) -> dict:
    """Return a detached canonical request; never normalize source selections."""
    limits = limits or Limits()
    _keys(request, ('format', 'version', 'objects', 'scale_mappings', 'storage_policy'), 'selection request')
    if (request['format'] != 'h5carry-selection' or type(request['version']) is not int or
            request['version'] != 1):
        _bad('unsupported selection format/version')
    if request['storage_policy'] != 'fixed-snapshot-v1':
        _bad('unsupported storage policy')
    objects = request['objects']
    if type(objects) is not list or not objects:
        _bad('objects must be a nonempty list')
    if len(objects) > limits.max_objects:
        raise CarryError('RESOURCE', 'selection objects exceed list limit')
    canonical_objects, paths = [], set()
    for obj in objects:
        _keys(obj, ('path', 'selection'), 'requested object')
        path = validate_path(obj['path'], limits)
        if path in paths:
            _bad('duplicate requested object path')
        paths.add(path)
        canonical_objects.append({'path': path, 'selection': _selection(obj['selection'], limits)})
    mappings = request['scale_mappings']
    if type(mappings) is not list:
        _bad('scale_mappings must be a list')
    if len(mappings) > limits.max_edges:
        raise CarryError('RESOURCE', 'scale mappings exceed list limit')
    canonical_mappings, assertions = [], set()
    for mapping in mappings:
        _keys(mapping, ('consumer', 'axis', 'scale', 'mapping'), 'scale mapping')
        consumer = validate_path(mapping['consumer'], limits)
        scale = validate_path(mapping['scale'], limits)
        _integer(mapping['axis'], maximum=limits.max_rank - 1)
        if mapping['mapping'] != 'index':
            _bad('only explicit index scale mappings are supported')
        key = consumer, mapping['axis'], scale
        if key in assertions:
            _bad('duplicate scale mapping assertion')
        assertions.add(key)
        canonical_mappings.append({'consumer': consumer, 'axis': mapping['axis'],
                                   'scale': scale, 'mapping': 'index'})
    value = {'format': 'h5carry-selection', 'version': 1,
             'objects': sorted(canonical_objects, key=lambda obj: obj['path']),
             'scale_mappings': sorted(canonical_mappings, key=lambda item: (item['consumer'], item['axis'], item['scale'])),
             'storage_policy': 'fixed-snapshot-v1'}
    if len(canonical_json(value)) > limits.max_plan_bytes:
        raise CarryError('RESOURCE', 'selection request exceeds byte limit')
    return value


def load_selection(path: str | Path, limits: Limits | None = None) -> dict:
    """Load bounded strict UTF-8 JSON from a regular local file, rejecting duplicates."""
    limits = limits or Limits()
    try:
        with open_regular(path) as stream:
            data = stream.read(limits.max_plan_bytes + 1)
    except OSError as exc:
        raise CarryError('INVALID', f'cannot read selection request: {exc.strerror}') from exc
    if len(data) > limits.max_plan_bytes:
        raise CarryError('RESOURCE', 'selection request exceeds byte limit')
    _json_preflight(data)

    def integer(text):
        if len(text) > 20:
            _bad('JSON integer exceeds digit limit')
        return int(text)

    def finite_float(text):
        value = float(text)
        if not math.isfinite(value):
            _bad('nonfinite JSON number')
        return value

    try:
        value = json.loads(data.decode('utf-8', 'strict'), object_pairs_hook=_unique_pairs,
                           parse_int=integer, parse_float=finite_float,
                           parse_constant=lambda _: _bad('nonfinite JSON number'))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CarryError('INVALID', 'malformed selection JSON') from exc
    return validate_selection(value, limits)
