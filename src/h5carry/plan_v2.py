"""Strict typed-snapshot plan envelope, isolated from version-1 semantics.

Source descriptors retain source shape and storage but never whole-payload
hashes. Only the output graph's logical payload is charged to the payload cap.
No native library is imported here and no scientific mapping is inferred.
"""
from __future__ import annotations

import copy
from pathlib import Path

from .model import CarryError, Limits, canonical_json, fingerprint, validate_path
from .plan import HEX64, _bad, _dtype, _integer, _keys, _metadata, _shape, _validate_plan_v1
from .selection import _selection, validate_selection


def _dataset_structure(metadata, limits):
    """Check source/output dataspace and creation consistency without payload sums."""
    shape, creation = metadata['shape'], metadata['creation']
    rank = 0 if shape is None else len(shape)
    if len(metadata['labels']) != rank:
        _bad('dimension labels must match dataspace rank')
    maximum, chunks = creation['maxshape'], creation['chunks']
    if shape is None:
        if maximum is not None or chunks is not None or creation['layout'] != 'contiguous':
            _bad('null dataspace must have null maximum and contiguous storage')
    else:
        if type(maximum) is not list or len(maximum) != rank:
            _bad('maximum shape must match dataspace rank')
        for size, limit in zip(shape, maximum):
            if limit is not None and size > limit:
                _bad('current extent exceeds maximum extent')
        if creation['layout'] == 'contiguous':
            if chunks is not None or any(limit is None for limit in maximum):
                _bad('contiguous storage cannot have chunks or unlimited maximum')
        else:
            if not rank or type(chunks) is not list or len(chunks) != rank or any(size < 1 for size in chunks):
                _bad('chunks must contain a positive extent for every axis')
            if any(limit is not None and limit != 0 and chunk > limit for chunk, limit in zip(chunks, maximum)):
                _bad('chunk extent exceeds finite nonzero maximum')
    itemsize = _dtype(metadata['dtype'])
    fill = creation['fill']
    if metadata['dtype']['kind'] == 'fixed' and (type(fill) is not str or len(fill) != 2 * itemsize):
        _bad('fixed fill byte count must match dtype')
    if metadata['dtype']['kind'] == 'reference' and fill is not None:
        _bad('reference fill must be null')
    filters, seen = creation['filters'], set()
    if filters and creation['layout'] != 'chunked':
        _bad('filters require chunked storage')
    for filt in filters:
        ident, flags, values = filt['id'], filt['flags'], filt['values']
        if ident in seen or flags not in (0, 1):
            _bad('duplicate or noncanonical filter')
        if ((ident == 1 and (len(values) != 1 or not 0 <= values[0] <= 9)) or
                (ident == 2 and values != [itemsize]) or (ident == 3 and values)):
            _bad('noncanonical built-in filter parameters')
        seen.add(ident)


def validate_plan_v2(plan: dict, limits: Limits | None = None) -> dict:
    """Validate exact source/output descriptors and declared proper-crop changes."""
    ceiling = limits or Limits()
    _keys(plan, ('format', 'version', 'source', 'request', 'limits', 'graph',
                 'source_objects', 'selections', 'transformations'), 'v2 plan')
    if plan['format'] != 'h5carry-plan' or type(plan['version']) is not int or plan['version'] != 2:
        _bad('unsupported v2 plan format/version')
    active = Limits.from_dict(plan['limits'])
    for name, maximum in ceiling.to_dict().items():
        if active.to_dict()[name] > maximum:
            raise CarryError('RESOURCE', f'plan {name} exceeds invocation ceiling')
    request = validate_selection(plan['request'], active)
    if request != plan['request']:
        _bad('original request must be canonical')
    # A separate v1 decoder checks the unchanged output graph format, including
    # namespace closure and its payload sum. It never sees source extents.
    graph_envelope = {'format': 'h5carry-plan', 'version': 1, 'source': plan['source'],
                      'selections': ['/'], 'limits': plan['limits'], 'graph': plan['graph']}
    if (type(plan['graph']) is dict and type(plan['graph'].get('payload_bytes')) is int
            and plan['graph']['payload_bytes'] > active.max_payload_bytes):
        raise CarryError('RESOURCE', 'selected output payload exceeds byte limit')
    _validate_plan_v1(graph_envelope, ceiling)
    graph = plan['graph']
    if graph['references'] != sorted(graph['references'], key=lambda edge: (
            edge['owner'], edge['attribute'] is not None, edge['attribute'] or '', edge['index'])):
        _bad('reference edges must be sorted canonically')
    if graph['scales'] != sorted(graph['scales'], key=lambda edge: (edge['consumer'], edge['axis'], edge['scale'])):
        _bad('scale edges must be sorted canonically')
    if any(entries != sorted(set(entries)) for entries in graph['reasons'].values()):
        _bad('inclusion reasons must be sorted and unique')
    output = {obj['id']: obj for obj in plan['graph']['objects']}
    source_descriptors = plan['source_objects']
    if type(source_descriptors) is not list:
        _bad('source_objects must be a list')
    if len(source_descriptors) > active.max_objects:
        raise CarryError('RESOURCE', 'source objects exceed list limit')
    source = {}
    for obj in source_descriptors:
        _keys(obj, ('id', 'kind', 'metadata'), 'source object')
        ident = validate_path(obj['id'], active)
        if ident in source or obj['kind'] not in ('group', 'dataset'):
            _bad('duplicate source object ID or invalid kind')
        if ident not in output or obj['kind'] != output[ident]['kind']:
            _bad('source descriptors must match retained output identities and kinds')
        _metadata(obj['metadata'], obj['kind'], active)
        if obj['kind'] == 'dataset':
            if obj['metadata']['payload_sha256'] is not None:
                _bad('source payload digest must be null')
            _dataset_structure(obj['metadata'], active)
            _dataset_structure(output[ident]['metadata'], active)
            digest = output[ident]['metadata']['payload_sha256']
            if obj['metadata']['dtype']['kind'] == 'fixed' and (type(digest) is not str or not HEX64.fullmatch(digest)):
                _bad('fixed output dataset requires a selected-payload digest')
            if obj['metadata']['dtype']['kind'] == 'reference' and digest is not None:
                _bad('reference payload must be represented by graph edges')
        source[ident] = obj
    if list(source) != sorted(source) or set(source) != set(output):
        _bad('source descriptors must be sorted and cover every retained object exactly once')
    entries = plan['selections']
    if type(entries) is not list:
        _bad('normalized selections must be a list')
    if len(entries) > active.max_objects:
        raise CarryError('RESOURCE', 'normalized selections exceed list limit')
    selections, expected_changes = {}, []
    for entry in entries:
        _keys(entry, ('id', 'selection'), 'normalized selection')
        ident = validate_path(entry['id'], active)
        if ident in selections or ident not in source:
            _bad('duplicate or missing normalized selection identity')
        selection = _selection(entry['selection'], active)
        selections[ident] = selection
        original, result = source[ident], output[ident]
        if original['kind'] == 'group':
            if selection['kind'] != 'whole' or result['metadata'] != original['metadata']:
                _bad('groups must be preserved whole with unchanged metadata')
            continue
        old, new = original['metadata'], result['metadata']
        expected = copy.deepcopy(old)
        expected['payload_sha256'] = new['payload_sha256']
        if selection['kind'] == 'box':
            shape = old['shape']
            if old['dtype']['kind'] != 'fixed' or shape is None or not shape or len(shape) != len(selection['start']):
                _bad('box selections require a simple fixed primitive dataset of matching rank')
            start, stop = selection['start'], selection['stop']
            if any(last > extent for last, extent in zip(stop, shape)):
                _bad('box stop exceeds source extent')
            if start == [0] * len(shape) and stop == shape:
                _bad('full-extent box must normalize to whole')
            selected_shape = [last - first for first, last in zip(start, stop)]
            old_creation = old['creation']
            chunks = old_creation['chunks']
            selected_chunks = None if chunks is None else [min(chunk, max(1, extent)) for chunk, extent in zip(chunks, selected_shape)]
            expected['shape'] = selected_shape
            expected['creation'].update(maxshape=selected_shape, chunks=selected_chunks)
            expected_changes.append({'id': ident, 'source_shape': shape, 'output_shape': selected_shape,
                                     'source_maxshape': old_creation['maxshape'], 'output_maxshape': selected_shape,
                                     'source_chunks': chunks, 'output_chunks': selected_chunks})
        if new != expected:
            _bad('output metadata does not match normalized selection and fixed snapshot storage')
    if list(selections) != sorted(selections) or set(selections) != set(source):
        _bad('normalized selections must be sorted and cover every retained object exactly once')
    transformations = plan['transformations']
    if type(transformations) is not list:
        _bad('transformations must be a list')
    if len(transformations) > active.max_objects:
        raise CarryError('RESOURCE', 'transformations exceed list limit')
    for change in transformations:
        _keys(change, ('id', 'source_shape', 'output_shape', 'source_maxshape',
                       'output_maxshape', 'source_chunks', 'output_chunks'), 'transformation')
        validate_path(change['id'], active)
        for name in ('source_shape', 'output_shape', 'output_maxshape'):
            _shape(change[name], active, allow_null=False)
        for name in ('source_chunks', 'output_chunks'):
            _shape(change[name], active)
        maximum = change['source_maxshape']
        if type(maximum) is not list or len(maximum) > active.max_rank:
            _bad('source maximum must be a bounded dimension list')
        for extent in maximum:
            if extent is not None:
                _integer(extent)
    if transformations != expected_changes:
        _bad('transformations must describe exactly the sorted proper crops')
    if len(canonical_json(plan)) > min(ceiling.max_plan_bytes, active.max_plan_bytes):
        raise CarryError('RESOURCE', 'serialized plan exceeds byte limit')
    return plan


def make_selection_plan(source: str | Path, request: dict, limits: Limits | None = None) -> dict:
    """Fingerprint source and wrap the supervised native typed planner result."""
    from .supervisor import run_native
    limits = limits or Limits()
    request = validate_selection(request, limits)
    before = fingerprint(source)
    result = run_native({'operation': 'plan_v2', 'source': str(source),
                         'request': request, 'limits': limits.to_dict()})
    if not result['ok']:
        raise CarryError(**result['error'])
    if fingerprint(source) != before:
        raise CarryError('SOURCE_CHANGED', 'source changed while planning; close all writers')
    native = result['result']
    _keys(native, ('graph', 'source_objects', 'selections', 'transformations'), 'native v2 plan result')
    value = {'format': 'h5carry-plan', 'version': 2, 'source': before, 'request': request,
             'limits': limits.to_dict(), **native}
    return validate_plan_v2(value, limits)
