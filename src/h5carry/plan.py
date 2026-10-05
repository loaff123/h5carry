"""Strict bounded JSON plan codec. No HDF5 import or native parsing here."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import posixpath
import re
import tempfile

from .model import CarryError, Limits, canonical_json, fingerprint, open_regular, validate_path

HEX64 = re.compile(r'[0-9a-f]{64}\Z')
HEX = re.compile(r'(?:[0-9a-f]{2})*\Z')


def _bad(message):
    raise CarryError('INVALID', message)


def _keys(value, names, context):
    if type(value) is not dict or set(value) != set(names):
        _bad(f'{context} must contain exactly: {", ".join(sorted(names))}')


def _integer(value, minimum=0, maximum=2**63-1):
    if type(value) is not int or not minimum <= value <= maximum:
        _bad('integer outside supported bounds')


def _text(value, maximum, context='text'):
    if type(value) is not str or '\x00' in value:
        _bad(f'{context} must be a string without NUL')
    try:
        size = len(value.encode('utf-8'))
    except UnicodeError:
        _bad(f'{context} must be UTF-8')
    if size > maximum:
        raise CarryError('RESOURCE', f'{context} exceeds byte limit')


def _shape(value, limits, allow_null=True):
    if value is None and allow_null:
        return
    if type(value) is not list or len(value) > limits.max_rank:
        _bad('shape must be a bounded dimension list')
    for extent in value:
        _integer(extent)


def _dtype(value):
    _keys(value, ('kind','str','encoding'), 'dtype')
    if value['kind'] not in ('fixed','reference','text'):
        _bad('unsupported dtype kind')
    _text(value['str'], 80, 'dtype string')
    if value['encoding'] not in (None, 'ascii', 'utf-8'):
        _bad('invalid dtype encoding')
    if value['kind']=='fixed':
        match=re.fullmatch(r'([<>|])([iubfcS])([1-9][0-9]{0,9})',value['str'])
        if not match or value['encoding'] is not None:
            _bad('invalid fixed dtype descriptor')
        order,kind,width=match.groups(); width=int(width)
        allowed={'i':(1,2,4,8),'u':(1,2,4,8),'b':(1,),'f':(4,8),'c':(8,16)}
        if kind!='S' and width not in allowed[kind]:
            _bad('unsupported fixed dtype width')
        if (kind in ('b','S') or width==1) != (order=='|'):
            _bad('noncanonical dtype byte-order spelling')
        return width
    if value['str']!='|O' or (value['kind']=='reference' and value['encoding'] is not None) or (value['kind']=='text' and value['encoding'] is None):
        _bad('invalid reference/text dtype descriptor')
    return 8


def _metadata(value, kind, limits):
    expected = {'attributes'}
    if kind == 'dataset':
        expected |= {'dtype','shape','creation','scale_name','labels','payload_sha256'}
    _keys(value, expected, 'object metadata')
    attrs = value['attributes']
    if type(attrs) is not list or len(attrs) > limits.max_edges:
        _bad('attributes must be a bounded list')
    names = []
    for attr in attrs:
        _keys(attr, ('name','dtype','shape','value'), 'attribute descriptor')
        _text(attr['name'], limits.max_name_bytes, 'attribute name')
        names.append(attr['name'])
        itemsize=_dtype(attr['dtype']); _shape(attr['shape'], limits)
        count=0 if attr['shape'] is None else math.prod(attr['shape'])
        if count*itemsize>limits.max_attribute_bytes:
            raise CarryError('RESOURCE','attribute shape exceeds byte limit')
        if attr['shape'] is None and attr['value'] is not None:
            _bad('null-dataspace attribute value must be null')
        encoded = canonical_json(attr['value'])
        if len(encoded) > 4 * limits.max_attribute_bytes + 1024:
            raise CarryError('RESOURCE', 'attribute descriptor exceeds byte limit')
        if attr['dtype']['kind'] == 'reference' and attr['value'] is not None:
            _bad('reference values must be represented as graph edges')
        if attr['dtype']['kind'] == 'fixed' and attr['shape'] is not None and (type(attr['value']) is not str or not HEX.fullmatch(attr['value'])):
            _bad('fixed attribute must contain hexadecimal bytes')
        if attr['dtype']['kind']=='fixed' and attr['shape'] is not None and len(attr['value'])!=2*count*itemsize:
            _bad('fixed attribute byte count does not match its dtype/shape')
        if attr['dtype']['kind']=='text' and attr['shape'] is not None:
            entries=attr['value']
            if type(entries) is not list or len(entries)!=count or any(type(entry) is not str or not HEX.fullmatch(entry) for entry in entries):
                _bad('text attribute must contain one hex byte string per element')
            if sum(len(entry)//2 for entry in entries)>limits.max_attribute_bytes:
                raise CarryError('RESOURCE','text attribute exceeds byte limit')
            try:
                for entry in entries:
                    bytes.fromhex(entry).decode(attr['dtype']['encoding'],'strict')
            except UnicodeError:
                _bad('text attribute has invalid encoding')
    if names != sorted(set(names)):
        _bad('attribute names must be sorted and unique')
    if kind != 'dataset':
        return
    itemsize=_dtype(value['dtype']); _shape(value['shape'], limits)
    if itemsize>limits.chunk_bytes:
        raise CarryError('RESOURCE','dataset element exceeds read-block ceiling')
    if value['dtype']['kind'] == 'text':
        _bad('variable-length text datasets are outside this profile')
    if value['scale_name'] is not None:
        _text(value['scale_name'], limits.max_attribute_bytes, 'scale name')
    if type(value['labels']) is not list or len(value['labels']) > limits.max_rank:
        _bad('labels must be a bounded list')
    for label in value['labels']:
        _text(label, limits.max_attribute_bytes, 'dimension label')
    digest = value['payload_sha256']
    if digest is not None and (type(digest) is not str or not HEX64.fullmatch(digest)):
        _bad('invalid logical payload digest')
    creation = value['creation']
    _keys(creation, ('layout','chunks','maxshape','filters','fill'), 'creation properties')
    if creation['layout'] not in ('contiguous','chunked'):
        _bad('unsupported storage layout')
    _shape(creation['chunks'], limits)
    maximum = creation['maxshape']
    if maximum is not None:
        if type(maximum) is not list or len(maximum) > limits.max_rank:
            _bad('maxshape must be bounded')
        for extent in maximum:
            if extent is not None:
                _integer(extent)
    if type(creation['filters']) is not list or len(creation['filters']) > 3:
        _bad('invalid filter list')
    for filt in creation['filters']:
        _keys(filt, ('id','flags','values'), 'filter')
        if type(filt['id']) is not int or filt['id'] not in (1,2,3):
            _bad('unsupported filter')
        _integer(filt['flags'], maximum=2**32-1)
        if type(filt['values']) is not list or len(filt['values']) > 32:
            _bad('filter values must be bounded')
        for item in filt['values']:
            _integer(item, maximum=2**32-1)
    fill = creation['fill']
    if fill is not None and (type(fill) is not str or not HEX.fullmatch(fill)):
        _bad('fill must be hexadecimal bytes or null')


def validate_plan(plan: dict, limits: Limits | None = None) -> dict:
    ceiling = limits or Limits()
    _keys(plan, ('format','version','source','selections','limits','graph'), 'plan')
    if plan['format'] != 'h5carry-plan' or type(plan['version']) is not int or plan['version'] != 1:
        _bad('unsupported plan format/version')
    active = Limits.from_dict(plan['limits'])
    for key, max_value in ceiling.to_dict().items():
        if active.to_dict()[key] > max_value:
            raise CarryError('RESOURCE', f'plan {key} exceeds invocation ceiling')
    _keys(plan['source'], ('sha256','size'), 'source fingerprint')
    if type(plan['source']['sha256']) is not str or not HEX64.fullmatch(plan['source']['sha256']):
        _bad('source sha256 must be 64 lowercase hexadecimal characters')
    _integer(plan['source']['size'])
    roots = plan['selections']
    if type(roots) is not list or not roots or len(roots) > active.max_objects:
        _bad('selections must be a nonempty bounded list')
    for root in roots:
        validate_path(root, active)
    if roots != sorted(set(roots)):
        _bad('selections must be sorted and unique')
    graph = plan['graph']
    _keys(graph, ('objects','links','references','scales','payload_bytes','reasons'), 'graph')
    for name, maximum in (('objects',active.max_objects), ('links',active.max_edges),
                          ('references',active.max_edges), ('scales',active.max_edges)):
        if type(graph[name]) is not list or len(graph[name]) > maximum:
            raise CarryError('RESOURCE', f'graph {name} exceeds list limit')
    if sum(len(graph[k]) for k in ('links','references','scales')) > active.max_edges:
        raise CarryError('RESOURCE', 'graph edges exceed aggregate limit')
    _integer(graph['payload_bytes'], maximum=active.max_payload_bytes)
    objects = {}
    for obj in graph['objects']:
        _keys(obj, ('id','kind','metadata'), 'object')
        ident = validate_path(obj['id'], active)
        if ident in objects or obj['kind'] not in ('group','dataset'):
            _bad('duplicate object ID or invalid kind')
        _metadata(obj['metadata'], obj['kind'], active)
        objects[ident] = obj
    computed_payload=0
    for obj in objects.values():
        if obj['kind']=='dataset':
            metadata=obj['metadata']
            count=0 if metadata['shape'] is None else math.prod(metadata['shape'])
            computed_payload+=count*_dtype(metadata['dtype'])
            if computed_payload>active.max_payload_bytes:
                raise CarryError('RESOURCE','dataset metadata exceeds selected payload ceiling')
    if computed_payload!=graph['payload_bytes']:
        _bad('declared payload_bytes does not match dataset metadata')
    if '/' not in objects or objects['/']['kind'] != 'group':
        _bad('root group is required')
    if list(objects) != sorted(objects):
        _bad('objects must be sorted by ID')
    links, hard = {}, {'/':'/'}
    for link in graph['links']:
        _keys(link, ('path','kind','target'), 'link')
        path = validate_path(link['path'], active)
        if path == '/' or path in links or link['kind'] not in ('hard','soft'):
            _bad('duplicate link or invalid kind')
        if link['kind'] == 'hard':
            validate_path(link['target'], active)
            if link['target'] not in objects:
                _bad('hard link has missing object target')
            hard[path] = link['target']
        else:
            _text(link['target'], active.max_name_bytes, 'soft-link target')
            if not link['target']:
                _bad('soft-link target is empty')
        links[path] = link
    if list(links) != sorted(links):
        _bad('links must be sorted by path')
    for path in links:
        parent = posixpath.dirname(path)
        if parent not in hard or objects[hard[parent]]['kind'] != 'group':
            _bad('link has no retained parent group')
    for ident, obj in objects.items():
        aliases = sorted(path for path, target in hard.items() if target == ident)
        if not aliases or ident != aliases[0]:
            _bad('object ID must be its first retained hard path')
        if obj['kind'] == 'group' and aliases != [ident]:
            _bad('group aliases are outside supported profile')
    ref_keys = set()
    for edge in graph['references']:
        _keys(edge, ('owner','attribute','index','target'), 'reference')
        validate_path(edge['owner'], active)
        if edge['target'] is not None:
            validate_path(edge['target'], active)
        if edge['owner'] not in objects or (edge['target'] is not None and edge['target'] not in objects):
            _bad('reference edge has missing object')
        if edge['attribute'] is not None:
            _text(edge['attribute'], active.max_name_bytes, 'reference attribute')
        _integer(edge['index'], maximum=active.max_edges)
        key = (edge['owner'], edge['attribute'], edge['index'])
        if key in ref_keys:
            _bad('duplicate reference slot')
        ref_keys.add(key)
    scale_keys = set()
    for edge in graph['scales']:
        _keys(edge, ('consumer','axis','scale'), 'scale edge')
        for key in ('consumer','scale'):
            validate_path(edge[key], active)
        if any(edge[k] not in objects or objects[edge[k]]['kind'] != 'dataset' for k in ('consumer','scale')):
            _bad('scale edge has missing dataset')
        _integer(edge['axis'], maximum=active.max_rank-1)
        key = (edge['consumer'], edge['axis'], edge['scale'])
        if key in scale_keys:
            _bad('duplicate scale edge')
        scale_keys.add(key)
    reasons = graph['reasons']
    if type(reasons) is not dict or set(reasons) != set(objects):
        _bad('reasons must cover exactly the retained objects')
    for entries in reasons.values():
        if type(entries) is not list or not entries or len(entries) > active.max_edges:
            _bad('reason chain must be a nonempty bounded list')
        for reason in entries:
            _text(reason, active.max_name_bytes * 2, 'inclusion reason')
    if len(canonical_json(plan)) > min(ceiling.max_plan_bytes, active.max_plan_bytes):
        raise CarryError('RESOURCE', 'serialized plan exceeds byte limit')
    return plan


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _bad('duplicate JSON object key')
        result[key] = value
    return result


def _json_preflight(data: bytes):
    # Prevent recursive-decoder exhaustion before allocating nested containers.
    depth, quoted, escape, containers = 0, False, False, 0
    for byte in data:
        if quoted:
            if escape:
                escape = False
            elif byte == 92:
                escape = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91,123):
            depth += 1
            containers += 1
            if containers > 500_000:
                raise CarryError('RESOURCE', 'JSON container count exceeds 500000')
            if depth > 32:
                raise CarryError('RESOURCE', 'JSON nesting exceeds 32 levels')
        elif byte in (93,125):
            depth -= 1
            if depth < 0:
                _bad('invalid JSON nesting')


def load_plan(path: str | Path, limits: Limits | None = None) -> dict:
    limits = limits or Limits()
    try:
        with open_regular(path) as stream:
            data = stream.read(limits.max_plan_bytes + 1)
    except OSError as exc:
        raise CarryError('INVALID', f'cannot read plan: {exc.strerror}') from exc
    if len(data) > limits.max_plan_bytes:
        raise CarryError('RESOURCE', 'plan exceeds byte limit')
    _json_preflight(data)
    def integer(text):
        if len(text) > 20:
            _bad('JSON integer exceeds digit limit')
        return int(text)
    try:
        value = json.loads(data, object_pairs_hook=_unique_pairs, parse_int=integer,
                           parse_constant=lambda _: _bad('nonfinite JSON number'))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CarryError('INVALID', 'malformed plan JSON') from exc
    return validate_plan(value, limits)


def save_plan(plan: dict, path: str | Path) -> None:
    validate_plan(plan)
    # Link publication is atomic and fails if any destination already exists.
    destination = Path(path)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix='.h5carry-plan-', dir=destination.parent)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical_json(plan)); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, destination, follow_symlinks=False)
    except OSError as exc:
        raise CarryError('INVALID', f'cannot publish plan without clobbering: {exc.strerror}') from exc
    finally:
        if temporary is not None:
            os.unlink(temporary)


def make_plan(source: str | Path, selections, limits: Limits | None = None) -> dict:
    from .supervisor import run_native
    limits = limits or Limits()
    roots = sorted(set(validate_path(root, limits) for root in selections))
    if not roots or len(roots) > limits.max_objects:
        _bad('at least one bounded selection is required')
    before = fingerprint(source)
    result = run_native({'operation':'plan','source':str(source),'selections':roots,'limits':limits.to_dict()})
    if not result['ok']:
        raise CarryError(**result['error'])
    if fingerprint(source) != before:
        raise CarryError('SOURCE_CHANGED', 'source changed while planning; close all writers')
    value = {'format':'h5carry-plan','version':1,'source':before,'selections':roots,
             'limits':limits.to_dict(),'graph':result['result']}
    return validate_plan(value)
