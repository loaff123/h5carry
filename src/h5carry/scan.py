"""Bounded source inventory and forward-only retained-graph projection.

Native-only module. Transient object addresses never leave this process.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import math
import posixpath
import time

import h5py
import numpy as np

from .model import CarryError, canonical_json, validate_path
from . import profile


def _identity(obj):
    return int(h5py.h5o.get_info(obj.id).addr)


class _Inventory:
    def __init__(self, file, limits):
        self.file, self.limits = file, limits
        self.deadline = time.monotonic() + limits.discovery_seconds
        self.nodes = {}
        self.links = {}
        self.paths = defaultdict(list)
        self.children = defaultdict(list)
        self.references = defaultdict(list)
        self.scales = []
        self.soft_targets = {}
        self._resolved = {}
        self.edges = 0
        self.root = _identity(file['/'])
        self._walk()
        self._relations()

    def check(self):
        if time.monotonic() > self.deadline:
            raise CarryError('RESOURCE', 'Discovery deadline exceeded')

    def edge(self, count=1):
        self.edges += count
        if self.edges > self.limits.max_edges:
            raise CarryError('RESOURCE', 'Source links/references/attributes exceed max_edges')
        self.check()

    def _walk(self):
        todo = deque([('/', self.file['/'])])
        self.paths[self.root].append('/')
        while todo:
            self.check()
            path, obj = todo.popleft()
            uid = _identity(obj)
            if uid in self.nodes:
                if isinstance(obj, h5py.Group):
                    profile.unsupported('Group hard alias or cycle', path)
                continue
            if len(self.nodes) >= self.limits.max_objects:
                raise CarryError('RESOURCE', 'Source object count exceeds max_objects', path)
            if not isinstance(obj, (h5py.Group, h5py.Dataset)):
                profile.unsupported('Named datatype or unknown object class', path)
            self.edge(len(obj.attrs))
            meta = profile.describe_object(obj, self.limits, hash_payload=False)
            self.nodes[uid] = {'object': obj, 'kind': 'group' if isinstance(obj, h5py.Group) else 'dataset', 'metadata': meta}
            if isinstance(obj, h5py.Group):
                if len(obj) + self.edges > self.limits.max_edges:
                    raise CarryError('RESOURCE', 'Group link count exceeds max_edges', path)
                names = list(obj)
                for name in names:
                    profile.bounded_name(name, self.limits, path)
                for name in sorted(names):
                    child = ('/' if path == '/' else path + '/') + name
                    validate_path(child, self.limits)
                    link = obj.get(name, getlink=True)
                    self.edge()
                    self.children[path].append(child)
                    if isinstance(link, h5py.ExternalLink):
                        profile.unsupported('External link (target not opened)', child)
                    if isinstance(link, h5py.SoftLink):
                        profile.bounded_name(link.path, self.limits, child)
                        if '\x00' in link.path or not link.path:
                            profile.unsupported('Invalid soft-link text', child)
                        self.links[child] = {'kind': 'soft', 'target': link.path}
                    elif isinstance(link, h5py.HardLink):
                        linked = obj[name]
                        target = _identity(linked)
                        self.links[child] = {'kind': 'hard', 'target': target}
                        self.paths[target].append(child)
                        todo.append((child, linked))
                    else:
                        profile.unsupported('Unknown link class', child)
        for paths in self.paths.values():
            paths.sort()
        for path, link in self.links.items():
            if link['kind'] == 'soft':
                target_text = link['target']
                target_path = target_text if target_text.startswith('/') else posixpath.dirname(path) + '/' + target_text
                target, chain = self.resolve(target_path)
                self.soft_targets[path] = (target, chain)
                # All namespace entries have already been classified and every
                # external link rejected, so native usability can now be checked.
                try:
                    actual = self.file[path]
                except (KeyError, ValueError, RuntimeError) as exc:
                    profile.unsupported('Soft link cannot resolve under the native traversal limit: ' + str(exc), path)
                expected = self.root if target == '/' else self.links[target]['target']
                if _identity(actual) != expected:
                    profile.unsupported('Native and lexical soft-link targets disagree', path)

    def resolve(self, path):
        """Resolve lexically inventoried links using an explicit active stack."""
        def components(value):
            # HDF5 treats '.' and repeated separators as lexical no-ops, but
            # does not interpret '..' as a parent-directory operator.
            result = [part for part in value.split('/') if part not in ('', '.')]
            if '..' in result:
                profile.unsupported('Unresolvable parent component in soft link', path)
            if len(result) > self.limits.max_depth:
                raise CarryError('RESOURCE', 'Soft resolution exceeds max_depth', path)
            return deque(result)
        pending = components(path)
        prefix = '/'
        frames = []
        active = set()
        encountered = set()
        steps = 0
        while True:
            self.check()
            if not pending:
                if not frames:
                    return prefix, tuple(sorted(encountered))
                remaining, marker = frames.pop()
                self._resolved[marker] = prefix
                active.remove(marker)
                pending = remaining
                continue
            steps += 1
            if steps > self.limits.max_edges * (self.limits.max_depth + 1):
                raise CarryError('RESOURCE', 'Soft resolution work limit exceeded', path)
            part = pending.popleft()
            candidate = ('/' if prefix == '/' else prefix + '/') + part
            link = self.links.get(candidate)
            if link is None:
                profile.unsupported('Dangling internal soft link or missing selection', candidate)
            if link['kind'] == 'hard':
                prefix = candidate
            else:
                encountered.add(candidate)
                if candidate in active:
                    profile.unsupported('Internal soft-link cycle', candidate)
                if candidate in self._resolved:
                    prefix = self._resolved[candidate]
                else:
                    active.add(candidate)
                    frames.append((pending, candidate))
                    text = link['target']
                    target = text if text.startswith('/') else posixpath.dirname(candidate) + '/' + text
                    pending, prefix = components(target), '/'
            if pending and prefix != '/':
                actual = self.links.get(prefix)
                if actual is None or self.nodes[actual['target']]['kind'] != 'group':
                    profile.unsupported('Soft-link path traverses a non-group', candidate)

    def target(self, ref, path):
        if not ref:
            return None
        try:
            obj = self.file[ref]
        except (KeyError, ValueError, RuntimeError) as exc:
            profile.unsupported('Dangling object reference: ' + str(exc), path)
        if obj is None:
            profile.unsupported('Dangling object reference', path)
        uid = _identity(obj)
        if uid not in self.nodes:
            profile.unsupported('Reference to an anonymous or unsupported object', path)
        return uid

    def _relations(self):
        reverse = []
        for uid, node in self.nodes.items():
            self.check()
            obj, meta = node['object'], node['metadata']
            for attr in meta['attributes']:
                if attr['dtype']['kind'] == 'reference' and attr['shape'] is not None:
                    values = np.asarray(obj.attrs[attr['name']], dtype=object).reshape(-1)
                    for index, ref in enumerate(values):
                        self.edge()
                        self.references[uid].append((attr['name'], index, self.target(ref, obj.name + '@' + attr['name'])))
            if node['kind'] != 'dataset':
                continue
            if meta['dtype']['kind'] == 'reference':
                profile.admit_payload_read(obj)
                count = 0 if obj.shape is None else math.prod(obj.shape)
                if self.edges + count > self.limits.max_edges:
                    raise CarryError('RESOURCE', 'Reference dataset exceeds max_edges', obj.name)
                offset = 0
                for selection in profile.iter_blocks(obj.shape, obj.dtype.itemsize, self.limits.chunk_bytes):
                    for ref in np.asarray(obj[selection], dtype=object).reshape(-1):
                        self.edge()
                        self.references[uid].append((None, offset, self.target(ref, obj.name)))
                        offset += 1
            if 'DIMENSION_LIST' in obj.attrs:
                for axis, refs in enumerate(obj.attrs['DIMENSION_LIST']):
                    for ref in refs:
                        self.edge()
                        target = self.target(ref, obj.name + '@DIMENSION_LIST')
                        if target is None or self.nodes[target]['kind'] != 'dataset' or self.nodes[target]['metadata']['scale_name'] is None:
                            profile.unsupported('DIMENSION_LIST must target a dimension scale', obj.name)
                        self.scales.append((uid, axis, target))
            if 'REFERENCE_LIST' in obj.attrs:
                for row in obj.attrs['REFERENCE_LIST']:
                    self.edge()
                    target = self.target(row['dataset'], obj.name + '@REFERENCE_LIST')
                    axis = int(row['dimension'])
                    if target is None or self.nodes[target]['kind'] != 'dataset':
                        profile.unsupported('REFERENCE_LIST must target a dataset', obj.name)
                    rank = len(self.nodes[target]['object'].shape or ())
                    if axis >= rank:
                        profile.unsupported('REFERENCE_LIST dimension out of range', obj.name)
                    reverse.append((target, axis, uid))
        forward_counts, reverse_counts = Counter(self.scales), Counter(reverse)
        if forward_counts != reverse_counts or any(count != 1 for count in forward_counts.values()):
            profile.unsupported('Dimension-scale forward and reverse metadata disagree or repeat edges')


def make_plan_native(source, selections, limits):
    if not selections:
        raise CarryError('INVALID', 'At least one selection is required')
    selections = sorted({validate_path(path, limits) for path in selections})
    with h5py.File(source, 'r') as file:
        inv = _Inventory(file, limits)
        retained = set()
        included = set()
        expanded = set()
        dependencies = deque()
        soft_work = deque()
        reasons = defaultdict(set)

        def keep_node(uid, reason):
            reasons[uid].add(reason)
            if uid not in included:
                included.add(uid)
                dependencies.append(uid)

        def keep_container(path, reason):
            while path != '/':
                retained.add(path)
                uid = inv.links[path]['target']
                keep_node(uid, reason)
                path = posixpath.dirname(path)
            keep_node(inv.root, 'ancestor container')

        def keep_path(path, reason, full=True):
            pending = deque([(path, full)])
            while pending:
                current, descend = pending.popleft()
                inv.check()
                if current == '/':
                    uid = inv.root
                else:
                    link = inv.links[current]
                    if current not in retained:
                        retained.add(current)
                        keep_container(posixpath.dirname(current), 'ancestor container')
                        if link['kind'] == 'soft':
                            soft_work.append(current)
                    if link['kind'] == 'soft':
                        continue
                    uid = link['target']
                keep_node(uid, reason)
                if descend and inv.nodes[uid]['kind'] == 'group' and uid not in expanded:
                    expanded.add(uid)
                    pending.extend((child, True) for child in inv.children[current])

        # Seed all selected names before choosing fallback paths for dependencies.
        for selection in selections:
            terminal, chain = inv.resolve(selection)
            for soft in chain:
                keep_path(soft, 'selected soft-link path')
            keep_path(terminal, 'selected ' + selection)
        forward = defaultdict(list)
        for consumer, axis, scale in inv.scales:
            forward[consumer].append((axis, scale))
        while dependencies or soft_work:
            while soft_work:
                soft = soft_work.popleft()
                target, chain = inv.soft_targets[soft]
                for other in chain:
                    keep_path(other, 'soft-link dependency from ' + soft)
                keep_path(target, 'soft-link target from ' + soft)
            if dependencies:
                uid = dependencies.popleft()
                for attr, index, target in inv.references[uid]:
                    if target is not None:
                        if target not in included:
                            keep_path(inv.paths[target][0], 'reference from ' + inv.paths[uid][0])
                        elif inv.nodes[target]['kind'] == 'group' and target not in expanded:
                            keep_path(inv.paths[target][0], 'referenced group from ' + inv.paths[uid][0])
                for axis, target in forward[uid]:
                    if target not in included:
                        keep_path(inv.paths[target][0], 'scale for ' + inv.paths[uid][0] + ' axis ' + str(axis))
        canonical = {uid: min(path for path in inv.paths[uid] if path == '/' or path in retained) for uid in included}
        objects = []
        payload = 0
        for uid in sorted(included, key=canonical.__getitem__):
            inv.check()
            node = inv.nodes[uid]
            meta = node['metadata']
            if node['kind'] == 'dataset':
                obj = node['object']
                payload += (0 if obj.shape is None else math.prod(obj.shape)) * obj.dtype.itemsize
                if payload > limits.max_payload_bytes:
                    raise CarryError('RESOURCE', 'Selected logical payload exceeds max_payload_bytes', canonical[uid])
                if meta['dtype']['kind'] == 'fixed':
                    meta['payload_sha256'] = profile.payload_digest(obj, limits)
                    inv.check()
            objects.append({'id': canonical[uid], 'kind': node['kind'], 'metadata': meta})
        links = [{'path': path, 'kind': inv.links[path]['kind'],
                  'target': canonical[inv.links[path]['target']] if inv.links[path]['kind'] == 'hard' else inv.links[path]['target']}
                 for path in sorted(retained)]
        references = [{'owner': canonical[uid], 'attribute': attr, 'index': index,
                       'target': None if target is None else canonical[target]}
                      for uid in included for attr, index, target in inv.references[uid]]
        references.sort(key=lambda x: (x['owner'], x['attribute'] is not None, x['attribute'] or '', x['index']))
        scales = [{'consumer': canonical[consumer], 'axis': axis, 'scale': canonical[scale]}
                  for consumer, axis, scale in inv.scales if consumer in included]
        scales.sort(key=lambda x: (x['consumer'], x['axis'], x['scale']))
        graph = {'objects': objects, 'links': links, 'references': references, 'scales': scales,
                 'payload_bytes': payload, 'reasons': {canonical[uid]: sorted(reasons[uid]) for uid in sorted(included, key=canonical.__getitem__)}}
        if len(canonical_json(graph)) > limits.max_plan_bytes:
            raise CarryError('RESOURCE', 'Serialized graph exceeds max_plan_bytes')
        return graph
