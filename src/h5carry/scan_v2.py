"""Typed rectangular dependency decisions, isolated from version-one planning."""
from __future__ import annotations
from collections import Counter, defaultdict, deque
import copy
import hashlib
import math
import posixpath
import h5py
import numpy as np
from .model import CarryError, canonical_json
from .scan import _Inventory
from .slicing import transfers

WHOLE = {'kind':'whole'}


class _TypedInventory(_Inventory):
    def _relations(self):
        """Decode source relations with typed-mode guarded bounded reads."""
        reverse = []
        for uid, node in self.nodes.items():
            self.check()
            obj, meta = node['object'], node['metadata']
            for attr in meta['attributes']:
                if attr['dtype']['kind'] == 'reference' and attr['shape'] is not None:
                    for index, ref in enumerate(np.asarray(obj.attrs[attr['name']], dtype=object).reshape(-1)):
                        self.edge()
                        self.references[uid].append((attr['name'], index, self.target(ref, obj.name + '@' + attr['name'])))
            if node['kind'] != 'dataset':
                continue
            if meta['dtype']['kind'] == 'reference':
                count = 0 if obj.shape is None else math.prod(obj.shape)
                if self.edges + count > self.limits.max_edges:
                    raise CarryError('RESOURCE', 'Reference dataset exceeds max_edges', obj.name)
                offset = 0
                for selection, _ in transfers(obj, WHOLE, self.limits):
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
                            raise CarryError('UNSUPPORTED', 'DIMENSION_LIST must target a dimension scale', obj.name)
                        self.scales.append((uid, axis, target))
            if 'REFERENCE_LIST' in obj.attrs:
                for row in obj.attrs['REFERENCE_LIST']:
                    self.edge()
                    target = self.target(row['dataset'], obj.name + '@REFERENCE_LIST')
                    axis = int(row['dimension'])
                    if target is None or self.nodes[target]['kind'] != 'dataset' or axis >= len(self.nodes[target]['object'].shape or ()):
                        raise CarryError('UNSUPPORTED', 'Invalid reverse scale consumer or dimension', obj.name)
                    reverse.append((target, axis, uid))
        counts = Counter(self.scales)
        if counts != Counter(reverse) or any(n != 1 for n in counts.values()):
            raise CarryError('UNSUPPORTED', 'Dimension-scale forward and reverse metadata disagree or repeat edges')


def _normalize(node, selection, path):
    if selection['kind'] == 'whole':
        return dict(WHOLE)
    obj = node['object']
    if node['kind'] != 'dataset' or obj.shape is None or obj.ndim == 0 or node['metadata']['dtype']['kind'] != 'fixed':
        raise CarryError('UNSUPPORTED', 'Boxes require a simple nonscalar primitive dataset', path)
    start, stop = selection['start'], selection['stop']
    if len(start) != obj.ndim or len(stop) != obj.ndim or any(not 0 <= a <= b <= n for a,b,n in zip(start,stop,obj.shape)):
        raise CarryError('INVALID', 'Box rank or original-coordinate bounds do not match source', path)
    if all(a == 0 and b == n for a,b,n in zip(start,stop,obj.shape)):
        return dict(WHOLE)
    return copy.deepcopy(selection)


def make_plan_native(source, request, limits):
    from .selection import validate_selection
    request = validate_selection(request, limits)
    with h5py.File(source, 'r', rdcc_nbytes=0) as file:
        inv = _TypedInventory(file, limits)
        retained, included, expanded = set(), set(), set()
        choices, reasons = {}, defaultdict(set)
        pending, soft_work = deque(), deque()

        def keep_node(uid, reason, selection=None):
            reasons[uid].add(reason)
            if inv.nodes[uid]['kind'] == 'dataset':
                normalized = _normalize(inv.nodes[uid], selection or WHOLE, inv.paths[uid][0])
                if uid in choices and choices[uid] != normalized:
                    raise CarryError('INVALID', 'Selection conflict for source identity: ' + str(choices[uid]) + ' versus ' + str(normalized) + '; ' + reason, inv.paths[uid][0])
                choices[uid] = normalized
            else:
                choices[uid] = dict(WHOLE)
            if uid not in included:
                included.add(uid); pending.append(uid)

        def container(path):
            while path != '/':
                retained.add(path); keep_node(inv.links[path]['target'], 'ancestor container')
                path = posixpath.dirname(path)
            keep_node(inv.root, 'ancestor container')

        def keep_path(path, reason, selection=None, full=True):
            work = deque([(path, selection or WHOLE, full)])
            while work:
                current, selected, descend = work.popleft(); inv.check()
                if current == '/':
                    uid = inv.root
                else:
                    link = inv.links[current]
                    if current not in retained:
                        retained.add(current); container(posixpath.dirname(current))
                        if link['kind'] == 'soft': soft_work.append(current)
                    if link['kind'] == 'soft':
                        if selected['kind'] != 'whole': raise CarryError('INVALID', 'Box path must be a source hard path', current)
                        continue
                    uid = link['target']
                keep_node(uid, reason, selected)
                if descend and inv.nodes[uid]['kind'] == 'group' and uid not in expanded:
                    if selected['kind'] != 'whole': raise CarryError('INVALID', 'Groups cannot be box-selected', current)
                    expanded.add(uid)
                    work.extend((child, WHOLE, True) for child in inv.children[current])

        for entry in request['objects']:
            terminal, chain = inv.resolve(entry['path'])
            if entry['selection']['kind'] == 'box' and chain:
                raise CarryError('INVALID', 'Box path must be a source hard path', entry['path'])
            for soft in chain: keep_path(soft, 'selected soft-link path')
            keep_path(terminal, 'selected ' + entry['path'], entry['selection'])
        mappings = {}
        for item in request['scale_mappings']:
            consumer, cchain = inv.resolve(item['consumer']); scale, schain = inv.resolve(item['scale'])
            if cchain or schain:
                raise CarryError('INVALID', 'Index mappings require source hard paths', item['consumer'])
            cuid = inv.root if consumer == '/' else inv.links[consumer]['target']
            suid = inv.root if scale == '/' else inv.links[scale]['target']
            key = (cuid,item['axis'],suid)
            if key in mappings:
                raise CarryError('INVALID', 'Duplicate index mapping through aliases', item['consumer'])
            if key not in inv.scales:
                raise CarryError('INVALID', 'Index mapping does not name a source attachment', item['consumer'])
            cobj, sobj = inv.nodes[cuid]['object'], inv.nodes[suid]['object']
            if sobj.shape is None or sobj.ndim != 1 or sobj.shape[0] != cobj.shape[item['axis']]:
                raise CarryError('INVALID', 'Index assertion requires rank-one scale matching original axis', item['scale'])
            mappings[key] = item
        used = set()
        forward = defaultdict(list)
        for consumer, axis, scale in inv.scales: forward[consumer].append((axis,scale))
        while pending or soft_work:
            while soft_work:
                soft = soft_work.popleft(); target, chain = inv.soft_targets[soft]
                for other in chain: keep_path(other, 'soft-link dependency from ' + soft)
                keep_path(target, 'soft-link target from ' + soft)
            if not pending: continue
            uid = pending.popleft()
            for attr, index, target in inv.references[uid]:
                if target is not None:
                    names = [p for p in inv.paths[target] if p == '/' or p in retained]
                    keep_path(names[0] if names else inv.paths[target][0], 'whole reference from ' + inv.paths[uid][0])
            for axis, target in forward[uid]:
                selected = WHOLE
                key = (uid,axis,target)
                if choices[uid]['kind'] == 'box':
                    if key not in mappings:
                        raise CarryError('INVALID', 'Missing explicit index mapping for axis ' + str(axis) + ' scale ' + inv.paths[target][0], inv.paths[uid][0])
                    used.add(key)
                    consumer, scale = inv.nodes[uid]['object'], inv.nodes[target]['object']
                    if scale.shape is None or scale.ndim != 1 or scale.shape[0] != consumer.shape[axis]:
                        raise CarryError('INVALID', 'Index mapping requires rank-one scale matching original axis ' + str(axis), inv.paths[target][0])
                    selected = {'kind':'box','start':[choices[uid]['start'][axis]],'stop':[choices[uid]['stop'][axis]]}
                elif key in mappings:
                    used.add(key)
                names = [p for p in inv.paths[target] if p in retained]
                keep_path(names[0] if names else inv.paths[target][0], 'scale for ' + inv.paths[uid][0] + ' axis ' + str(axis), selected)
        if set(mappings) != used:
            unused = mappings[next(iter(set(mappings)-used))]
            raise CarryError('INVALID', 'Extraneous index mapping for excluded consumer', unused['consumer'])
        canonical = {uid:min(p for p in inv.paths[uid] if p == '/' or p in retained) for uid in included}
        objects, originals, selections, transformations = [], [], [], []
        payload = 0
        for uid in sorted(included, key=canonical.__getitem__):
            node=inv.nodes[uid]; path=canonical[uid]; meta=copy.deepcopy(node['metadata']); obj=node['object']; selected=choices[uid]
            originals.append({'id':path,'kind':node['kind'],'metadata':copy.deepcopy(meta)})
            selections.append({'id':path,'selection':copy.deepcopy(selected)})
            if node['kind']=='dataset':
                if selected['kind']=='box':
                    from .native_properties import chunk_options
                    if obj.chunks and chunk_options(obj.id.get_create_plist()) != 0:
                        raise CarryError('UNSUPPORTED','Proper crops require default native chunk options',path)
                    shape=[b-a for a,b in zip(selected['start'],selected['stop'])]
                    old=copy.deepcopy(meta['creation'])
                    meta['shape']=shape; meta['creation']['maxshape']=shape[:]
                    if obj.chunks: meta['creation']['chunks']=[min(c,max(1,n)) for c,n in zip(obj.chunks,shape)]
                    transformations.append({'id':path,'source_shape':list(obj.shape),'output_shape':shape[:],
                        'source_maxshape':old['maxshape'],'output_maxshape':shape[:],
                        'source_chunks':old['chunks'],'output_chunks':meta['creation']['chunks']})
                payload += (0 if meta['shape'] is None else math.prod(meta['shape']))*obj.dtype.itemsize
                if payload > limits.max_payload_bytes:
                    raise CarryError('RESOURCE','Selected logical payload exceeds max_payload_bytes',path)
                if meta['shape'] is not None and math.prod(meta['shape']) != 0:
                    from .slicing import guard_read
                    guard_read(obj,limits)
            objects.append({'id':path,'kind':node['kind'],'metadata':meta})
        # Resolve all identity/selection/storage/resource decisions before hashing.
        for record in objects:
            if record['kind']=='dataset' and record['metadata']['dtype']['kind']=='fixed':
                obj=file[record['id']]; uid=inv.links[record['id']]['target']
                digest=hashlib.sha256()
                for selection,_ in transfers(obj,choices[uid],limits):
                    digest.update(np.asarray(obj[selection],dtype=obj.dtype).tobytes(order='C'));inv.check()
                record['metadata']['payload_sha256']=digest.hexdigest()
        links=[{'path':p,'kind':inv.links[p]['kind'],'target':canonical[inv.links[p]['target']] if inv.links[p]['kind']=='hard' else inv.links[p]['target']} for p in sorted(retained)]
        references=[{'owner':canonical[uid],'attribute':attr,'index':index,'target':None if target is None else canonical[target]} for uid in included for attr,index,target in inv.references[uid]]
        references.sort(key=lambda x:(x['owner'],x['attribute'] is not None,x['attribute'] or '',x['index']))
        scales=[{'consumer':canonical[c],'axis':a,'scale':canonical[s]} for c,a,s in inv.scales if c in included]
        scales.sort(key=lambda x:(x['consumer'],x['axis'],x['scale']))
        graph={'objects':objects,'links':links,'references':references,'scales':scales,'payload_bytes':payload,
               'reasons':{canonical[uid]:sorted(reasons[uid]) for uid in sorted(included,key=canonical.__getitem__)}}
        result={'graph':graph,'source_objects':originals,'selections':selections,'transformations':transformations}
        if len(canonical_json(result))>limits.max_plan_bytes:
            raise CarryError('RESOURCE','Serialized typed graph exceeds max_plan_bytes')
        return result
