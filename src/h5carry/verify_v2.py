"""Independently derive and verify rectangular snapshots from source + request.

Only the original verifier's namespace/profile machinery is reused. Request
validation, identity constraints, fixed-point closure, storage transformations
and constant-rank-space coordinate iteration are implemented here, without the
v2 planner, writer or their selection helpers. Run inside a bounded native child.
"""
from __future__ import annotations

from collections import Counter, deque
import copy
import hashlib
import math

import h5py
import numpy as np

from .model import CarryError, Limits, canonical_json, fingerprint, validate_path
from .profile import attribute_descriptor, dataset_creation, dtype_descriptor, reserved_metadata
from .verify import _Inventory as _BaseInventory
from .verify import _RESERVED, _coverage, _graph_differences, _problem, _report


def _request(value, limits):
    """A separate strict validator, including canonical order of original intent."""
    def keys(item, expected, label):
        if type(item) is not dict or set(item) != set(expected):
            _problem("INVALID", "invalid " + label + " fields")
    keys(value, ("format", "version", "objects", "scale_mappings", "storage_policy"), "selection request")
    if value["format"] != "h5carry-selection" or type(value["version"]) is not int or value["version"] != 1:
        _problem("INVALID", "unsupported selection request format/version")
    if value["storage_policy"] != "fixed-snapshot-v1":
        _problem("INVALID", "unsupported snapshot storage policy")
    if len(canonical_json(value)) > limits.max_plan_bytes:
        _problem("RESOURCE", "selection request exceeds serialized budget")
    objects, mappings = value["objects"], value["scale_mappings"]
    if type(objects) is not list or not objects or type(mappings) is not list:
        _problem("INVALID", "selection requires nonempty objects and a mapping list")
    if len(objects) > limits.max_objects or len(mappings) > limits.max_edges:
        _problem("RESOURCE", "selection request exceeds object/edge budget")
    paths, edges = set(), set()
    for item in objects:
        keys(item, ("path", "selection"), "selected object")
        path = validate_path(item["path"], limits)
        if path in paths:
            _problem("INVALID", "duplicate selected path", path)
        paths.add(path)
        selection = item["selection"]
        if type(selection) is not dict:
            _problem("INVALID", "selection must be an object", path)
        if selection.get("kind") == "whole":
            keys(selection, ("kind",), "whole selection")
        elif selection.get("kind") == "box":
            keys(selection, ("kind", "start", "stop"), "box selection")
            start, stop = selection["start"], selection["stop"]
            if type(start) is not list or type(stop) is not list or not start or len(start) != len(stop):
                _problem("INVALID", "box needs equal nonempty rank vectors", path)
            if len(start) > limits.max_rank:
                _problem("RESOURCE", "selection rank exceeds limit", path)
            for a, b in zip(start, stop):
                if type(a) is not int or type(b) is not int or not 0 <= a <= b <= (1 << 63) - 1:
                    _problem("INVALID", "box endpoints must be ordered nonnegative signed-63-bit integers", path)
        else:
            _problem("INVALID", "unknown selection kind", path)
    for item in mappings:
        keys(item, ("consumer", "axis", "scale", "mapping"), "scale mapping")
        validate_path(item["consumer"], limits)
        validate_path(item["scale"], limits)
        if type(item["axis"]) is not int or not 0 <= item["axis"] < limits.max_rank or item["mapping"] != "index":
            _problem("INVALID", "invalid positional scale mapping")
        edge = item["consumer"], item["axis"], item["scale"]
        if edge in edges:
            _problem("INVALID", "duplicate scale mapping", item["consumer"])
        edges.add(edge)
    result = copy.deepcopy(value)
    result["objects"].sort(key=lambda item: item["path"])
    result["scale_mappings"].sort(key=lambda item: (item["consumer"], item["axis"], item["scale"]))
    if canonical_json(result) != canonical_json(value):
        _problem("INVALID", "original selection request is not canonical")
    return result


def _tiles(shape, itemsize, cap, origin=None, source_chunks=None, output_chunks=None):
    """C-order row segments; O(rank) state even for enormous prefix extents.

    Yield matching source/output coordinates. Coalesce within one row only,
    bounded by logical bytes and each dataset's aggregate raw-chunk exposure.
    """
    if shape is None or any(n == 0 for n in shape):
        return
    if itemsize > cap:
        _problem("RESOURCE", "one logical element exceeds comparison budget")
    if not shape:
        yield (), ()
        return
    origin = (0,) * len(shape) if origin is None else tuple(origin)
    prefix = [0] * (len(shape) - 1)
    while True:
        column = 0
        while column < shape[-1]:
            width = min(cap // itemsize, shape[-1] - column)
            for chunks, absolute_column in ((source_chunks, origin[-1] + column), (output_chunks, column)):
                if chunks is not None:
                    chunk_allowance = cap // (math.prod(chunks) * itemsize)
                    if chunk_allowance < 1:
                        _problem("RESOURCE", "one raw chunk exceeds comparison budget")
                    width = min(width, chunk_allowance * chunks[-1] - absolute_column % chunks[-1])
            source = tuple(a + b for a, b in zip(origin[:-1], prefix)) + (slice(origin[-1] + column, origin[-1] + column + width),)
            output = tuple(prefix) + (slice(column, column + width),)
            yield source, output
            column += width
        axis = len(prefix) - 1
        while axis >= 0:
            prefix[axis] += 1
            if prefix[axis] < shape[axis]:
                break
            prefix[axis] = 0
            axis -= 1
        if axis < 0:
            return


class _Inventory(_BaseInventory):
    def read(self, dataset, selection):
        """Prove logical and raw-chunk budgets BEFORE every dataset payload read."""
        self.tick()
        shape = dataset.shape
        if shape is None:
            _problem("INVALID", "null dataspace has no payload", dataset.name)
        count, touched = 1, 1
        for axis, index in enumerate(selection):
            if isinstance(index, slice):
                start, stop, step = index.indices(shape[axis])
                if step != 1:
                    _problem("INVALID", "non-unit verifier tile", dataset.name)
                count *= max(0, stop - start)
                if dataset.chunks is not None:
                    touched *= 0 if stop <= start else (stop - 1) // dataset.chunks[axis] - start // dataset.chunks[axis] + 1
        size = count * dataset.dtype.itemsize
        if size > self.limits.chunk_bytes:
            _problem("RESOURCE", "native read exceeds logical comparison budget", dataset.name)
        if count:
            dcpl = dataset.id.get_create_plist()
            if dcpl.get_fill_time() == h5py.h5d.FILL_TIME_NEVER:
                _problem("UNSUPPORTED", "nonempty FILL_TIME_NEVER payload reads are unqualified", dataset.name)
            exposure = 0 if dataset.chunks is None else math.prod(dataset.chunks) * dataset.dtype.itemsize * touched
            if exposure > self.limits.chunk_bytes:
                _problem("RESOURCE", "native uncompressed chunk exposure exceeds comparison budget", dataset.name)
            self.coverage["max_native_expansion_bytes"] = max(self.coverage.get("max_native_expansion_bytes", 0), exposure)
        data = np.asarray(dataset[selection], dtype=dataset.dtype)
        if data.size * dataset.dtype.itemsize != size:
            _problem("MISMATCH", "native read shape differs from independently bounded tile", dataset.name)
        self.coverage["max_read_bytes"] = max(self.coverage["max_read_bytes"], size)
        self.coverage["data_blocks"] += 1
        return data

    def _describe_and_decode(self):
        for oid in sorted(self.objects, key=lambda o: self.aliases[o][0]):
            self.tick()
            obj, path = self.objects[oid], self.aliases[oid][0]
            self.tick(len(obj.attrs))
            attr_names = list(obj.attrs)
            if any(not isinstance(name, str) for name in attr_names):
                _problem("UNSUPPORTED", "attribute names must be valid UTF-8 text", path)
            attrs = []
            refs = []
            if isinstance(obj, h5py.Dataset):
                if obj.ndim > self.limits.max_rank:
                    _problem("RESOURCE", "dataset rank exceeds limit", path)
                try:
                    dt = dtype_descriptor(obj.id.get_type(), obj.dtype, attribute=False)
                except (TypeError, ValueError) as exc:
                    raise CarryError("UNSUPPORTED", "datatype has no qualified NumPy representation", path) from exc
                except CarryError as exc:
                    if exc.path is None:
                        exc.path = path
                    raise
                if obj.dtype.itemsize > self.limits.chunk_bytes:
                    _problem("RESOURCE", "one logical element exceeds the comparison chunk budget", path)
                creation = dataset_creation(obj)
                special = reserved_metadata(obj, self.limits)
                meta = {"attributes": attrs, "dtype": dt,
                        "shape": None if obj.shape is None else list(obj.shape),
                        "creation": creation, "scale_name": special["scale_name"],
                        "labels": special["labels"], "payload_sha256": None}
            else:
                if any(n in _RESERVED for n in obj.attrs):
                    _problem("UNSUPPORTED", "reserved dimension-scale attribute on group", path)
                meta = {"attributes": attrs}
            for name in sorted(attr_names):
                if isinstance(obj, h5py.Dataset) and name in _RESERVED:
                    continue
                descriptor = attribute_descriptor(obj, name, self.limits)
                attrs.append(descriptor)
                if descriptor["dtype"]["kind"] == "reference" and descriptor["shape"] is not None:
                    value = np.asarray(obj.attrs[name]).reshape(-1)
                    for index, ref in enumerate(value):
                        self.tick(1)
                        refs.append((name, index, self._target(ref, path)))
            if isinstance(obj, h5py.Dataset):
                if meta["dtype"]["kind"] == "reference":
                    count = 0 if obj.shape is None else math.prod(obj.shape)
                    if count > self.limits.max_edges - self.edge_count:
                        _problem("RESOURCE", "reference inventory exceeds remaining edge budget", path)
                    offset = 0
                    for selection, _ in _tiles(obj.shape, obj.dtype.itemsize, self.limits.chunk_bytes, source_chunks=obj.chunks):
                        for ref in self.read(obj, selection).reshape(-1):
                            self.tick(1)
                            refs.append((None, offset, self._target(ref, path)))
                            offset += 1
                self._scales(oid, obj, path)
            self.metadata[oid] = meta
            self.references[oid] = refs
        if Counter(self.forward) != Counter(self.reverse):
            _problem("UNSUPPORTED", "forward and reverse dimension-scale relations disagree")
        if any(count != 1 for count in Counter(self.forward).values()):
            _problem("UNSUPPORTED", "duplicate dimension-scale relation")
        for consumer, axis, scale in self.forward:
            ds = self.objects[scale]
            if not isinstance(ds, h5py.Dataset) or self.metadata[scale]["scale_name"] is None:
                _problem("UNSUPPORTED", "dimension attachment target is not a marked scale", self.aliases[scale][0])

    def normalize(self, oid, selection, path):
        if selection["kind"] == "whole":
            return {"kind": "whole"}
        obj = self.objects[oid]
        if not isinstance(obj, h5py.Dataset) or self.metadata[oid]["dtype"]["kind"] != "fixed" or not obj.shape:
            _problem("UNSUPPORTED", "boxes require fixed-width nonscalar simple datasets", path)
        start, stop = selection["start"], selection["stop"]
        if len(start) != len(obj.shape) or any(not 0 <= a <= b <= n for a, b, n in zip(start, stop, obj.shape)):
            _problem("INVALID", "box bounds differ from source rank or extent", path)
        if all(a == 0 and b == n for a, b, n in zip(start, stop, obj.shape)):
            return {"kind": "whole"}
        return copy.deepcopy(selection)

    def required_v2(self, request):
        hard, soft, included, expanded = {"/"}, set(), {self.names["/"]}, set()
        selections = {self.names["/"]: {"kind": "whole"}}
        origins = {self.names["/"]: "/"}
        pending = deque([self.names["/"]])
        whole = {"kind": "whole"}

        def require(oid, selection, path, context=None):
            selected = self.normalize(oid, selection, path)
            if oid in selections and selections[oid] != selected:
                _problem("INVALID", f"incompatible selections from {origins[oid]} {selections[oid]} and {context or path} {selected}", path)
            if oid not in included:
                included.add(oid)
                pending.append(oid)
            selections[oid] = selected
            origins.setdefault(oid, context or path)

        def entry(path, selection, deep=True, context=None):
            work = deque([(path, selection, deep)])
            while work:
                named, selected, descend = work.popleft()
                self.tick()
                terminal, through = self.resolve(named)
                if selected["kind"] == "box" and through:
                    _problem("UNSUPPORTED", "box path must be a proven hard-linked dataset", named)
                for softpath in through:
                    if softpath not in soft:
                        soft.add(softpath)
                        work.append((softpath, whole, True))
                    parent = softpath.rsplit("/", 1)[0] or "/"
                    work.append((parent, whole, False))
                oid = self.names[terminal]
                hard.add(terminal)
                require(oid, selected, named, context)
                parent = terminal.rsplit("/", 1)[0] or "/"
                while terminal != "/":
                    hard.add(parent)
                    require(self.names[parent], whole, parent)
                    terminal, parent = parent, parent.rsplit("/", 1)[0] or "/"
                if descend and isinstance(self.objects[oid], h5py.Group) and oid not in expanded:
                    expanded.add(oid)
                    work.extend((child, whole, True) for child in self.children[oid])

        for item in request["objects"]:
            entry(item["path"], item["selection"])
        assertions = {}
        for item in request["scale_mappings"]:
            if item["consumer"] not in self.names or item["scale"] not in self.names:
                _problem("UNSUPPORTED", "mapping paths must be proven source hard links", item["consumer"])
            consumer, _ = self.resolve(item["consumer"])
            scale, _ = self.resolve(item["scale"])
            edge = self.names[consumer], item["axis"], self.names[scale]
            if edge in assertions:
                _problem("INVALID", "duplicate mapping aliases identify the same attachment", consumer)
            assertions[edge] = item
        used = set()
        forward = {}
        for consumer, axis, scale in self.forward:
            forward.setdefault(consumer, []).append((axis, scale))
        done = set()
        while pending:
            oid = pending.popleft()
            if oid in done:
                continue
            done.add(oid)
            self.tick()
            targets = {target for _, _, target in self.references[oid] if target is not None}
            for target in sorted(targets, key=lambda item: self.aliases[item][0]):
                names = [p for p in self.aliases[target] if p in hard]
                entry(names[0] if names else self.aliases[target][0], whole)
            for axis, scale in forward.get(oid, ()):
                selected = selections[oid]
                edge = oid, axis, scale
                if selected["kind"] == "box" and edge not in assertions:
                    _problem("INVALID", f"missing explicit positional mapping for axis {axis} to {self.aliases[scale][0]}", self.aliases[oid][0])
                if edge in assertions:
                    consumer_obj, scale_obj = self.objects[oid], self.objects[scale]
                    if scale_obj.shape is None or scale_obj.ndim != 1 or scale_obj.shape[0] != consumer_obj.shape[axis]:
                        _problem("INVALID", f"positional scale rank/length disagrees on axis {axis}", self.aliases[oid][0])
                    used.add(edge)
                scale_selection = whole if selected["kind"] == "whole" else {
                    "kind": "box", "start": [selected["start"][axis]], "stop": [selected["stop"][axis]]}
                names = [p for p in self.aliases[scale] if p in hard]
                entry(names[0] if names else self.aliases[scale][0], scale_selection,
                      context=f"{self.aliases[oid][0]} axis {axis} scale {self.aliases[scale][0]}")
        if used != set(assertions):
            unused = next(edge for edge in assertions if edge not in used)
            _problem("INVALID", "extraneous mapping does not identify a retained attachment", assertions[unused]["consumer"])
        return self.project_v2(hard, soft, selections)

    def project_v2(self, hard, soft, selections=None):
        canonical = {}
        for path in sorted(hard):
            canonical.setdefault(self.names[path], path)
        selections = selections or {oid: {"kind": "whole"} for oid in canonical}
        objects, source_objects, choices, transformations = [], [], [], []
        payload = 0
        for oid, path in sorted(canonical.items(), key=lambda item: item[1]):
            self.tick()
            obj = self.objects[oid]
            kind = "dataset" if isinstance(obj, h5py.Dataset) else "group"
            original = copy.deepcopy(self.metadata[oid])
            meta = copy.deepcopy(original)
            selected = selections[oid]
            source_objects.append({"id": path, "kind": kind, "metadata": original})
            choices.append({"id": path, "selection": copy.deepcopy(selected)})
            if kind == "dataset":
                if selected["kind"] == "box":
                    # H5Pset_chunk can silently clear flags that H5Pequal ignores.
                    # The primitive accessor reads only this native property.
                    if obj.chunks is not None:
                        from .native_properties import chunk_options
                        if chunk_options(obj.id.get_create_plist()):
                            _problem("UNSUPPORTED", "cropping nondefault native chunk options is unqualified", path)
                    shape = [b - a for a, b in zip(selected["start"], selected["stop"])]
                    creation = meta["creation"]
                    chunks = creation["chunks"]
                    new_chunks = None if chunks is None else [min(c, max(1, n)) for c, n in zip(chunks, shape)]
                    transformations.append({"id": path, "source_shape": original["shape"], "output_shape": shape,
                        "source_maxshape": creation["maxshape"], "output_maxshape": shape,
                        "source_chunks": chunks, "output_chunks": new_chunks})
                    meta["shape"] = shape
                    creation["maxshape"], creation["chunks"] = shape, new_chunks
                payload += (0 if meta["shape"] is None else math.prod(meta["shape"])) * obj.dtype.itemsize
                if payload > self.limits.max_payload_bytes:
                    _problem("RESOURCE", "selected logical payload exceeds budget", path)
            objects.append({"id": path, "kind": kind, "metadata": meta})
        # Account the whole selection before hashing any fixed-width payload.
        for record, choice in zip(objects, choices):
            if record["kind"] != "dataset" or record["metadata"]["dtype"]["kind"] != "fixed":
                continue
            obj = self.objects[self.names[record["id"]]]
            origin = choice["selection"].get("start")
            digest = hashlib.sha256()
            for index, _ in _tiles(record["metadata"]["shape"], obj.dtype.itemsize, self.limits.chunk_bytes,
                                   origin=origin, source_chunks=obj.chunks):
                digest.update(self.read(obj, index).tobytes(order="C"))
            record["metadata"]["payload_sha256"] = digest.hexdigest()
        links = [{"path": p, "kind": "hard", "target": canonical[self.names[p]]} for p in sorted(hard) if p != "/"]
        links.extend({"path": p, "kind": "soft", "target": self.soft[p]} for p in sorted(soft))
        references, scales = [], []
        for oid, owner in canonical.items():
            for attribute, index, target in self.references[oid]:
                if target is not None and target not in canonical:
                    _problem("MISMATCH", "retained reference target absent from closure", owner)
                references.append({"owner": owner, "attribute": attribute, "index": index,
                                   "target": None if target is None else canonical[target]})
        for consumer, axis, scale in self.forward:
            if consumer in canonical:
                if scale not in canonical:
                    _problem("MISMATCH", "retained scale absent from closure", canonical[consumer])
                scales.append({"consumer": canonical[consumer], "axis": axis, "scale": canonical[scale]})
        graph = {"objects": objects, "links": sorted(links, key=lambda item: item["path"]),
                 "references": sorted(references, key=canonical_json), "scales": sorted(scales, key=canonical_json), "payload_bytes": payload}
        result = {"graph": graph, "source_objects": source_objects, "selections": choices, "transformations": transformations}
        if len(canonical_json(result)) > self.limits.max_plan_bytes:
            _problem("RESOURCE", "independently derived snapshot exceeds serialized budget")
        return result


def _creation_equal(left, right, selection):
    expected = left.id.get_create_plist().copy()
    actual = right.id.get_create_plist()
    if left.chunks is not None:
        from .native_properties import chunk_options
        if chunk_options(expected) != chunk_options(actual):
            return False
        if selection["kind"] == "box":
            expected.set_chunk(tuple(min(c, max(1, n)) for c, n in zip(left.chunks, right.shape)))
    return bool(expected.equal(actual))


def verify_export(source, output, plan, limits):
    """Require source-derived plan agreement and exact reopened-output equality."""
    coverage = _coverage()
    coverage.update(max_native_expansion_bytes=0, positional_mapping_verified=False, scientific_validity_verified=False)
    try:
        if type(plan) is not dict or set(plan) != {"format", "version", "source", "request", "limits", "graph", "source_objects", "selections", "transformations"}:
            _problem("INVALID", "invalid version-2 plan envelope")
        if plan["format"] != "h5carry-plan" or type(plan["version"]) is not int or plan["version"] != 2:
            _problem("INVALID", "unsupported plan format/version")
        if len(canonical_json(plan)) > limits.max_plan_bytes:
            _problem("RESOURCE", "plan exceeds serialized comparison budget")
        Limits.from_dict(plan["limits"])
        request = _request(plan["request"], limits)
        if fingerprint(source) != plan["source"]:
            _problem("SOURCE_CHANGED", "source does not match plan fingerprint")
        with h5py.File(source, "r", rdcc_nbytes=0) as src:
            original = _Inventory(src, limits, coverage)
            expected = original.required_v2(request)
            coverage["closure_independently_derived"] = True
            diagnostics = _graph_differences(expected["graph"], plan["graph"], "plan")
            for field in ("source_objects", "selections", "transformations"):
                if canonical_json(expected[field]) != canonical_json(plan[field]):
                    diagnostics.append({"code": "MISMATCH", "message": f"plan {field} differ from independently derived source request"})
            if not diagnostics:
                with h5py.File(output, "r", rdcc_nbytes=0) as dst:
                    delivered = _Inventory(dst, limits, coverage)
                    actual = delivered.project_v2(set(delivered.names), set(delivered.soft))["graph"]
                    diagnostics.extend(_graph_differences(expected["graph"], actual, "output"))
                    if not diagnostics:
                        for record, choice in zip(expected["graph"]["objects"], expected["selections"]):
                            if record["kind"] != "dataset":
                                continue
                            path = record["id"]
                            left, right = src[path], dst[path]
                            if not left.id.get_type().equal(right.id.get_type()):
                                diagnostics.append({"code": "MISMATCH", "message": "exact low-level datatype differs", "path": path})
                            if not _creation_equal(left, right, choice["selection"]):
                                diagnostics.append({"code": "MISMATCH", "message": "exact low-level creation properties differ", "path": path})
                            if record["metadata"]["dtype"]["kind"] != "fixed":
                                continue
                            for source_index, output_index in _tiles(right.shape, left.dtype.itemsize, limits.chunk_bytes,
                                    origin=choice["selection"].get("start"), source_chunks=left.chunks, output_chunks=right.chunks):
                                a = original.read(left, source_index).tobytes(order="C")
                                b = delivered.read(right, output_index).tobytes(order="C")
                                if a != b:
                                    diagnostics.append({"code": "MISMATCH", "message": "translated source/output logical bytes differ", "path": path})
                                    break
            for field in ("objects", "links", "references", "scales"):
                coverage[field] = len(expected["graph"][field])
            coverage["payload_bytes"] = expected["graph"]["payload_bytes"]
        if fingerprint(source) != plan["source"]:
            _problem("SOURCE_CHANGED", "source changed during verification")
        coverage["source_equality"] = not diagnostics
        return _report("mismatch" if diagnostics else "verified", diagnostics, coverage, limits)
    except CarryError as exc:
        return _report("mismatch" if exc.code == "MISMATCH" else "incomplete", [exc.to_dict()], coverage, limits)
    except OSError as exc:
        return _report("incomplete", [{"code": "ERROR", "message": f"cannot complete native file read: {exc}"}], coverage, limits)
