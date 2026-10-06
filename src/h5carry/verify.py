"""Independent source-closure and reopened-output verification.

This module deliberately does not import the planner, scanner or writer. Its
namespace walk, soft-link resolver, dependency fixed point, reference decoding,
scale relation checks and data tiling are separate implementations. Only primitive
HDF5 encoding descriptors and the public JSON/limit model are shared. Native
operations in this module must be invoked in the bounded worker process.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import itertools
import math
import platform
import time

import h5py
import numpy as np

from .model import CarryError, Limits, canonical_json, fingerprint, validate_path
from .native_properties import chunk_options
from .profile import (attribute_descriptor, dataset_creation, dtype_descriptor,
                      reserved_metadata)

_RESERVED = {"CLASS", "NAME", "DIMENSION_LIST", "REFERENCE_LIST", "DIMENSION_LABELS"}


def _identity(obj):
    return int(h5py.h5o.get_info(obj.id).addr)


def _runtime():
    return {"python": platform.python_version(), "h5py": h5py.__version__,
            "hdf5": h5py.version.hdf5_version, "numpy": np.__version__}


def _problem(code, message, path=None):
    raise CarryError(code, message, path)


def _blocks(shape, itemsize, cap):
    """Row-major slabs with no read larger than cap, including wide rows."""
    if shape is None or any(n == 0 for n in shape):
        return
    if itemsize > cap:
        _problem("RESOURCE", "one logical element exceeds the comparison chunk budget")
    if not shape:
        yield ()
        return
    width = max(1, cap // itemsize)
    # Prefixes are integer indexes; only the final axis needs a bounded slice.
    for prefix in itertools.product(*(range(n) for n in shape[:-1])):
        for start in range(0, shape[-1], width):
            yield prefix + (slice(start, min(shape[-1], start + width)),)


class _Inventory:
    """A full named namespace. No inclusion decisions are borrowed from a plan."""
    def __init__(self, file, limits, coverage):
        self.file, self.limits, self.coverage = file, limits, coverage
        self.started = time.monotonic()
        self.objects = {}           # local object address -> open object
        self.names = {}             # exact hard path -> local address
        self.aliases = {}           # local address -> sorted hard paths
        self.soft = {}              # exact soft path -> unchanged stored text
        self.children = {}          # group address -> immediate link paths
        self.metadata = {}
        self.references = {}        # owner address -> (attribute,index,target)
        self.forward = []           # (consumer address,axis,scale address)
        self.reverse = []
        self.edge_count = 0
        self._walk()
        self._describe_and_decode()

    def tick(self, edges=0):
        self.edge_count += edges
        if self.edge_count > self.limits.max_edges:
            _problem("RESOURCE", "verification edge budget exceeded")
        if time.monotonic() - self.started > self.limits.discovery_seconds:
            _problem("RESOURCE", "verification discovery deadline exceeded")

    def _walk(self):
        queue = deque([(self.file["/"], "/")])
        root = queue[0][0]
        rid = _identity(root)
        self.objects[rid], self.names["/"], self.aliases[rid] = root, rid, ["/"]
        while queue:
            group, path = queue.popleft()
            self.tick()
            gid = _identity(group)
            self.children[gid] = []
            if len(group) > self.limits.max_edges - self.edge_count:
                _problem("RESOURCE", "group link count exceeds remaining edge budget", path)
            link_names = list(group.keys())
            if any(not isinstance(name, str) for name in link_names):
                _problem("UNSUPPORTED", "link names must be valid UTF-8 text", path)
            for name in sorted(link_names):
                self.tick(1)
                child = ("" if path == "/" else path) + "/" + name
                validate_path(child, self.limits)
                self.children[gid].append(child)
                link = group.get(name, getlink=True)
                if isinstance(link, h5py.ExternalLink):
                    _problem("UNSUPPORTED", "external link is forbidden; target was not opened", child)
                if isinstance(link, h5py.SoftLink):
                    if not isinstance(link.path, str) or "\x00" in link.path:
                        _problem("UNSUPPORTED", "invalid soft-link text", child)
                    if len(link.path.encode("utf-8")) > self.limits.max_name_bytes:
                        _problem("RESOURCE", "soft-link text exceeds name budget", child)
                    self.soft[child] = link.path
                    continue
                if not isinstance(link, h5py.HardLink):
                    _problem("UNSUPPORTED", "unknown link class", child)
                obj = group[name]  # only a proven hard link is opened
                if not isinstance(obj, (h5py.Group, h5py.Dataset)):
                    _problem("UNSUPPORTED", "named datatype or unknown object kind", child)
                oid = _identity(obj)
                if oid in self.objects and isinstance(obj, h5py.Group):
                    _problem("UNSUPPORTED", "group hard alias or cycle", child)
                self.names[child] = oid
                self.aliases.setdefault(oid, []).append(child)
                if oid not in self.objects:
                    self.objects[oid] = obj
                    if len(self.objects) > self.limits.max_objects:
                        _problem("RESOURCE", "verification object budget exceeded", child)
                    if isinstance(obj, h5py.Group):
                        queue.append((obj, child))
        for names in self.aliases.values():
            names.sort()
        # First reject external links across the COMPLETE hard namespace, then
        # resolve soft text ourselves. Only after that is a native usability
        # check safe: every possible named target is in this same file.
        for path in self.soft:
            terminal, _ = self.resolve(path)
            try:
                resolved = self.file[path]
            except (KeyError, RuntimeError, ValueError) as exc:
                raise CarryError("UNSUPPORTED", "soft link is not usable with native default access", path) from exc
            if _identity(resolved) != self.names[terminal]:
                _problem("UNSUPPORTED", "native and lexical soft-link targets disagree", path)

    def resolve(self, path):
        """Return terminal hard path and every soft entry traversed."""
        used = []
        active = set()
        pending = deque(p for p in path.split("/") if p and p != ".")
        current = "/"
        steps = 0
        while pending:
            self.tick()
            token = pending.popleft()
            if isinstance(token, tuple):
                active.remove(token[0])
                continue
            steps += 1
            if steps > self.limits.max_edges * (self.limits.max_depth + 1):
                _problem("RESOURCE", "soft-link resolution work budget exceeded", path)
            candidate = ("" if current == "/" else current) + "/" + token
            if candidate in self.soft:
                if candidate in active:
                    _problem("UNSUPPORTED", "soft-link cycle", candidate)
                active.add(candidate)
                used.append(candidate)
                text = self.soft[candidate]
                parent = candidate.rsplit("/", 1)[0] or "/"
                combined = text if text.startswith("/") else parent.rstrip("/") + "/" + text
                # A completion marker removes the active link BEFORE traversing
                # the remaining caller suffix. Reusing a completed link later in
                # a finite path is valid; reentering an active target is a cycle.
                pending.appendleft((candidate,))
                for part in reversed([p for p in combined.split("/") if p and p != "."]):
                    pending.appendleft(part)
                current = "/"
                continue
            if candidate not in self.names:
                _problem("UNSUPPORTED", "dangling internal soft link or missing selected path", path)
            current = candidate
            if any(isinstance(p, str) for p in pending) and not isinstance(self.objects[self.names[current]], h5py.Group):
                _problem("UNSUPPORTED", "non-group inside link path", path)
        return current, used

    def _target(self, ref, path):
        if not ref:
            return None
        if isinstance(ref, h5py.RegionReference):
            _problem("UNSUPPORTED", "region references are unsupported", path)
        try:
            obj = self.file[ref]
        except (ValueError, KeyError, RuntimeError) as exc:
            raise CarryError("UNSUPPORTED", "dangling object reference", path) from exc
        oid = _identity(obj)
        if oid not in self.objects:
            _problem("UNSUPPORTED", "reference to unnamed or unsupported object", path)
        return oid

    def read(self, dataset, selection):
        self.tick()
        # Independent admission, not the planner/writer's whole-payload guard.
        # Sparse allocation status is not proof that every element was written.
        if (dataset.shape is not None and math.prod(dataset.shape) and
                dataset.id.get_create_plist().get_fill_time() == h5py.h5d.FILL_TIME_NEVER):
            _problem("UNSUPPORTED", "nonempty FILL_TIME_NEVER payload reads are unqualified", dataset.name)
        data = np.asarray(dataset[selection], dtype=dataset.dtype)
        size = int(data.size) * dataset.dtype.itemsize
        if size > self.limits.chunk_bytes:
            _problem("RESOURCE", "native read exceeds comparison chunk budget", dataset.name)
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
                    offset = 0
                    for selection in _blocks(obj.shape, obj.dtype.itemsize, self.limits.chunk_bytes):
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

    def _scales(self, oid, obj, path):
        if "DIMENSION_LIST" in obj.attrs:
            dimensions = obj.attrs["DIMENSION_LIST"]
            if np.asarray(dimensions).shape != (obj.ndim,):
                _problem("UNSUPPORTED", "dimension list rank mismatch", path)
            for axis, refs in enumerate(dimensions):
                for ref in np.asarray(refs).reshape(-1):
                    self.tick(1)
                    target = self._target(ref, path)
                    if target is None:
                        _problem("UNSUPPORTED", "null forward dimension-scale reference", path)
                    self.forward.append((oid, axis, target))
        if "REFERENCE_LIST" in obj.attrs:
            reverse = np.asarray(obj.attrs["REFERENCE_LIST"])
            if reverse.dtype.names != ("dataset", "dimension"):
                _problem("UNSUPPORTED", "malformed reverse dimension-scale record", path)
            for pair in reverse.reshape(-1):
                self.tick(1)
                target = self._target(pair["dataset"], path)
                axis = int(pair["dimension"])
                if target is None or not isinstance(self.objects[target], h5py.Dataset):
                    _problem("UNSUPPORTED", "invalid reverse scale consumer", path)
                if axis < 0 or axis >= self.objects[target].ndim:
                    _problem("UNSUPPORTED", "reverse scale axis is outside consumer rank", path)
                self.reverse.append((target, axis, oid))

    def required(self, selections):
        """Independent fixed point: retain names, promote referenced groups,
        then discover dependencies of every retained object and ancestor."""
        hard, soft, included, expanded = {"/"}, set(), {self.names["/"]}, set()

        def entry(path, descendants):
            pending = deque([(path, descendants)])
            while pending:
                p, deep = pending.popleft()
                terminal, through = self.resolve(p)
                for softpath in through:
                    if softpath not in soft:
                        soft.add(softpath)
                        # Keeping an intermediate soft entry also retains its
                        # complete target when that target is a group, even if
                        # the original selection continues to one child.
                        pending.append((softpath, True))
                    parent = softpath.rsplit("/", 1)[0] or "/"
                    pending.append((parent, False))
                # A soft link through a group can require hard containers on both
                # sides; preserve the exact named hard target needed by its text.
                p = terminal
                hard.add(p)
                oid = self.names[p]
                included.add(oid)
                parent = p.rsplit("/", 1)[0] or "/"
                while p != "/":
                    hard.add(parent)
                    included.add(self.names[parent])
                    p, parent = parent, parent.rsplit("/", 1)[0] or "/"
                if deep and isinstance(self.objects[oid], h5py.Group) and oid not in expanded:
                    expanded.add(oid)
                    pending.extend((child, True) for child in self.children[oid])

        # Seed all explicitly retained names before choosing names for deps.
        for path in selections:
            validate_path(path, self.limits)
            entry(path, True)
        done = set()
        scale_targets = {}
        for consumer, _, scale in self.forward:
            scale_targets.setdefault(consumer, set()).add(scale)
        while included - done:
            for oid in sorted(included - done, key=lambda o: self.aliases[o][0]):
                self.tick()
                done.add(oid)
                targets = {target for _, _, target in self.references[oid] if target is not None}
                targets.update(scale_targets.get(oid, ()))
                for target in sorted(targets, key=lambda o: self.aliases[o][0]):
                    names = [p for p in self.aliases[target] if p in hard]
                    chosen = names[0] if names else self.aliases[target][0]
                    # Ancestor-only groups become full selections when referenced.
                    entry(chosen, True)
        return self.project(hard, soft)

    def project(self, hard, soft):
        canonical = {}
        for path in sorted(hard):
            canonical.setdefault(self.names[path], path)
        objects, payload = [], 0
        for oid, path in sorted(canonical.items(), key=lambda v: v[1]):
            obj = self.objects[oid]
            meta = self.metadata[oid].copy()
            if isinstance(obj, h5py.Dataset):
                size = math.prod(obj.shape) if obj.shape is not None else 0
                payload += size * obj.dtype.itemsize
                if payload > self.limits.max_payload_bytes:
                    _problem("RESOURCE", "selected logical payload exceeds budget", path)
                if meta["dtype"]["kind"] == "fixed":
                    digest = hashlib.sha256()
                    for selection in _blocks(obj.shape, obj.dtype.itemsize, self.limits.chunk_bytes):
                        digest.update(self.read(obj, selection).tobytes(order="C"))
                    meta["payload_sha256"] = digest.hexdigest()
            objects.append({"id": path, "kind": "dataset" if isinstance(obj, h5py.Dataset) else "group",
                            "metadata": meta})
        links = [{"path": path, "kind": "hard", "target": canonical[self.names[path]]}
                 for path in sorted(hard) if path != "/"]
        links += [{"path": path, "kind": "soft", "target": self.soft[path]} for path in sorted(soft)]
        references = []
        for oid, owner in canonical.items():
            for attribute, index, target in self.references[oid]:
                if target is not None and target not in canonical:
                    _problem("MISMATCH", "retained reference target absent from closure", owner)
                references.append({"owner": owner, "attribute": attribute, "index": index,
                                   "target": None if target is None else canonical[target]})
        scales = []
        for consumer, axis, scale in self.forward:
            if consumer in canonical:
                if scale not in canonical:
                    _problem("MISMATCH", "retained scale absent from closure", canonical[consumer])
                scales.append({"consumer": canonical[consumer], "axis": axis, "scale": canonical[scale]})
        graph = {"objects": objects, "links": sorted(links, key=lambda v: v["path"]),
                 "references": sorted(references, key=canonical_json),
                 "scales": sorted(scales, key=canonical_json), "payload_bytes": payload}
        if len(canonical_json(graph)) > self.limits.max_plan_bytes:
            _problem("RESOURCE", "independent graph exceeds serialized plan budget")
        return graph


def _graph_differences(expected, actual, location):
    diagnostics = []
    for field in ("objects", "links", "references", "scales"):
        expected_items = {canonical_json(v): v for v in expected[field]}
        actual_items = {canonical_json(v): v for v in actual.get(field, [])}
        want = Counter(canonical_json(v) for v in expected[field])
        got = Counter(canonical_json(v) for v in actual.get(field, []))
        if want != got:
            missing, extra = want - got, got - want
            first = next(iter(missing or extra))
            item = expected_items[first] if missing else actual_items[first]
            path = item.get("id", item.get("path", item.get("owner", item.get("consumer"))))
            diagnostic = {"code": "MISMATCH", "message":
                          f"{location} {field} differ from independently derived source closure "
                          f"({sum(missing.values())} missing/changed, {sum(extra.values())} extra/changed)"}
            if path is not None:
                diagnostic["path"] = path
            diagnostics.append(diagnostic)
    if expected["payload_bytes"] != actual.get("payload_bytes"):
        diagnostics.append({"code": "MISMATCH", "message": f"{location} logical payload accounting differs"})
    return diagnostics


def _coverage():
    return {"source_equality": False, "max_read_bytes": 0, "data_blocks": 0,
            "objects": 0, "links": 0, "references": 0, "scales": 0,
            "closure_independently_derived": False}


def _report(status, diagnostics, coverage, limits):
    return {"status": status, "diagnostics": diagnostics, "coverage": coverage,
            "runtime": _runtime(), "limits": limits.to_dict()}


def verify_export(source, output, plan, limits):
    """Check the plan against fresh source closure, then exact output semantics.

    Expected, qualified failures are reports. Unexpected programmer/native
    exceptions propagate to the worker and can never produce verified status.
    """
    coverage = _coverage()
    try:
        if fingerprint(source) != plan["source"]:
            _problem("SOURCE_CHANGED", "source does not match the plan fingerprint")
        with h5py.File(source, "r") as src:
            original = _Inventory(src, limits, coverage)
            expected = original.required(plan["selections"])
            coverage["closure_independently_derived"] = True
            diagnostics = _graph_differences(expected, plan["graph"], "plan")
            if diagnostics:
                return _report("mismatch", diagnostics, coverage, limits)
            with h5py.File(output, "r") as dst:
                delivered = _Inventory(dst, limits, coverage)
                actual = delivered.project(set(delivered.names), set(delivered.soft))
                diagnostics.extend(_graph_differences(expected, actual, "output"))
                # JSON descriptors intentionally retain the exact v1 schema.
                # Direct source-derived native checks cover the complete DCPL,
                # including reference, empty and null datasets. H5Pequal omits
                # chunk-option flags in the two qualified builds.
                if not diagnostics:
                    for record in expected["objects"]:
                        if record["kind"] != "dataset":
                            continue
                        path = record["id"]
                        left, right = src[path], dst[path]
                        if not left.id.get_type().equal(right.id.get_type()):
                            diagnostics.append({"code": "MISMATCH", "message": "low-level dataset datatype differs", "path": path})
                        source_dcpl, output_dcpl = left.id.get_create_plist(), right.id.get_create_plist()
                        if not source_dcpl.equal(output_dcpl):
                            diagnostics.append({"code": "MISMATCH", "message": "native dataset creation properties differ", "path": path})
                        if source_dcpl.get_layout() == h5py.h5d.CHUNKED:
                            try:
                                source_options = chunk_options(source_dcpl)
                                output_options = chunk_options(output_dcpl)
                            except CarryError as exc:
                                if exc.path is None:
                                    exc.path = path
                                raise
                            if source_options != output_options:
                                diagnostics.append({"code": "MISMATCH", "message": "native dataset chunk options differ", "path": path})
                # Independent byte comparison as well as descriptor/hash checks.
                # There is no array equality coercion: -0 and NaN payload bits count.
                if not diagnostics:
                    for record in expected["objects"]:
                        if record["kind"] != "dataset" or record["metadata"]["dtype"]["kind"] != "fixed":
                            continue
                        path = record["id"]
                        left, right = src[path], dst[path]
                        for selection in _blocks(left.shape, left.dtype.itemsize, limits.chunk_bytes):
                            a = original.read(left, selection).tobytes(order="C")
                            b = delivered.read(right, selection).tobytes(order="C")
                            if a != b:
                                diagnostics.append({"code": "MISMATCH", "message": "logical data bytes differ", "path": path})
                                break
                for field in ("objects", "links", "references", "scales"):
                    coverage[field] = len(expected[field])
                coverage["payload_bytes"] = expected["payload_bytes"]
        if fingerprint(source) != plan["source"]:
            _problem("SOURCE_CHANGED", "source changed during verification")
        coverage["source_equality"] = not diagnostics
        return _report("mismatch" if diagnostics else "verified", diagnostics, coverage, limits)
    except CarryError as exc:
        return _report("mismatch" if exc.code == "MISMATCH" else "incomplete", [exc.to_dict()], coverage, limits)
    except OSError as exc:
        return _report("incomplete", [{"code": "ERROR", "message": f"cannot complete native file read: {exc}"}], coverage, limits)


def inspect_output(output, limits):
    """Output-only supported-profile and reference consistency; no equality claim."""
    coverage = _coverage()
    try:
        with h5py.File(output, "r") as file:
            inventory = _Inventory(file, limits, coverage)
            graph = inventory.project(set(inventory.names), set(inventory.soft))
            for field in ("objects", "links", "references", "scales"):
                coverage[field] = len(graph[field])
            coverage["payload_bytes"] = graph["payload_bytes"]
        return _report("inspected", [], coverage, limits)
    except CarryError as exc:
        return _report("incomplete", [exc.to_dict()], coverage, limits)
    except OSError as exc:
        return _report("incomplete", [{"code": "ERROR", "message": f"cannot complete native file read: {exc}"}], coverage, limits)
