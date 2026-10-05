"""Independent verifier acceptance cases, using only original benign fixtures.

The unittest parent never imports a native library. Each case starts a bounded
Linux child which creates its own fixture, plan and independent output.
"""
from __future__ import annotations

import ast
import json
import importlib.util
import os
from pathlib import Path
import resource
import subprocess
import sys
import unittest

def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (40, 40))
    resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class VerifierTests(unittest.TestCase):
    def run_case(self, case):
        package = importlib.util.find_spec("h5carry")
        self.assertIsNotNone(package, "h5carry package is not importable")
        package_parent = Path(package.origin).resolve().parent.parent
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", HDF5_PLUGIN_PRELOAD="::",
                   PYTHONPATH=str(package_parent))
        env.pop("HDF5_PLUGIN_PATH", None)
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), case],
                              env=env, preexec_fn=_limits, capture_output=True,
                              text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return json.loads(proc.stdout)

    def test_fresh_manual_export_and_output_only_inspection(self):
        self.run_case("baseline")

    def test_independent_import_boundary(self):
        package = importlib.util.find_spec("h5carry")
        self.assertIsNotNone(package, "h5carry package is not importable")
        package_root = Path(package.origin).resolve().parent
        verifier = importlib.util.find_spec("h5carry.verify")
        self.assertIsNotNone(verifier, "independent verifier module is missing")
        file = Path(verifier.origin).resolve()
        self.assertTrue(file.is_relative_to(package_root),
                        "verifier must belong to the imported h5carry package")
        self.assertTrue(file.exists(), "independent verifier module is missing")
        tree = ast.parse(file.read_text())
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
            elif isinstance(node, ast.Import):
                imported.extend(n.name for n in node.names)
        self.assertFalse([name for name in imported
                          if any(p in {"scan", "write", "plan"} for p in name.split("."))])

    def test_mutations_are_rejected(self):
        for case in ("data", "signed_zero", "nan_payload", "dtype", "shape", "compression",
                     "attribute", "attribute_type", "split_alias", "wrong_ref", "null_ref",
                     "attribute_ref", "soft_text", "missing_scale", "extra_scale",
                     "reverse_scale", "unexpected", "external", "missing_child_both",
                     "missing_dependency_both", "ancestor_dependency_both", "missing_alias_both",
                     "missing_group_dependency_child_both", "conflate_objects", "changed_source", "resource_objects",
                     "resource_edges", "resource_payload", "resource_attribute", "resource_chunk",
                     "inspect_dangling", "inspect_reverse"):
            with self.subTest(case=case):
                self.run_case(case)

    def test_bounded_reads_and_reference_chunks(self):
        self.run_case("bounded")
        self.run_case("bounded_large")

    def test_unselected_large_logical_payload_is_not_hashed(self):
        self.run_case("unselected_payload")

    def test_non_utf8_names_are_explicitly_unsupported(self):
        for case in ("invalid_link_name", "invalid_attr_name"):
            with self.subTest(case=case):
                self.run_case(case)

    def test_selection_through_soft_group_retains_its_complete_target(self):
        self.run_case("soft_group_path")

    def test_soft_links_must_resolve_with_native_default_access(self):
        self.run_case("native_soft_limit")

    def test_empty_large_element_is_bounded_before_fill_read(self):
        self.run_case("oversized_empty_element")

    def test_repeated_soft_link_in_finite_path_is_not_a_cycle(self):
        self.run_case("repeated_soft")


def _native(case):
    import copy
    import hashlib
    import json
    import tempfile
    import h5py
    import numpy as np
    from h5carry.model import Limits, fingerprint
    from h5carry.verify import inspect_output, verify_export
    from h5carry.profile import describe_object

    limits = Limits() if case == "bounded_large" else Limits(chunk_bytes=32)
    with tempfile.TemporaryDirectory(prefix="h5carry-verifier-fixture-") as tmp:
        source, output = Path(tmp) / "source.h5", Path(tmp) / "output.h5"

        def create(path, full):
            with h5py.File(path, "w") as f:
                run = f.create_group("run")
                if case == "soft_group_path":
                    f["selection"] = h5py.SoftLink("/run")
                cal = f.create_group("cal")
                gain = f.create_dataset("deps/gain", data=np.array([2., 3.]))
                gain.attrs["units"] = "counts"
                if full:
                    f["deps/gain_unused"] = gain
                anchor = f.create_dataset("audit/anchor", data=np.array(17, dtype=">i4"))
                f.attrs["audit"] = anchor.ref
                cfg = f.create_group("config")
                cfg.create_dataset("item", data=np.array([8, 9], dtype="<i2"))
                run.attrs["configuration"] = cfg.ref
                f.attrs["title"] = "Original verifier fixture"
                run.attrs["note"] = np.array([1, 2], dtype=">i2")
                data = run.create_dataset("data", data=np.arange(4, dtype=">i4").reshape(2, 2),
                                          chunks=(1, 2), compression="gzip", compression_opts=4,
                                          shuffle=True, fletcher32=True, maxshape=(None, 2), fillvalue=-7)
                run["alias"] = data
                data.attrs["gain"] = gain.ref
                data.attrs["ordinary"] = np.array([4, 5], dtype="<i4")
                r = run.create_dataset("refs", (9,), dtype=h5py.ref_dtype, chunks=(3,),
                                       compression="gzip", compression_opts=7, maxshape=(None,))
                r[:] = [gain.ref, h5py.Reference(), cfg.ref] * 3
                run.attrs["null_ref"] = h5py.Reference()
                s1 = cal.create_dataset("scale", data=np.array([10, 20], dtype="<i8"))
                s2 = cal.create_dataset("scale2", data=np.array([30, 40], dtype="<i8"))
                s1.make_scale("coordinate")
                s2.make_scale("coordinate")
                data.dims[0].attach_scale(s1)
                data.dims[0].attach_scale(s2)
                data.dims[1].attach_scale(s1)
                data.dims[0].label = "rows"
                data.dims[1].label = "columns"
                run["soft"] = h5py.SoftLink("/cal/scale")
                run["soft_chain"] = h5py.SoftLink("/run/soft")
                if case == "repeated_soft":
                    run["back"] = h5py.SoftLink("/")
                    run["finite"] = h5py.SoftLink("/run/back/run/back/run/data")
                run.create_dataset("zero", data=np.array(-0., dtype="<f8"))
                run.create_dataset("nan", data=np.array([0x7ff8000000000042], dtype="<u8").view("<f8"))
                run.create_dataset("empty", shape=(0, 2), dtype="<f4")
                run.create_dataset("null", data=h5py.Empty("<f8"))
                scalar = run.create_dataset("scalar", data=np.array(5, dtype="<i8"))
                run["scalar_alias"] = scalar
                run.create_dataset("twin1", data=np.array([1, 2], dtype="<i4"))
                run.create_dataset("twin2", data=np.array([1, 2], dtype="<i4"))
                run.create_dataset("complex", data=np.array([complex(-0., 2), 1+3j], dtype="<c16"))
                big_shape = (2, 3, 500000) if case == "bounded_large" else (2, 3, 20)
                run.create_dataset("big", data=np.arange(np.prod(big_shape), dtype="<i4").reshape(big_shape))
                if full and case == "unselected_payload":
                    f.create_dataset("unselected/huge", shape=(2**40,), dtype="u1")
                if full:
                    other = f.create_dataset("excluded/marker", data=np.array([91, 92]))
                    other.attrs["unique_excluded"] = "not retained"
                    other.dims[0].attach_scale(s1)

        create(source, True)
        create(output, False)
        paths = ["/", "/run", "/cal", "/cal/scale", "/cal/scale2", "/deps", "/deps/gain",
                 "/audit", "/audit/anchor", "/config", "/config/item", "/run/alias", "/run/refs",
                 "/run/zero", "/run/nan", "/run/empty", "/run/null", "/run/scalar", "/run/complex", "/run/big", "/run/twin1", "/run/twin2"]
        with h5py.File(source, "r") as f:
            objects = [{"id": p, "kind": "group" if isinstance(f[p], h5py.Group) else "dataset",
                        "metadata": describe_object(f[p], limits)} for p in sorted(paths)]
        links = [{"path": p, "kind": "hard", "target": p} for p in paths if p != "/"]
        links += [{"path": "/run/scalar_alias", "kind": "hard", "target": "/run/scalar"},
                  {"path": "/run/data", "kind": "hard", "target": "/run/alias"},
                  {"path": "/run/soft", "kind": "soft", "target": "/cal/scale"},
                  {"path": "/run/soft_chain", "kind": "soft", "target": "/run/soft"}]
        if case == "soft_group_path":
            links += [{"path": "/selection", "kind": "soft", "target": "/run"}]
        if case == "repeated_soft":
            links += [{"path": "/run/back", "kind": "soft", "target": "/"},
                      {"path": "/run/finite", "kind": "soft", "target": "/run/back/run/back/run/data"}]
        refs = [{"owner": "/", "attribute": "audit", "index": 0, "target": "/audit/anchor"},
                {"owner": "/run", "attribute": "configuration", "index": 0, "target": "/config"},
                {"owner": "/run", "attribute": "null_ref", "index": 0, "target": None},
                {"owner": "/run/alias", "attribute": "gain", "index": 0, "target": "/deps/gain"}]
        refs += [{"owner": "/run/refs", "attribute": None, "index": i,
                  "target": ["/deps/gain", None, "/config"][i % 3]} for i in range(9)]
        scales = [{"consumer": "/run/alias", "axis": 0, "scale": "/cal/scale"},
                  {"consumer": "/run/alias", "axis": 0, "scale": "/cal/scale2"},
                  {"consumer": "/run/alias", "axis": 1, "scale": "/cal/scale"}]
        with h5py.File(source, "r") as f:
            payload_bytes = sum(int(f[p].size or 0) * f[p].dtype.itemsize for p in paths if isinstance(f[p], h5py.Dataset))
        graph = {"objects": objects, "links": sorted(links, key=lambda x: x["path"]),
                 "references": refs, "scales": scales, "payload_bytes": payload_bytes,
                 "reasons": {p: ["manual expected fixture"] for p in paths}}
        plan = {"format": "h5carry-plan", "version": 1, "source": fingerprint(source),
                "selections": ["/selection/data"] if case == "soft_group_path" else ["/run"], "limits": limits.to_dict(), "graph": graph}

        if case in {"invalid_link_name", "invalid_attr_name"}:
            with h5py.File(output, "r+") as f:
                if case == "invalid_link_name":
                    f.create_dataset(b"invalid_\xff", data=[1])
                else:
                    f.attrs[b"invalid_\xff"] = 1
            report = inspect_output(str(output), limits)
            assert report["status"] == "incomplete" and report["diagnostics"][0]["code"] == "UNSUPPORTED", report
            assert report["diagnostics"][0].get("path"), report
            return {"case": case, "report": report}
        if case == "native_soft_limit":
            with h5py.File(output, "r+") as f:
                for i in range(20):
                    f[f"chain{i}"] = h5py.SoftLink(f"/chain{i+1}" if i < 19 else "/run/data")
            with h5py.File(output, "r") as f:
                try:
                    f["chain0"]
                except (RuntimeError, KeyError):
                    pass
                else:
                    raise AssertionError("fixture does not exceed this runtime's native default traversal bound")
            report = inspect_output(str(output), limits)
            assert report["status"] == "incomplete", report
            return {"case": case, "report": report}
        if case == "oversized_empty_element":
            with h5py.File(output, "r+") as f:
                f.create_dataset("oversized", shape=(0,), dtype="S1024")
            import h5carry.verify as verifier
            original_creation = verifier.dataset_creation
            def creation_with_bound(dataset):
                assert dataset.dtype.itemsize <= limits.chunk_bytes, "unbounded element reaches fill-value reader"
                return original_creation(dataset)
            verifier.dataset_creation = creation_with_bound
            report = inspect_output(str(output), limits)
            assert report["status"] == "incomplete" and report["diagnostics"][0]["code"] == "RESOURCE", report
            return {"case": case, "report": report}
        if case == "repeated_soft":
            inspection = inspect_output(str(output), limits)
            assert inspection["status"] == "inspected", inspection
            return {"case": case, "report": inspection}
        if case in {"baseline", "bounded", "bounded_large", "unselected_payload", "soft_group_path"}:
            actual_reads = []
            original_getitem = h5py.Dataset.__getitem__
            def measured_getitem(dataset, selection, *args, **kwargs):
                assert dataset.name != "/unselected/huge", "unselected payload was read"
                result = original_getitem(dataset, selection, *args, **kwargs)
                actual_reads.append(np.asarray(result, dtype=dataset.dtype).nbytes)
                assert actual_reads[-1] <= limits.chunk_bytes, (dataset.name, selection, actual_reads[-1])
                return result
            h5py.Dataset.__getitem__ = measured_getitem
            report = verify_export(str(source), str(output), plan, limits)
            assert report["status"] == "verified", report
            assert report["coverage"]["source_equality"] is True, report
            assert report["coverage"]["closure_independently_derived"] is True, report
            inspection = inspect_output(str(output), limits)
            assert inspection["status"] == "inspected", inspection
            assert inspection["coverage"]["source_equality"] is False, inspection
            assert actual_reads and max(actual_reads) <= limits.chunk_bytes
            h5py.Dataset.__getitem__ = original_getitem
            if case in {"bounded", "bounded_large"}:
                assert report["coverage"]["max_read_bytes"] <= limits.chunk_bytes, report
                assert report["coverage"]["data_blocks"] > 10, report
            return {"case": case, "report": report, "inspection": inspection,
                    "observed_max_read_bytes": max(actual_reads)}

        if case.startswith("resource_"):
            settings = {"resource_objects": {"max_objects": 1}, "resource_edges": {"max_edges": 1},
                        "resource_payload": {"max_payload_bytes": 1},
                        "resource_attribute": {"max_attribute_bytes": 1}, "resource_chunk": {"chunk_bytes": 1}}
            report = verify_export(str(source), str(output), plan, Limits(**settings[case]))
            assert report["status"] == "incomplete" and report["diagnostics"][0]["code"] == "RESOURCE", report
            return {"case": case, "report": report}
        if case == "changed_source":
            with h5py.File(source, "r+") as f: f["run/big"][0, 0, 0] = 777
            report = verify_export(str(source), str(output), plan, limits)
            assert report["status"] == "incomplete" and report["diagnostics"][0]["code"] == "SOURCE_CHANGED", report
            assert report["coverage"]["source_equality"] is False, report
            assert report["coverage"]["closure_independently_derived"] is False, report
            return {"case": case, "report": report}

        def omit(path):
            removed = next((o for o in graph["objects"] if o["id"] == path), None)
            if removed and removed["kind"] == "dataset":
                import math
                meta = removed["metadata"]
                count = 0 if meta["shape"] is None else math.prod(meta["shape"])
                graph["payload_bytes"] -= count * np.dtype(meta["dtype"]["str"]).itemsize
            graph["objects"] = [o for o in graph["objects"] if o["id"] != path]
            graph["links"] = [v for v in graph["links"] if v["path"] != path and v["target"] != path]
            graph["references"] = [v for v in graph["references"] if v["owner"] != path and v["target"] != path]
            graph["scales"] = [v for v in graph["scales"] if v["consumer"] != path and v["scale"] != path]
            graph["reasons"].pop(path, None)

        with h5py.File(output, "r+") as f:
            if case == "data": f["/run/data"][0, 0] = 88
            elif case == "signed_zero": f["/run/zero"][()] = 0.
            elif case == "nan_payload": f["/run/nan"][:] = np.array([0x7ff8000000000099], dtype="<u8").view("<f8")
            elif case in {"dtype", "shape", "compression"}:
                name = "/run/big"
                values = f[name][()]
                del f[name]
                if case == "dtype": f.create_dataset(name, data=values.astype(">i4"))
                elif case == "shape": f.create_dataset(name, data=values.reshape(6, 20))
                else: f.create_dataset(name, data=values, compression="gzip", compression_opts=8)
            elif case == "attribute": del f["/run/data"].attrs["ordinary"]
            elif case == "attribute_type":
                del f["/run/data"].attrs["ordinary"]
                f["/run/data"].attrs["ordinary"] = np.array([4, 5], dtype=">i4")
            elif case == "split_alias":
                del f["/run/scalar_alias"]
                f.create_dataset("/run/scalar_alias", data=np.array(5, dtype="<i8"))
            elif case == "conflate_objects":
                del f["/run/twin2"]
                f["/run/twin2"] = f["/run/twin1"]
            elif case == "wrong_ref": f["/run/refs"][0] = f["/run/zero"].ref
            elif case == "null_ref": f["/run/refs"][1] = f["/deps/gain"].ref
            elif case == "attribute_ref": f["/run/data"].attrs.modify("gain", f["/run/zero"].ref)
            elif case == "soft_text":
                del f["/run/soft_chain"]
                f["/run/soft_chain"] = h5py.SoftLink("/cal/scale")
            elif case == "missing_scale": f["/run/data"].dims[0].detach_scale(f["/cal/scale2"])
            elif case == "extra_scale": f["/run/data"].dims[1].attach_scale(f["/cal/scale2"])
            elif case in {"reverse_scale", "inspect_reverse"}:
                arr = f["/cal/scale"].attrs["REFERENCE_LIST"]
                arr[0]["dimension"] = 17
                f["/cal/scale"].attrs.modify("REFERENCE_LIST", arr)
            elif case == "inspect_dangling":
                del f["/run/soft"]
                f["/run/soft"] = h5py.SoftLink("/missing")
            elif case == "unexpected": f.create_dataset("/unexpected", data=[1])
            elif case == "external": f["/outside"] = h5py.ExternalLink("never-open-this-original-absent.h5", "/data")
            elif case == "missing_alias_both":
                del f["/run/data"]
                graph["links"] = [v for v in graph["links"] if v["path"] != "/run/data"]
            elif case == "missing_child_both":
                del f["/run/twin1"]
                omit("/run/twin1")
            elif case == "missing_dependency_both":
                del f["/deps/gain"]
                omit("/deps/gain")
            elif case == "ancestor_dependency_both":
                del f.attrs["audit"]
                del f["/audit/anchor"]
                omit("/audit/anchor")
                for obj in graph["objects"]:
                    if obj["id"] == "/": obj["metadata"]["attributes"] = [a for a in obj["metadata"]["attributes"] if a["name"] != "audit"]
            elif case == "missing_group_dependency_child_both":
                del f["/config/item"]
                omit("/config/item")
            else: raise AssertionError(case)
        if case.startswith("inspect_"):
            report = inspect_output(str(output), limits)
            assert report["status"] == "incomplete" and report["coverage"]["source_equality"] is False, report
            return {"case": case, "report": report}
        # Validate omitted-plan structure before verification: omission tests are
        # not merely exercising dangling IDs or stale payload accounting.
        if case in {"missing_child_both", "missing_group_dependency_child_both", "missing_alias_both"}:
            from h5carry.plan import validate_plan
            validate_plan(plan)
        report = verify_export(str(source), str(output), plan, limits)
        assert fingerprint(source) == plan["source"], "verifier changed the source"
        assert report["status"] in {"mismatch", "incomplete"}, (case, report)
        assert report["diagnostics"], (case, report)
        assert report["coverage"]["source_equality"] is False, (case, report)
        if case.endswith("_both"):
            assert any(d["code"] == "MISMATCH" and "plan" in d["message"].lower()
                       for d in report["diagnostics"]), (case, report)
        return {"case": case, "report": report}


if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        print(json.dumps(_native(sys.argv[1]), sort_keys=True))
    else:
        unittest.main()
