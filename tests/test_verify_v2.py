"""Bounded-native v2 verifier tests; fixtures never use the v2 planner or writer."""
from __future__ import annotations
import ast
import copy
import importlib
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest
from h5carry.model import Limits, canonical_json, fingerprint
if os.environ.get("H5CARRY_TEST_NATIVE_CHILD") == "1":
    import h5py
    import numpy as np
    from h5carry.profile import RESERVED, describe_object


def request(objects, mappings=()):
    return {"format":"h5carry-selection","version":1,"objects":sorted(objects,key=lambda x:x["path"]),
        "scale_mappings":sorted(mappings,key=lambda x:(x["consumer"],x["axis"],x["scale"])),"storage_policy":"fixed-snapshot-v1"}


def whole(path): return {"path":path,"selection":{"kind":"whole"}}

def box(path,start,stop): return {"path":path,"selection":{"kind":"box","start":list(start),"stop":list(stop)}}

def mapping(consumer,axis,scale): return {"consumer":consumer,"axis":axis,"scale":scale,"mapping":"index"}

def address(obj): return int(h5py.h5o.get_info(obj.id).addr)


def manual_graph(file,paths,limits):
    """Inventory literal expected paths, without selection/closure derivation."""
    ids={}
    for path in sorted(paths):
        if path == "/" or not isinstance(file.get(path,getlink=True),h5py.SoftLink): ids.setdefault(address(file[path]),path)
    objects=[]; links=[]; refs=[]; scales=[]; payload=0
    for uid,path in sorted(ids.items(),key=lambda pair:pair[1]):
        obj=file[path]; meta=describe_object(obj,limits,hash_payload=True)
        objects.append({"id":path,"kind":"dataset" if isinstance(obj,h5py.Dataset) else "group","metadata":meta})
        for attr in meta["attributes"]:
            if attr["dtype"]["kind"]=="reference" and attr["shape"] is not None:
                for index,ref in enumerate(np.asarray(obj.attrs[attr["name"]]).reshape(-1)):
                    refs.append({"owner":path,"attribute":attr["name"],"index":index,"target":ids[address(file[ref])] if ref else None})
        if isinstance(obj,h5py.Dataset):
            payload+=(0 if obj.shape is None else math.prod(obj.shape))*obj.dtype.itemsize
            if h5py.check_dtype(ref=obj.dtype) is not None and obj.shape is not None:
                for index,ref in enumerate(np.asarray(obj[()]).reshape(-1)):
                    refs.append({"owner":path,"attribute":None,"index":index,"target":ids[address(file[ref])] if ref else None})
            for axis in range(obj.ndim):
                for scale in obj.dims[axis].values(): scales.append({"consumer":path,"axis":axis,"scale":ids[address(scale)]})
    for path in sorted(set(paths)-{"/"}):
        link=file.get(path,getlink=True)
        links.append({"path":path,"kind":"soft" if isinstance(link,h5py.SoftLink) else "hard","target":link.path if isinstance(link,h5py.SoftLink) else ids[address(file[path])]})
    return {"objects":objects,"links":links,"references":sorted(refs,key=canonical_json),"scales":sorted(scales,key=canonical_json),
        "payload_bytes":payload,"reasons":{item["id"]:["manual fixture"] for item in objects}}


def manual_plan(source,output,req,paths,boxes,limits):
    with h5py.File(source,"r") as src,h5py.File(output,"r") as dst:
        graph=manual_graph(dst,paths,limits); originals=[]; selections=[]; transforms=[]
        for record in graph["objects"]:
            path=record["id"]; meta=describe_object(src[path],limits,hash_payload=False)
            originals.append({"id":path,"kind":record["kind"],"metadata":meta})
            selected=boxes.get(path,{"kind":"whole"}); selections.append({"id":path,"selection":selected})
            if selected["kind"]=="box":
                shape=[b-a for a,b in zip(selected["start"],selected["stop"])]; chunks=meta["creation"]["chunks"]
                transforms.append({"id":path,"source_shape":meta["shape"],"output_shape":shape,
                    "source_maxshape":meta["creation"]["maxshape"],"output_maxshape":shape,"source_chunks":chunks,
                    "output_chunks":None if chunks is None else [min(c,max(1,n)) for c,n in zip(chunks,shape)]})
        return {"format":"h5carry-plan","version":2,"source":fingerprint(source),"request":req,"limits":limits.to_dict(),
            "graph":graph,"source_objects":originals,"selections":selections,"transformations":transforms}


def manual_copy(source,output,paths,boxes):
    """Independent literal h5py construction, not the product copying engine."""
    with h5py.File(source,"r") as src,h5py.File(output,"w") as dst:
        names={}
        for path in sorted(paths):
            if path != "/" and isinstance(src.get(path,getlink=True),h5py.SoftLink): dst[path]=src.get(path,getlink=True); continue
            obj=src[path]; uid=address(obj)
            if uid in names: dst[path]=dst[names[uid]]; continue
            names[uid]=path
            if isinstance(obj,h5py.Group):
                if path!="/": dst.create_group(path)
                continue
            selected=boxes.get(path); shape=obj.shape if selected is None else tuple(b-a for a,b in zip(selected["start"],selected["stop"]))
            dcpl=obj.id.get_create_plist().copy()
            if selected is not None and obj.chunks is not None: dcpl.set_chunk(tuple(min(c,max(1,n)) for c,n in zip(obj.chunks,shape)))
            maximum=shape if selected is not None else obj.maxshape
            if shape is None: space=h5py.h5s.create(h5py.h5s.NULL)
            elif shape==(): space=h5py.h5s.create(h5py.h5s.SCALAR)
            else: space=h5py.h5s.create_simple(shape,tuple(h5py.h5s.UNLIMITED if n is None else n for n in maximum))
            target=h5py.Dataset(h5py.h5d.create(dst.id,path.encode(),obj.id.get_type(),space,dcpl=dcpl))
            if h5py.check_dtype(ref=obj.dtype) is None and shape is not None and all(shape):
                index=() if selected is None else tuple(slice(a,b) for a,b in zip(selected["start"],selected["stop"]))
                target[()]=obj[index]
        def remap(value):
            values=np.asarray(value); out=np.empty(values.shape,dtype=h5py.ref_dtype)
            for index in np.ndindex(values.shape):
                ref=values[index]; out[index]=dst[names[address(src[ref])]].ref if ref else h5py.Reference()
            return out
        for uid,path in names.items():
            obj,target=src[path],dst[path]
            for name in obj.attrs:
                if name in RESERVED: continue
                aid=obj.attrs.get_id(name); value=obj.attrs[name]
                if h5py.check_dtype(ref=aid.dtype) is not None: value=remap(value)
                target.attrs.create(name,value,dtype=aid.dtype)
            if isinstance(obj,h5py.Dataset):
                if h5py.check_dtype(ref=obj.dtype) is not None and obj.shape is not None and all(obj.shape): target[()]=remap(obj[()])
                if obj.is_scale: target.make_scale(bytes(obj.attrs["NAME"]).decode())
        for uid,path in names.items():
            obj,target=src[path],dst[path]
            if isinstance(obj,h5py.Dataset):
                for axis in range(obj.ndim):
                    if obj.dims[axis].label: target.dims[axis].label=obj.dims[axis].label
                    for scale in obj.dims[axis].values(): target.dims[axis].attach_scale(dst[names[address(scale)]])


class SnapshotVerifierTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get("H5CARRY_TEST_NATIVE_CHILD") != "1":
            raise RuntimeError("Native fixtures require python -m tests.native_runner unittest ...")
        self.tmp=tempfile.TemporaryDirectory(prefix="h5carry-v2-independent-"); self.addCleanup(self.tmp.cleanup)
        self.source=Path(self.tmp.name)/"source.h5"; self.output=Path(self.tmp.name)/"output.h5"; self.limits=Limits(chunk_bytes=32)

    def verifier(self):
        self.assertIsNotNone(importlib.util.find_spec("h5carry.verify_v2"),"v2 source-derived verifier is not implemented")
        return importlib.import_module("h5carry.verify_v2")

    def fixture(self):
        with h5py.File(self.source,"w") as f:
            run=f.create_group("run"); f.create_group("cal")
            gain=f.create_dataset("cal/gain",data=np.array([2,3],dtype=">i4"))
            t=f.create_dataset("cal/time",data=np.arange(6,dtype=">i8"),chunks=(2,)); c=f.create_dataset("cal/channel",data=np.arange(4,dtype="i4"))
            t.make_scale("time"); c.make_scale("channel")
            for name in ("A","B"):
                ds=run.create_dataset(name,data=np.arange(24,dtype=">i4").reshape(6,4),chunks=(2,2),maxshape=(None,4),compression="gzip",compression_opts=4,shuffle=True,fletcher32=True,fillvalue=-7)
                ds.dims[0].attach_scale(t); ds.dims[1].attach_scale(c); ds.dims[0].label="time"
            run["A_alias"]=run["A"]; f["unused_alias"]=run["A"]; run["A"].attrs["gain"]=gain.ref
            anchor=f.create_dataset("meta/anchor",data=np.array(17,dtype=">i4")); f.attrs["audit"]=anchor.ref
            cfg=f.create_group("configuration"); cfg.create_dataset("item",data=np.array([8,9],dtype="i2")); run.attrs["configuration"]=cfg.ref
            refs=f.create_dataset("refs",(3,),dtype=h5py.ref_dtype,chunks=(2,)); refs[:]=[gain.ref,h5py.Reference(),cfg.ref]
            f.create_dataset("C",data=np.arange(6,dtype="i4")).dims[0].attach_scale(t)
        self.paths=["/","/cal","/cal/gain","/cal/time","/cal/channel","/run","/run/A","/run/A_alias","/run/B","/meta","/meta/anchor","/configuration","/configuration/item","/refs"]
        self.boxes={"/run/A":box("",[1,1],[4,3])["selection"],"/run/B":box("",[1,1],[4,3])["selection"],"/cal/time":box("",[1],[4])["selection"],"/cal/channel":box("",[1],[3])["selection"]}
        self.req=request([box("/run/A",[1,1],[4,3]),box("/run/A_alias",[1,1],[4,3]),box("/run/B",[1,1],[4,3]),whole("/refs")],[mapping(name,axis,scale) for name in ("/run/A","/run/B") for axis,scale in ((0,"/cal/time"),(1,"/cal/channel"))])
        manual_copy(self.source,self.output,self.paths,self.boxes)
        self.plan=manual_plan(self.source,self.output,self.req,self.paths,self.boxes,self.limits)

    def verify(self): return self.verifier().verify_export(self.source,self.output,self.plan,self.limits)

    def test_correct_hand_built_snapshot_without_planner(self):
        self.fixture(); result=self.verify(); self.assertEqual(result["status"],"verified",result)
        self.assertTrue(result["coverage"]["source_equality"]); self.assertTrue(result["coverage"]["closure_independently_derived"])
        self.assertLessEqual(result["coverage"]["max_read_bytes"],32)
        self.assertFalse(result["coverage"]["positional_mapping_verified"]); self.assertFalse(result["coverage"]["scientific_validity_verified"])

    def test_import_boundary(self):
        module=self.verifier(); tree=ast.parse(Path(module.__file__).read_text()); names=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom): names.append(node.module or "")
            if isinstance(node,ast.Import): names.extend(alias.name for alias in node.names)
        forbidden={"scan","scan_v2","plan","plan_v2","write","write_v2","selection","slicing"}
        self.assertFalse([name for name in names if forbidden.intersection(name.split("."))]); self.assertNotIn("itertools.product",Path(module.__file__).read_text())

    def test_equal_shape_shifted_plan_and_output_are_rejected(self):
        self.fixture(); shifted=copy.deepcopy(self.boxes); shifted["/run/A"]["start"]=[2,1]; shifted["/run/A"]["stop"]=[5,3]
        with h5py.File(self.source,"r") as src,h5py.File(self.output,"r+") as dst: dst["run/A"][:]=src["run/A"][2:5,1:3]
        self.plan=manual_plan(self.source,self.output,self.req,self.paths,shifted,self.limits)
        self.assertEqual(self.verify()["status"],"mismatch")

    def test_coordinated_required_alias_omission_is_rejected(self):
        self.fixture()
        with h5py.File(self.output,"r+") as dst: del dst["run/A_alias"]
        self.paths.remove("/run/A_alias"); self.plan=manual_plan(self.source,self.output,self.req,self.paths,self.boxes,self.limits)
        self.assertEqual(self.verify()["status"],"mismatch")

    def test_coordinated_dependency_data_change_is_rejected(self):
        self.fixture()
        with h5py.File(self.output,"r+") as dst: dst["configuration/item"][0]=99
        self.plan=manual_plan(self.source,self.output,self.req,self.paths,self.boxes,self.limits)
        self.assertEqual(self.verify()["status"],"mismatch")

    def test_source_descriptors_and_transformations_are_not_trusted(self):
        self.fixture()
        for key,field,value in (("source_objects","metadata",{}),("transformations","source_shape",[999,999])):
            with self.subTest(key=key):
                saved=copy.deepcopy(self.plan); self.plan[key][0][field]=value
                self.assertEqual(self.verify()["status"],"mismatch"); self.plan=saved

    def test_full_low_level_creation_properties_are_checked(self):
        with h5py.File(self.source,"w") as src:
            dcpl=h5py.h5p.create(h5py.h5p.DATASET_CREATE); dcpl.set_obj_track_times(False); dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
            ds=h5py.Dataset(h5py.h5d.create(src.id,b"A",h5py.h5t.STD_I32LE,h5py.h5s.create_simple((6,)),dcpl=dcpl)); ds[:]=np.arange(6,dtype="i4")
        paths=["/","/A"]; boxes={"/A":box("",[1],[4])["selection"]}; req=request([box("/A",[1],[4])])
        manual_copy(self.source,self.output,paths,boxes); self.plan=manual_plan(self.source,self.output,req,paths,boxes,self.limits)
        self.assertEqual(self.verify()["status"],"verified")
        with h5py.File(self.output,"w") as dst: dst.create_dataset("A",data=np.arange(1,4,dtype="i4"))
        self.plan=manual_plan(self.source,self.output,req,paths,boxes,self.limits); result=self.verify()
        self.assertEqual(result["status"],"mismatch",result); self.assertTrue(any("creation" in d["message"] for d in result["diagnostics"]),result)

    def test_mapping_paths_must_be_proven_hard_links(self):
        self.fixture()
        with h5py.File(self.source,"r+") as src: src["time_soft"]=h5py.SoftLink("/cal/time")
        self.plan["source"]=fingerprint(self.source)
        self.plan["request"]["scale_mappings"][0]["scale"]="/time_soft"
        result=self.verify()
        self.assertEqual(result["status"],"incomplete",result)
        self.assertEqual(result["diagnostics"][0]["code"],"UNSUPPORTED",result)

    def test_typed_opens_disable_aggregate_raw_chunk_cache(self):
        from unittest.mock import patch
        self.fixture(); module=self.verifier(); openings=[]
        original=h5py.File
        def record(*args,**kwargs):
            openings.append(kwargs.get("rdcc_nbytes")); return original(*args,**kwargs)
        with patch.object(module.h5py,"File",side_effect=record): result=self.verify()
        self.assertEqual(result["status"],"verified",result)
        self.assertEqual(openings,[0,0])

    def test_whole_nonpositional_scales_and_full_box_normalization(self):
        with h5py.File(self.source,"w") as src:
            ds=src.create_dataset("A",data=np.arange(6,dtype="i4"))
            scale=src.create_dataset("S",data=np.arange(3,dtype="i4")); scale.make_scale("not positional"); ds.dims[0].attach_scale(scale)
        paths=["/","/A","/S"]; manual_copy(self.source,self.output,paths,{})
        self.plan=manual_plan(self.source,self.output,request([box("/A",[0],[6])]),paths,{},self.limits)
        self.assertEqual(self.verify()["status"],"verified")
        self.plan["request"]["scale_mappings"]=[mapping("/A",0,"/S")]
        self.assertNotEqual(self.verify()["status"],"verified")

    def test_request_refusals_do_not_believe_valid_output(self):
        self.fixture(); baseline=copy.deepcopy(self.plan)
        variants=[]
        req=copy.deepcopy(self.req); req["scale_mappings"].pop(); variants.append(req)
        req=copy.deepcopy(self.req); req["objects"][2]=box("/run/B",[2,1],[5,3]); variants.append(req)
        req=copy.deepcopy(self.req); req["objects"].append(whole("/run")); req["objects"].sort(key=lambda x:x["path"]); variants.append(req)
        req=copy.deepcopy(self.req); req["objects"].append(whole("/cal/time")); req["objects"].sort(key=lambda x:x["path"]); variants.append(req)
        req=copy.deepcopy(self.req); req["objects"][1]["selection"]["start"][0]=True; variants.append(req)
        req=copy.deepcopy(self.req); req["objects"].append(copy.deepcopy(req["objects"][0])); variants.append(req)
        req=copy.deepcopy(self.req); req["scale_mappings"].append(mapping("/run/A_alias",0,"/cal/time")); req["scale_mappings"].sort(key=lambda x:(x["consumer"],x["axis"],x["scale"])); variants.append(req)
        req=copy.deepcopy(self.req); req["scale_mappings"].append(mapping("/C",0,"/cal/time")); req["scale_mappings"].sort(key=lambda x:(x["consumer"],x["axis"],x["scale"])); variants.append(req)
        req=copy.deepcopy(self.req); req["objects"][0]["selection"]={"kind":"box","start":[0],"stop":[3]}; variants.append(req)
        for index,req in enumerate(variants):
            with self.subTest(index=index):
                self.plan=copy.deepcopy(baseline); self.plan["request"]=req
                self.assertNotEqual(self.verify()["status"],"verified")

    def test_root_and_ancestor_reference_requirements_conflict_with_crop(self):
        self.fixture()
        for owner,target in (("/","/run/A"),("/run","/run")):
            with self.subTest(owner=owner,target=target):
                with h5py.File(self.source,"r+") as src: src[owner].attrs["requires_whole"]=src[target].ref
                self.plan["source"]=fingerprint(self.source)
                result=self.verify(); self.assertEqual(result["status"],"incomplete",result)
                self.assertEqual(result["diagnostics"][0]["code"],"INVALID",result)
                with h5py.File(self.source,"r+") as src: del src[owner].attrs["requires_whole"]

    def test_coordinated_metadata_reference_scale_and_alias_mutations(self):
        for change in ("root_ref","ancestor_ref","ref_value","scale","alias_split","extra"):
            with self.subTest(change=change):
                self.fixture()
                with h5py.File(self.output,"r+") as dst:
                    if change=="root_ref": del dst.attrs["audit"]
                    if change=="ancestor_ref": del dst["run"].attrs["configuration"]
                    if change=="ref_value": dst["refs"][0]=h5py.Reference()
                    if change=="scale": dst["run/B"].dims[1].detach_scale(dst["cal/channel"])
                    if change=="alias_split":
                        del dst["run/A_alias"]; dst.copy("run/A","run/A_alias")
                    if change=="extra": dst.create_dataset("extra",data=np.array([2],dtype="i4")); self.paths.append("/extra")
                if change=="extra":
                    # A new output identity cannot have a source descriptor.
                    with h5py.File(self.output,"r") as dst: self.plan["graph"]=manual_graph(dst,self.paths,self.limits)
                else: self.plan=manual_plan(self.source,self.output,self.req,self.paths,self.boxes,self.limits)
                self.assertNotEqual(self.verify()["status"],"verified")

    def test_empty_boxes_keep_rank_and_do_not_visit_oversized_chunks(self):
        for shape,start,stop,chunks in (((6,4),[2,1],[2,3],(6,4)),((0,4),[0,1],[0,3],None)):
            with self.subTest(shape=shape):
                with h5py.File(self.source,"w") as src: src.create_dataset("A",shape=shape,dtype="i4",chunks=chunks)
                boxes={"/A":box("",start,stop)["selection"]}; paths=["/","/A"]
                manual_copy(self.source,self.output,paths,boxes)
                self.plan=manual_plan(self.source,self.output,request([box("/A",start,stop)]),paths,boxes,self.limits)
                result=self.verify(); self.assertEqual(result["status"],"verified",result)
                self.assertEqual(result["coverage"]["data_blocks"],0)
                with h5py.File(self.output,"r") as dst: self.assertEqual(dst["A"].shape,(0,2))

    def test_exact_signed_zero_nan_endian_and_complex_bytes(self):
        cases=(np.array([0,0x8000000000000000,0x7ff8000000000042,0x7ff8000000000055],dtype="<u8").view("<f8"),
               np.array([complex(-0.,2),1+3j,-4-5j,6+7j],dtype=">c16"))
        for values in cases:
            with self.subTest(dtype=str(values.dtype)):
                with h5py.File(self.source,"w") as src: src.create_dataset("A",data=values)
                paths=["/","/A"]; boxes={"/A":box("",[1],[3])["selection"]}
                manual_copy(self.source,self.output,paths,boxes)
                self.plan=manual_plan(self.source,self.output,request([box("/A",[1],[3])]),paths,boxes,self.limits)
                self.assertEqual(self.verify()["status"],"verified")
                with h5py.File(self.output,"r+") as dst: dst["A"][0]=0
                self.plan=manual_plan(self.source,self.output,self.plan["request"],paths,boxes,self.limits)
                self.assertEqual(self.verify()["status"],"mismatch")

    def test_offset_wide_reads_and_rank_limit_use_bounded_native_segments(self):
        from unittest.mock import patch
        for shape,start,stop,chunks in (((4,100),[1,13],[3,90],(2,4)),((1,)*31+(20,),[0]*31+[3],[1]*31+[17],None)):
            with self.subTest(shape=shape):
                with h5py.File(self.source,"w") as src: src.create_dataset("A",data=np.arange(math.prod(shape),dtype="i4").reshape(shape),chunks=chunks)
                paths=["/","/A"]; boxes={"/A":box("",start,stop)["selection"]}
                manual_copy(self.source,self.output,paths,boxes); self.plan=manual_plan(self.source,self.output,request([box("/A",start,stop)]),paths,boxes,self.limits)
                module=self.verifier(); original=h5py.Dataset.__getitem__; reads=[]
                def capture(dataset,index,*args,**kwargs):
                    value=original(dataset,index,*args,**kwargs); reads.append(value.nbytes); return value
                with patch.object(h5py.Dataset,"__getitem__",capture): result=self.verify()
                self.assertEqual(result["status"],"verified",result); self.assertTrue(reads); self.assertLessEqual(max(reads),32)
                self.assertLessEqual(result["coverage"]["max_native_expansion_bytes"],32)
        iterator=self.verifier()._tiles((1<<45,3,8),4,32,origin=(9,1,0))
        self.assertEqual(next(iterator),((9,1,slice(0,8)),(0,0,slice(0,8))))
        self.assertEqual(next(iterator),((9,2,slice(0,8)),(0,1,slice(0,8))))

    def test_raw_chunk_and_reference_inventory_guard_precede_native_reads(self):
        from unittest.mock import patch
        self.fixture()
        for refs in (False,True):
            with self.subTest(reference=refs):
                with h5py.File(self.source,"r+") as src:
                    if "oversized" in src: del src["oversized"]
                    dtype=h5py.ref_dtype if refs else np.dtype("i4")
                    src.create_dataset("oversized",shape=(100,),dtype=dtype,chunks=(100,))
                self.plan["source"]=fingerprint(self.source)
                if not refs: self.plan["request"]["objects"].insert(0,box("/oversized",[0],[1]))
                original=h5py.Dataset.__getitem__; visited=[]
                def capture(dataset,index,*args,**kwargs):
                    if dataset.name=="/oversized": visited.append(dataset.name)
                    return original(dataset,index,*args,**kwargs)
                with patch.object(h5py.Dataset,"__getitem__",capture): result=self.verify()
                self.assertEqual(result["status"],"incomplete",result); self.assertEqual(result["diagnostics"][0]["code"],"RESOURCE",result)
                self.assertEqual(visited,[])
                if not refs: self.plan["request"]["objects"].pop(0)

    def test_nonempty_fill_never_refuses_before_any_payload_read(self):
        from unittest.mock import patch
        self.fixture()
        with h5py.File(self.source,"r+") as src:
            dcpl=h5py.h5p.create(h5py.h5p.DATASET_CREATE); dcpl.set_fill_time(h5py.h5d.FILL_TIME_NEVER)
            h5py.h5d.create(src.id,b"never",h5py.h5t.STD_I32LE,h5py.h5s.create_simple((6,)),dcpl=dcpl)
        self.plan["source"]=fingerprint(self.source); self.plan["request"]["objects"].insert(0,box("/never",[1],[3]))
        original=h5py.Dataset.__getitem__; visited=[]
        def capture(dataset,index,*args,**kwargs):
            if dataset.name=="/never": visited.append(dataset.name)
            return original(dataset,index,*args,**kwargs)
        with patch.object(h5py.Dataset,"__getitem__",capture): result=self.verify()
        self.assertEqual(result["status"],"incomplete",result); self.assertEqual(result["diagnostics"][0]["code"],"UNSUPPORTED",result); self.assertEqual(visited,[])

    def test_scalar_null_reference_and_zero_extent_whole_roundtrip(self):
        with h5py.File(self.source,"w") as src:
            src.create_dataset("scalar",data=np.array(-0.,dtype=">f8"))
            src.create_dataset("null",shape=None,dtype="i4")
            src.create_dataset("zero",shape=(0,3),dtype="u1")
            src.create_dataset("refs",shape=(2,),dtype=h5py.ref_dtype)[:]=[src["scalar"].ref,h5py.Reference()]
            src.create_dataset("ref_scalar",shape=(),dtype=h5py.ref_dtype)[()]=h5py.Reference()
        paths=["/","/scalar","/null","/zero","/refs","/ref_scalar"]
        manual_copy(self.source,self.output,paths,{})
        self.plan=manual_plan(self.source,self.output,request([whole("/")]),paths,{},self.limits)
        self.assertEqual(self.verify()["status"],"verified")
        for path,start,stop in (("/scalar",[],[]),("/null",[0],[0]),("/refs",[0],[2])):
            with self.subTest(path=path):
                self.plan["request"]=request([box(path,start,stop)])
                self.assertNotEqual(self.verify()["status"],"verified")

    def test_soft_group_expansion_and_soft_box_refusal(self):
        with h5py.File(self.source,"w") as src:
            src.create_dataset("group/A",data=np.arange(6,dtype="i4"))
            src.create_dataset("group/B",data=np.arange(4,dtype="i4"))
            src["soft"]=h5py.SoftLink("/group")
        paths=["/","/group","/group/A","/group/B","/soft"]
        manual_copy(self.source,self.output,paths,{})
        self.plan=manual_plan(self.source,self.output,request([whole("/soft/A")]),paths,{},self.limits)
        self.assertEqual(self.verify()["status"],"verified")
        self.plan["request"]=request([box("/soft/A",[0],[6])])
        self.assertEqual(self.verify()["status"],"incomplete")

    def test_nondefault_chunk_flags_cannot_be_lost_by_crop_or_whole_copy(self):
        import ctypes
        library=ctypes.CDLL(h5py.h5p.__file__); set_options=library.H5Pset_chunk_opts
        set_options.argtypes=(ctypes.c_int64,ctypes.c_uint); set_options.restype=ctypes.c_int
        with h5py.File(self.source,"w") as src:
            dcpl=h5py.h5p.create(h5py.h5p.DATASET_CREATE); dcpl.set_chunk((4,)); dcpl.set_deflate(1)
            with h5py._objects.phil: self.assertEqual(set_options(dcpl.id,2),0)
            ds=h5py.Dataset(h5py.h5d.create(src.id,b"A",h5py.h5t.STD_I32LE,h5py.h5s.create_simple((6,)),dcpl=dcpl)); ds[:]=np.arange(6,dtype="i4")
        paths=["/","/A"]; manual_copy(self.source,self.output,paths,{})
        self.plan=manual_plan(self.source,self.output,request([whole("/A")]),paths,{},self.limits)
        self.assertEqual(self.verify()["status"],"verified")
        with h5py.File(self.output,"w") as dst: dst.create_dataset("A",data=np.arange(6,dtype="i4"),chunks=(4,),compression="gzip",compression_opts=1)
        self.plan=manual_plan(self.source,self.output,request([whole("/A")]),paths,{},self.limits)
        self.assertEqual(self.verify()["status"],"mismatch")
        boxes={"/A":box("",[1],[3])["selection"]}; manual_copy(self.source,self.output,paths,boxes)
        self.plan=manual_plan(self.source,self.output,request([box("/A",[1],[3])]),paths,boxes,self.limits)
        result=self.verify(); self.assertEqual(result["status"],"incomplete",result)
        self.assertEqual(result["diagnostics"][0]["code"],"UNSUPPORTED",result)

    def test_shared_scale_conflict_identifies_consumer_axis_and_interval(self):
        self.fixture()
        self.plan["request"]["objects"][-1]=box("/run/B",[2,1],[5,3])
        result=self.verify(); self.assertEqual(result["status"],"incomplete",result)
        message=result["diagnostics"][0]["message"]
        self.assertIn("/run/B",message); self.assertIn("axis 0",message); self.assertIn("[2]",message)

    def test_source_changed_before_and_during_verification_never_claims_equality(self):
        from unittest.mock import patch
        self.fixture()
        with h5py.File(self.source,"r+") as src: src["run/A"][0,0]=99
        result=self.verify(); self.assertEqual(result["diagnostics"][0]["code"],"SOURCE_CHANGED")
        self.plan["source"]=fingerprint(self.source); original=self.verifier().fingerprint; calls=[]
        def mutate_fingerprint(path):
            value=original(path); calls.append(path)
            if len(calls)==2: value["sha256"]="0"*64
            return value
        with patch.object(self.verifier(),"fingerprint",side_effect=mutate_fingerprint): result=self.verify()
        self.assertEqual(result["status"],"incomplete",result); self.assertFalse(result["coverage"]["source_equality"])
        self.assertEqual(result["diagnostics"][0]["code"],"SOURCE_CHANGED")

    def test_large_sparse_source_does_not_hash_unselected_extent(self):
        with h5py.File(self.source,"w") as src:
            src.create_dataset("A",shape=(1<<45,3),dtype="i4",fillvalue=7)
            src.create_dataset("unselected",shape=(1<<48,),dtype="u1")
        paths=["/","/A"]; boxes={"/A":box("",[(1<<45)-2,1],[(1<<45)-1,3])["selection"]}
        manual_copy(self.source,self.output,paths,boxes)
        self.plan=manual_plan(self.source,self.output,request([box("/A",[(1<<45)-2,1],[(1<<45)-1,3])]),paths,boxes,self.limits)
        result=self.verify(); self.assertEqual(result["status"],"verified",result)
        self.assertEqual(result["coverage"]["payload_bytes"],8)

    def test_repeated_scale_attachment_axes_and_multiple_scales(self):
        with h5py.File(self.source,"w") as src:
            a=src.create_dataset("A",data=np.arange(36,dtype="i4").reshape(6,6))
            s=src.create_dataset("S",data=np.arange(6,dtype="i4")); s.make_scale("same")
            t=src.create_dataset("T",data=np.arange(6,dtype="i4")); t.make_scale("same")
            a.dims[0].attach_scale(s); a.dims[1].attach_scale(s); a.dims[1].attach_scale(t)
        paths=["/","/A","/S","/T"]; boxes={"/A":box("",[1,1],[4,4])["selection"],"/S":box("",[1],[4])["selection"],"/T":box("",[1],[4])["selection"]}
        mappings=[mapping("/A",0,"/S"),mapping("/A",1,"/S"),mapping("/A",1,"/T")]
        manual_copy(self.source,self.output,paths,boxes)
        self.plan=manual_plan(self.source,self.output,request([box("/A",[1,1],[4,4])],mappings),paths,boxes,self.limits)
        self.assertEqual(self.verify()["status"],"verified")
        self.plan["request"]=request([box("/A",[1,2],[4,5])],mappings)
        result=self.verify(); self.assertEqual(result["diagnostics"][0]["code"],"INVALID",result)
        self.assertIn("axis 1",result["diagnostics"][0]["message"])
        self.plan["request"]=request([box("/A",[1,1],[4,4])],mappings[:-1])
        self.assertEqual(self.verify()["status"],"incomplete")

    def test_crop_requires_mapping_even_when_attached_axis_is_whole(self):
        with h5py.File(self.source,"w") as src:
            a=src.create_dataset("A",data=np.arange(24,dtype="i4").reshape(6,4))
            s=src.create_dataset("S",data=np.arange(6,dtype="i4")); s.make_scale("rows"); a.dims[0].attach_scale(s)
        paths=["/","/A","/S"]; boxes={"/A":box("",[0,1],[6,3])["selection"]}
        manual_copy(self.source,self.output,paths,boxes)
        self.plan=manual_plan(self.source,self.output,request([box("/A",[0,1],[6,3])],[mapping("/A",0,"/S")]),paths,boxes,self.limits)
        self.assertEqual(self.verify()["status"],"verified")
        self.plan["request"]["scale_mappings"]=[]
        self.assertEqual(self.verify()["status"],"incomplete")

    def test_unselected_reverse_consumer_is_absent_from_required_graph(self):
        self.fixture()
        with h5py.File(self.output,"r") as dst:
            self.assertNotIn("C",dst)
            consumers={dst[row["dataset"]].name for row in dst["cal/time"].attrs["REFERENCE_LIST"]}
            self.assertEqual(consumers,{"/run/A","/run/B"})
        self.assertEqual(self.verify()["status"],"verified")

    def test_reference_inventory_tiles_stop_at_each_native_chunk_boundary(self):
        from unittest.mock import patch
        with h5py.File(self.source,"w") as src:
            src.create_dataset("selected",data=np.array([3],dtype="i4"))
            refs=src.create_dataset("unselected_refs",shape=(11,),dtype=h5py.ref_dtype,chunks=(3,))
            refs[:]=[h5py.Reference()]*11
        paths=["/","/selected"]; manual_copy(self.source,self.output,paths,{})
        self.plan=manual_plan(self.source,self.output,request([whole("/selected")]),paths,{},self.limits)
        original=h5py.Dataset.__getitem__; selections=[]
        def capture(dataset,index,*args,**kwargs):
            if dataset.name=="/unselected_refs": selections.append(index[0])
            return original(dataset,index,*args,**kwargs)
        with patch.object(h5py.Dataset,"__getitem__",capture): result=self.verify()
        self.assertEqual(result["status"],"verified",result)
        self.assertEqual(selections,[slice(0,3),slice(3,6),slice(6,9),slice(9,11)])
        self.assertEqual(result["coverage"]["max_native_expansion_bytes"],24)

if __name__=="__main__": unittest.main()
