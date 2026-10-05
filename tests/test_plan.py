import copy
import json
import tempfile
import unittest
from pathlib import Path


def empty_plan():
    from h5carry.model import Limits
    return {'format':'h5carry-plan','version':1,'source':{'sha256':'0'*64,'size':0},
            'selections':['/'],'limits':Limits().to_dict(),
            'graph':{'objects':[{'id':'/','kind':'group','metadata':{'attributes':[]}}],
                     'links':[],'references':[],'scales':[],'payload_bytes':0,'reasons':{'/':['selected']}}}


class PlanTests(unittest.TestCase):
    def test_roundtrip_and_no_clobber(self):
        from h5carry.plan import load_plan, save_plan
        from h5carry.model import CarryError
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'plan.json'
            save_plan(empty_plan(), path)
            self.assertEqual(load_plan(path), empty_plan())
            with self.assertRaises(CarryError): save_plan(empty_plan(), path)

    def test_null_fixed_attribute_is_preserved(self):
        from h5carry.plan import validate_plan
        p=empty_plan()
        p['graph']['objects'][0]['metadata']['attributes']=[{'name':'null','dtype':{'kind':'fixed','str':'<f4','encoding':None},'shape':None,'value':None}]
        self.assertEqual(validate_plan(p),p)

    def test_strict_fields_and_types(self):
        from h5carry.plan import validate_plan
        from h5carry.model import CarryError
        variants=[]
        p=empty_plan(); p['version']=True; variants.append(p)
        p=empty_plan(); p['unknown']=1; variants.append(p)
        p=empty_plan(); p['source']['path']='private'; variants.append(p)
        p=empty_plan(); p['selections']=['/x','/x']; variants.append(p)
        p=empty_plan(); p['graph']['objects'][0]['metadata']['unknown']=1; variants.append(p)
        p=empty_plan(); p['graph']['payload_bytes']=True; variants.append(p)
        p=empty_plan(); p['graph']['links']=[{'path':'/x','kind':'hard','target':'/missing'}]; variants.append(p)
        p=empty_plan(); p['graph']['references']=[{'owner':'/','attribute':'x','index':0,'target':'/missing'}]; variants.append(p)
        for p in variants:
            with self.subTest(p=p), self.assertRaises(CarryError): validate_plan(p)

    def test_duplicate_keys_oversize_deep_and_nonfinite_json(self):
        from h5carry.plan import load_plan
        from h5carry.model import CarryError, Limits
        payloads=['{"format":"h5carry-plan","format":"h5carry-plan"}', '['*100+']'*100,
                  '{"a":NaN}', '{"a":'+ '9'*1000 +'}', json.dumps(empty_plan())+' ' * 4096]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'bad.json'
            for payload in payloads:
                path.write_text(payload)
                with self.subTest(payload=payload[:30]), self.assertRaises(CarryError):
                    load_plan(path, Limits(max_plan_bytes=2048))

    def test_json_container_budget_before_decode(self):
        from h5carry.plan import load_plan
        from h5carry.model import CarryError
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'containers.json'; p.write_bytes(b'['+b'[],'*600000+b'[]]')
            with patch('h5carry.plan.json.loads') as decoder:
                with self.assertRaises(CarryError): load_plan(p)
                self.assertEqual(decoder.call_count,0)

    def test_exact_byte_ceiling_roundtrip(self):
        from h5carry.plan import load_plan, save_plan
        from h5carry.model import Limits, canonical_json
        p=empty_plan()
        for _ in range(4): p['limits']['max_plan_bytes']=len(canonical_json(p))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'plan'; save_plan(p,path)
            self.assertEqual(load_plan(path,Limits(max_plan_bytes=p['limits']['max_plan_bytes'])),p)

    def test_edge_id_type_errors_are_invalid(self):
        from h5carry.plan import validate_plan
        from h5carry.model import CarryError
        variants=[]
        p=empty_plan(); p['graph']['links']=[{'path':'/x','kind':'hard','target':[]}]; variants.append(p)
        p=empty_plan(); p['graph']['references']=[{'owner':[],'attribute':'x','index':0,'target':'/'}]; variants.append(p)
        p=empty_plan(); p['graph']['references']=[{'owner':'/','attribute':'x','index':0,'target':[]}]; variants.append(p)
        p=empty_plan(); p['graph']['scales']=[{'consumer':[],'axis':0,'scale':'/'}]; variants.append(p)
        for p in variants:
            with self.subTest(p=p):
                with self.assertRaises(CarryError) as found: validate_plan(p)
                self.assertEqual(found.exception.code,'INVALID')

    def test_descriptor_payload_shapes_and_dtypes_are_strict(self):
        from h5carry.plan import validate_plan
        from h5carry.model import CarryError
        base={'name':'x','dtype':{'kind':'text','str':'|O','encoding':'utf-8'},'shape':[],'value':['78']}
        variants=[]
        for replacement in ({'value':{'hidden':1}}, {'value':['zz']}, {'value':[]}, {'shape':None,'value':['78']}, {'dtype':{'kind':'fixed','str':'madeup','encoding':None}}, {'dtype':{'kind':'fixed','str':'<i4','encoding':None},'shape':[2],'value':'00'}):
            p=empty_plan(); attr=copy.deepcopy(base); attr.update(replacement)
            p['graph']['objects'][0]['metadata']['attributes']=[attr]; variants.append(p)
        for p in variants:
            with self.subTest(p=p),self.assertRaises(CarryError): validate_plan(p)

    def test_declared_payload_equals_object_shapes_before_native(self):
        from h5carry.plan import validate_plan
        from h5carry.model import CarryError
        p=empty_plan()
        p['graph']['objects'].append({'id':'/data','kind':'dataset','metadata':{
            'attributes':[],'dtype':{'kind':'fixed','str':'|u1','encoding':None},'shape':[1024],
            'creation':{'layout':'contiguous','chunks':None,'maxshape':[1024],'filters':[],'fill':'00'},
            'labels':[''],'scale_name':None,'payload_sha256':'0'*64}})
        p['graph']['links']=[{'path':'/data','kind':'hard','target':'/data'}]
        p['graph']['reasons']['/data']=['selected child']
        p['limits']['max_payload_bytes']=1
        p['graph']['payload_bytes']=1
        with self.assertRaises(CarryError) as found: validate_plan(p)
        self.assertEqual(found.exception.code,'RESOURCE')
        p['limits']['max_payload_bytes']=2048
        with self.assertRaises(CarryError) as found: validate_plan(p)
        self.assertEqual(found.exception.code,'INVALID')
        p['graph']['payload_bytes']=1024
        self.assertEqual(validate_plan(p),p)

    def test_budget_and_namespace_checks(self):
        from h5carry.plan import validate_plan
        from h5carry.model import CarryError
        p=empty_plan(); p['limits']['max_objects']=1
        p['graph']['objects'].append({'id':'/x','kind':'group','metadata':{'attributes':[]}})
        p['graph']['links'].append({'path':'/x','kind':'hard','target':'/x'})
        p['graph']['reasons']['/x']=['child']
        with self.assertRaises(CarryError): validate_plan(p)
        p=empty_plan(); p['graph']['links']=[{'path':'/x/y','kind':'soft','target':'/'}]
        with self.assertRaises(CarryError): validate_plan(p)
