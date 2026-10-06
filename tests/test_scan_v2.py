"""Original rectangular graph cases, independent literal expected results."""
import copy
import importlib.util
import unittest
from tests.test_profile import NativeFixtureMixin


def request(*objects, mappings=()):
    return {'format':'h5carry-selection','version':1,'objects':list(objects),
            'scale_mappings':list(mappings),'storage_policy':'fixed-snapshot-v1'}

def box(path, start=(1,1), stop=(4,3)):
    return {'path':path,'selection':{'kind':'box','start':list(start),'stop':list(stop)}}

def whole(path):
    return {'path':path,'selection':{'kind':'whole'}}

def mapping(consumer, axis, scale):
    return {'consumer':consumer,'axis':axis,'scale':scale,'mapping':'index'}

class SliceScanTests(NativeFixtureMixin, unittest.TestCase):
    def build(self):
        n=self.np; f=self.f
        a=f.create_dataset('A',data=n.arange(6)[:,None]*100+n.arange(4)[None,:],chunks=(3,4),compression='gzip',maxshape=(None,4))
        b=f.create_dataset('B',data=1000+n.arange(6)[:,None]*100+n.arange(4)[None,:]); f['alias']=a
        t=f.create_dataset('time',data=n.arange(6)*10);t.make_scale('coordinate')
        c=f.create_dataset('channel',data=[400,500,600,700]);c.make_scale('coordinate')
        for d in (a,b):d.dims[0].attach_scale(t);d.dims[1].attach_scale(c)
        excluded=f.create_dataset('C',data=n.zeros((6,4),dtype='i8')); excluded.dims[0].attach_scale(t)
        return [mapping(p,axis,s) for p in ('/A','/B') for axis,s in enumerate(('/time','/channel'))]
    def plan(self,r,limits=None):
        self.assertIsNotNone(importlib.util.find_spec('h5carry.scan_v2'),'typed planner missing')
        from h5carry.scan_v2 import make_plan_native
        source=self.f.filename;self.f.flush()
        return make_plan_native(source,r,limits or self.limits)
    def test_shared_scales_alias_and_literal_crop(self):
        m=self.build(); result=self.plan(request(box('/A'),box('/B'),box('/alias'),mappings=m))
        g=result['graph']; objects={o['id']:o for o in g['objects']}
        self.assertEqual(set(objects),{'/','/A','/B','/time','/channel'})
        self.assertEqual(objects['/A']['metadata']['shape'],[3,2])
        self.assertEqual(objects['/A']['metadata']['creation']['chunks'],[3,2])
        self.assertEqual(objects['/A']['metadata']['creation']['maxshape'],[3,2])
        self.assertEqual(objects['/time']['metadata']['shape'],[3]);self.assertEqual(objects['/channel']['metadata']['shape'],[2])
        self.assertEqual({x['id']:x['metadata']['shape'] for x in result['source_objects'] if x['kind']=='dataset'}['/A'],[6,4])
        self.assertIn({'path':'/alias','kind':'hard','target':'/A'},g['links'])
        import hashlib
        self.assertEqual(objects['/A']['metadata']['payload_sha256'],hashlib.sha256(self.np.array([[101,102],[201,202],[301,302]],dtype='i8').tobytes()).hexdigest())
        self.assertEqual(g['payload_bytes'],3*2*8*2+3*8+2*8)
    def test_empty_box_keeps_rank_and_only_reads_nonempty_scale(self):
        m=self.build()[:2];r=request(box('/A',(2,1),(2,3)),mappings=m)
        reads=[];get=self.h5.Dataset.__getitem__
        def observe(d,s):
            reads.append(d.name);return get(d,s)
        from unittest.mock import patch
        with patch.object(self.h5.Dataset,'__getitem__',observe): result=self.plan(r)
        self.assertNotIn('/A',reads);self.assertNotIn('/time',reads);self.assertIn('/channel',reads)
        self.assertEqual(next(o for o in result['graph']['objects'] if o['id']=='/A')['metadata']['shape'],[0,2])
    def test_missing_extra_duplicate_alias_mappings_refuse(self):
        m=self.build()
        for r in (request(box('/A')), request(box('/A'),mappings=m), request(box('/A'),box('/alias'),mappings=m[:2]+[mapping('/alias',0,'/time')])):
            with self.subTest(r=r),self.assertRaises(self.error):self.plan(r)
    def test_conflicting_alias_or_shared_intervals_refuse(self):
        m=self.build()
        for r in (request(box('/A'),whole('/alias'),mappings=m[:2]),request(box('/A'),box('/B',(2,1),(5,3)),mappings=m),request(box('/A'),whole('/time'),mappings=m[:2])):
            with self.subTest(r=r),self.assertRaises(self.error) as cm:self.plan(r)
            self.assertIn('conflict',cm.exception.message.lower())
    def test_root_reference_and_referenced_group_force_whole(self):
        m=self.build()[:2]; self.f.attrs['ref']=self.f['A'].ref
        with self.assertRaises(self.error):self.plan(request(box('/A'),mappings=m))
        del self.f.attrs['ref'];g=self.f.create_group('g'); d=g.create_dataset('D',data=[1,2,3]); self.f.attrs['ref']=g.ref
        with self.assertRaises(self.error):self.plan(request(box('/g/D',(0,),(1,))))
    def test_outward_whole_refs_nulls_and_ancestor_references(self):
        m=self.build()[:2];g=self.f.create_group('selected');gain=self.f.create_dataset('gain',data=[9,8]);g.attrs['gain']=gain.ref;self.f['A'].attrs['gain']=gain.ref
        refs=g.create_dataset('refs',(2,),dtype=self.h5.ref_dtype);refs[:]=[gain.ref,self.h5.Reference()]
        result=self.plan(request(box('/A'),whole('/selected/refs'),mappings=m));self.assertIn('/gain',{o['id'] for o in result['graph']['objects']});self.assertEqual(len(result['graph']['references']),4)
    def test_full_shape_normalizes_whole_without_mapping(self):
        self.build();r=self.plan(request(box('/A',(0,0),(6,4))))
        self.assertFalse(r['transformations']);self.assertTrue(all(x['selection']=={'kind':'whole'} for x in r['selections']))
        self.assertEqual(next(o for o in r['graph']['objects'] if o['id']=='/A')['metadata']['creation']['maxshape'],[None,4])
    def test_soft_crop_and_scalar_null_reference_boxes_refuse(self):
        self.f.create_dataset('data',data=[1,2]); self.f['soft']=self.h5.SoftLink('/data')
        self.f.create_dataset('scalar',data=1);self.f.create_dataset('null',shape=None,dtype='i4');self.f.create_dataset('refs',(2,),dtype=self.h5.ref_dtype)
        for p,start,stop in [('/soft',[0],[1]),('/scalar',[],[]),('/null',[],[]),('/refs',[0],[1])]:
            with self.subTest(path=p),self.assertRaises(self.error):self.plan(request(box(p,start,stop)))
    def test_whole_nonpositional_scale_kept_box_refused(self):
        d=self.f.create_dataset('D',data=self.np.zeros((6,4)));s=self.f.create_dataset('S',data=[1,2]);s.make_scale('x');d.dims[0].attach_scale(s)
        result=self.plan(request(whole('/D')));self.assertEqual(len(result['graph']['scales']),1)
        with self.assertRaises(self.error):self.plan(request(box('/D'),mappings=[mapping('/D',0,'/S')]))
    def test_chunk_guard_before_read_and_large_sparse_small_crop(self):
        from dataclasses import replace
        d=self.f.create_dataset('D',shape=(1000,1000),dtype='i4',chunks=(100,100),compression='gzip')
        r=request(box('/D',(0,0),(1,1)))
        with self.assertRaises(self.error) as cm:self.plan(r,replace(self.limits,chunk_bytes=1024))
        self.assertEqual(cm.exception.code,'RESOURCE')
        result=self.plan(r);self.assertEqual(result['graph']['payload_bytes'],4)
