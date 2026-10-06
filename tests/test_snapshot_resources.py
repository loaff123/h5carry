"""Benign resource/accounting controls; no remote or hostile inputs."""
from dataclasses import replace
import hashlib
import unittest
from unittest.mock import patch
from tests.test_profile import NativeFixtureMixin
from tests.test_scan_v2 import request,box,whole

class SnapshotResourceTests(NativeFixtureMixin,unittest.TestCase):
    def plan(self,r,limits=None):
        from h5carry.scan_v2 import make_plan_native
        self.f.flush();return make_plan_native(self.f.filename,r,limits or self.limits)
    def test_offset_crossing_chunks_actual_read_and_expansion_caps(self):
        a=self.np.arange(7*19,dtype='i4').reshape(7,19)
        d=self.f.create_dataset('A',data=a,chunks=(2,4),compression='gzip')
        expected=a[1:6,3:17];seen=[];original=self.h5.Dataset.__getitem__
        def read(dataset,selection):
            if dataset.name=='/A':
                logical=__import__('math').prod(s.stop-s.start for s in selection)*dataset.dtype.itemsize
                chunks=__import__('math').prod((s.stop-1)//c-s.start//c+1 for s,c in zip(selection,dataset.chunks))
                seen.append((logical,chunks*__import__('math').prod(dataset.chunks)*dataset.dtype.itemsize,selection))
            return original(dataset,selection)
        with patch.object(self.h5.Dataset,'__getitem__',read):
            p=self.plan(request(box('/A',(1,3),(6,17))),replace(self.limits,chunk_bytes=32))
        self.assertTrue(seen);self.assertTrue(all(a<=32 and b<=32 for a,b,_ in seen))
        self.assertTrue(all(1<=s[0].start<s[0].stop<=6 and 3<=s[1].start<s[1].stop<=17 for _,_,s in seen))
        self.assertEqual(p['graph']['objects'][1]['metadata']['payload_sha256'],hashlib.sha256(expected.tobytes()).hexdigest())
    def test_nonempty_never_refused_before_payload_read_empty_admitted(self):
        p=self.h5.h5p.create(self.h5.h5p.DATASET_CREATE);p.set_fill_time(self.h5.h5d.FILL_TIME_NEVER)
        self.h5.h5d.create(self.f.id,b'A',self.h5.h5t.STD_I32LE,self.h5.h5s.create_simple((4,)),dcpl=p)
        with patch.object(self.h5.Dataset,'__getitem__',side_effect=AssertionError('undefined data must not be read')):
            with self.assertRaises(self.error) as cm:self.plan(request(box('/A',(0,),(1,))))
            self.assertEqual(cm.exception.code,'UNSUPPORTED')
            empty=self.plan(request(box('/A',(0,),(0,))))
        self.assertEqual(empty['graph']['payload_bytes'],0)
    def test_nondefault_chunk_options_crop_refused_whole_kept(self):
        import ctypes
        p=self.h5.h5p.create(self.h5.h5p.DATASET_CREATE);p.set_chunk((4,))
        lib=ctypes.CDLL(self.h5.h5p.__file__);fn=lib.H5Pset_chunk_opts;fn.argtypes=(ctypes.c_int64,ctypes.c_uint);fn.restype=ctypes.c_int
        with self.h5._objects.phil:self.assertEqual(fn(p.id,2),0)
        self.h5.h5d.create(self.f.id,b'A',self.h5.h5t.STD_I32LE,self.h5.h5s.create_simple((8,)),dcpl=p)
        with patch.object(self.h5.Dataset,'__getitem__',side_effect=AssertionError('crop must refuse before read')):
            with self.assertRaises(self.error) as cm:self.plan(request(box('/A',(0,),(1,))))
            self.assertEqual(cm.exception.code,'UNSUPPORTED')
        p=self.plan(request(whole('/A')));self.assertEqual(p['graph']['payload_bytes'],32)
    def test_huge_sparse_extent_tiny_crop_and_zero_rank32(self):
        self.f.create_dataset('wide',shape=(2**35,4),dtype='u1',chunks=(1,4))
        p=self.plan(request(box('/wide',(2**35-1,1),(2**35,3))))
        self.assertEqual(p['graph']['payload_bytes'],2)
        self.f.create_dataset('empty',shape=(0,)+(1,)*31,dtype='u1')
        p=self.plan(request(box('/empty',(0,)*32,(0,)+(1,)*31)))
        self.assertEqual(p['graph']['objects'][1]['metadata']['shape'],[0]+[1]*31)
    def test_unselected_reference_array_native_chunk_guard(self):
        self.f.create_dataset('A',data=[1])
        self.f.create_dataset('refs',(20000,),dtype=self.h5.ref_dtype,chunks=(20000,),compression='gzip')
        with patch.object(self.h5.Dataset,'__getitem__',side_effect=AssertionError('oversized ref chunk must refuse before read')):
            with self.assertRaises(self.error) as cm:self.plan(request(whole('/A')),replace(self.limits,chunk_bytes=32))
        self.assertEqual(cm.exception.code,'RESOURCE')
    def test_selected_payload_limit_refuses_before_any_numeric_read(self):
        self.f.create_dataset('A',shape=(32,32),dtype='i4')
        with patch.object(self.h5.Dataset,'__getitem__',side_effect=AssertionError('selected budget must precede reads')):
            with self.assertRaises(self.error) as cm:self.plan(request(box('/A',(0,0),(16,16))),replace(self.limits,max_payload_bytes=100))
        self.assertEqual(cm.exception.code,'RESOURCE')
