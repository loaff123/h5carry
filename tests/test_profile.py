"""Original benign profile fixtures; the native suite is subprocess bounded."""
import importlib.util
import os
import tempfile
import unittest


class NativeFixtureMixin:
    def setUp(self):
        if os.environ.get('H5CARRY_TEST_NATIVE_CHILD') != '1':
            raise RuntimeError('Native fixtures require python -m tests.native_runner unittest ...')
        self.assertIsNotNone(importlib.util.find_spec('h5carry.profile'), 'profile implementation is missing')
        import h5py
        import numpy as np
        from h5carry.model import CarryError, Limits
        from h5carry import profile
        self.h5, self.np, self.error, self.limits, self.p = h5py, np, CarryError, Limits(), profile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = h5py.File(self.tmp.name + '/original.h5', 'w')
        self.addCleanup(self.f.close)


class ProfileTests(NativeFixtureMixin, unittest.TestCase):
    def test_fixed_types_spaces_filters_and_exact_hash(self):
        for i, dtype in enumerate(['i1','u1','<i2','>i2','<u4','>u8','f4','>f8','c8','>c16','?','S7']):
            a = self.np.zeros((2,3), dtype=dtype)
            d = self.f.create_dataset(str(i), data=a, chunks=(1,3), compression='gzip', compression_opts=7, shuffle=True, fletcher32=True, maxshape=(None,3))
            m = self.p.describe_object(d, self.limits)
            self.assertEqual(m['dtype']['str'], a.dtype.str)
            self.assertEqual(m['creation']['chunks'], [1,3])
            self.assertEqual(m['creation']['maxshape'], [None,3])
            self.assertEqual([x['id'] for x in m['creation']['filters']], [2,1,3])
            self.assertEqual(len(m['payload_sha256']), 64)
        for name, shape in [('scalar',()),('empty',(0,)),('null',None)]:
            d=self.f.create_dataset(name,shape=shape,dtype='f8')
            self.assertEqual(self.p.describe_object(d,self.limits)['shape'], None if shape is None else list(shape))

    def test_noncanonical_low_level_types_rejected(self):
        for name, change in [('precision',lambda t:t.set_precision(23)),('offset',lambda t:t.set_offset(1)),('padding',lambda t:t.set_pad(self.h5.h5t.PAD_ONE,self.h5.h5t.PAD_ZERO))]:
            t=self.h5.h5t.STD_I32LE.copy();change(t)
            sid=self.h5.h5s.create_simple((2,))
            low=self.h5.h5d.create(self.f.id,name.encode(),t,sid)
            d=self.h5.Dataset(low)
            with self.assertRaises(self.error) as e:self.p.describe_object(d,self.limits)
            self.assertEqual(e.exception.code,'UNSUPPORTED')

    def test_unsupported_types_and_layouts(self):
        types=[self.h5.string_dtype(),self.h5.vlen_dtype(self.np.dtype('i4')),self.h5.enum_dtype({'X':1},basetype='i4'),self.np.dtype([('x','f4'),('y','f4')]),self.h5.regionref_dtype]
        for i,dtype in enumerate(types):
            d=self.f.create_dataset(str(i),(1,),dtype=dtype)
            with self.assertRaises(self.error):self.p.describe_object(d,self.limits)
        dcpl=self.h5.h5p.create(self.h5.h5p.DATASET_CREATE);dcpl.set_layout(self.h5.h5d.COMPACT)
        d=self.h5.Dataset(self.h5.h5d.create(self.f.id,b'compact',self.h5.h5t.NATIVE_INT32,self.h5.h5s.create_simple((2,)),dcpl=dcpl))
        with self.assertRaises(self.error):self.p.describe_object(d,self.limits)

    def test_attributes_preserve_bytes_text_and_nulls(self):
        g=self.f.create_group('g');g.attrs['bytes']=self.np.array([b'abc',b''],dtype='S5');g.attrs['text']='Δοκιμή';g.attrs['negative']=self.np.array(-0.,dtype='>f8');g.attrs['null']=self.h5.Empty('f4')
        values={x['name']:x for x in self.p.describe_object(g,self.limits)['attributes']}
        self.assertEqual(values['bytes']['value'],self.np.array([b'abc',b''],dtype='S5').tobytes().hex())
        self.assertEqual(values['text']['value'],['Δοκιμή'.encode().hex()])
        self.assertIsNone(values['null']['shape']);self.assertIsNone(values['null']['value'])

    def test_reserved_scale_metadata_strict(self):
        s=self.f.create_dataset('scale',data=[1,2]);s.make_scale('');d=self.f.create_dataset('data',data=[3,4]);d.dims[0].attach_scale(s);d.dims[0].label='x'
        self.assertEqual(self.p.describe_object(s,self.limits)['scale_name'],'')
        self.assertEqual(self.p.describe_object(d,self.limits)['labels'],['x'])
        g=self.f.create_group('bad');g.attrs['CLASS']='ordinary'
        with self.assertRaises(self.error):self.p.describe_object(g,self.limits)
        d.attrs['NAME']='ordinary'
        with self.assertRaises(self.error):self.p.describe_object(d,self.limits)

    def test_multiaxis_blocks_bounded_and_complete(self):
        shape=(3,19,7);a=self.np.arange(399,dtype='i4').reshape(shape);recovered=self.np.zeros_like(a)
        for sel in self.p.iter_blocks(shape,4,32):
            self.assertLessEqual(a[sel].nbytes,32);recovered[sel]=a[sel]
        self.np.testing.assert_array_equal(a,recovered)


    def test_external_virtual_unknown_filters_and_named_types(self):
        d=self.f.create_dataset('external',(2,),dtype='i4',external=[('not-opened.raw',0,self.h5.h5f.UNLIMITED)])
        with self.assertRaises(self.error):self.p.describe_object(d,self.limits)
        layout=self.h5.VirtualLayout((2,),dtype='i4');layout[:]=self.h5.VirtualSource('not-opened.h5','data',shape=(2,))
        d=self.f.create_virtual_dataset('virtual',layout)
        with self.assertRaises(self.error):self.p.describe_object(d,self.limits)
        dcpl=self.h5.h5p.create(self.h5.h5p.DATASET_CREATE);dcpl.set_chunk((2,));dcpl.set_filter(32042,self.h5.h5z.FLAG_OPTIONAL,())
        d=self.h5.Dataset(self.h5.h5d.create(self.f.id,b'unknown_filter',self.h5.h5t.NATIVE_INT32,self.h5.h5s.create_simple((2,)),dcpl=dcpl))
        with self.assertRaises(self.error):self.p.describe_object(d,self.limits)
        t=self.h5.h5t.NATIVE_INT32.copy();t.commit(self.f.id,b'named_type')
        d=self.h5.Dataset(self.h5.h5d.create(self.f.id,b'committed',t,self.h5.h5s.create_simple((2,))))
        with self.assertRaises(self.error):self.p.describe_object(d,self.limits)

    def test_attribute_region_nested_and_limits(self):
        from dataclasses import replace
        g=self.f.create_group('g');g.attrs.create('region',self.h5.RegionReference(),dtype=self.h5.regionref_dtype)
        with self.assertRaises(self.error):self.p.describe_object(g,self.limits)
        del g.attrs['region'];g.attrs['oversized']='a'*20
        with self.assertRaises(self.error) as e:self.p.describe_object(g,replace(self.limits,max_attribute_bytes=10))
        self.assertEqual(e.exception.code,'RESOURCE')
        del g.attrs['oversized'];nested=self.np.empty(1,dtype=[('r',self.h5.ref_dtype)]);nested['r'][0]=self.h5.Reference();g.attrs.create('nested',nested)
        with self.assertRaises(self.error):self.p.describe_object(g,self.limits)

    def test_null_empty_reference_and_noncanonical_bool_complex(self):
        for name,shape in [('null',None),('empty',(0,)),('scalar',())]:
            d=self.f.create_dataset(name,shape=shape,dtype=self.h5.ref_dtype)
            self.assertEqual(self.p.describe_object(d,self.limits)['dtype']['kind'],'reference')
        enum=self.h5.enum_dtype({'FALSE':0,'TRUE':2},basetype='i1')
        d=self.f.create_dataset('bad_bool',(1,),dtype=enum)
        with self.assertRaises(self.error):self.p.describe_object(d,self.limits)
        compound=self.np.dtype({'names':['r','i'],'formats':['f4','f4'],'offsets':[0,8],'itemsize':12})
        d=self.f.create_dataset('padded_complex',(1,),dtype=compound)
        with self.assertRaises(self.error):self.p.describe_object(d,self.limits)

    def test_nonrepresentable_attribute_type_is_unsupported(self):
        t=self.h5.h5t.STD_I32LE.copy();t.set_offset(1)
        aid=self.h5.h5a.create(self.f.id,b'custom',t,self.h5.h5s.create(self.h5.h5s.SCALAR));aid.close()
        with self.assertRaises(self.error) as e:self.p.describe_object(self.f,self.limits)
        self.assertEqual(e.exception.code,'UNSUPPORTED');self.assertEqual(e.exception.path,'/@custom')

    def test_invalid_vlen_attribute_and_label_encoding_reports_path(self):
        g=self.f.create_group('g');g.attrs.create('bad',self.np.array([b'\xff'],dtype=object),dtype=self.h5.string_dtype('ascii'))
        with self.assertRaises(self.error) as e:self.p.describe_object(g,self.limits)
        self.assertEqual(e.exception.code,'UNSUPPORTED');self.assertEqual(e.exception.path,'/g@bad')
        d=self.f.create_dataset('d',data=[1]);d.attrs.create('DIMENSION_LABELS',self.np.array([b'\xff'],dtype=object),dtype=self.h5.string_dtype('ascii'))
        with self.assertRaises(self.error) as e:self.p.describe_object(d,self.limits)
        self.assertEqual(e.exception.code,'UNSUPPORTED');self.assertEqual(e.exception.path,'/d@DIMENSION_LABELS')

    def test_huge_fixed_element_rejected_before_fill(self):
        from dataclasses import replace
        from unittest.mock import patch
        d=self.f.create_dataset('big_fixed',(0,),dtype='S128')
        with patch.object(self.h5.Dataset,'fillvalue',property(lambda obj: (_ for _ in ()).throw(AssertionError('must not read fill')))):
            with self.assertRaises(self.error) as e:self.p.describe_object(d,replace(self.limits,chunk_bytes=64))
        self.assertEqual(e.exception.code,'RESOURCE')

if __name__ == '__main__': unittest.main()
