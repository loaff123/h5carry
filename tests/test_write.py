"""Fresh native reconstruction acceptance tests."""
import unittest
from tests.test_profile import NativeFixtureMixin

class WriteTests(NativeFixtureMixin, unittest.TestCase):
    def test_reconstruct_aliases_refs_scales_properties(self):
        from h5carry.scan import make_plan_native
        from h5carry.write import write_staging
        f=self.f;f.attrs['experiment']='µ';g=f.create_group('run');d=g.create_dataset('data',data=self.np.array([-0.,2.],dtype='>f8'));g['alias']=d
        s=f.create_dataset('time',data=[0,1]);s.make_scale('');d.dims[0].attach_scale(s);d.dims[0].label='τ'
        x=f.create_dataset('excluded',data=[9,9]);x.dims[0].attach_scale(s)
        refs=g.create_dataset('refs',(2,),maxshape=(None,),chunks=(2,),compression='gzip',compression_opts=7,dtype=self.h5.ref_dtype);refs[:]=[d.ref,self.h5.Reference()];g.attrs['cal']=s.ref;g['soft']=self.h5.SoftLink('/time')
        source=f.filename;f.close();graph=make_plan_native(source,['/run'],self.limits);output=self.tmp.name+'/output.h5';open(output,'wb').close();write_staging(source,graph,output,self.limits)
        with self.h5.File(output,'r') as o:
            self.assertEqual(set(o),{'run','time'});self.assertEqual(o.attrs['experiment'],'µ');self.assertEqual(o['run/data'].id,o['run/alias'].id)
            self.assertEqual(o['run/data'].dtype.str,'>f8');self.assertEqual(o['run/data'][...].tobytes(),self.np.array([-0.,2.],dtype='>f8').tobytes())
            self.assertEqual(o['run/refs'].chunks,(2,));self.assertEqual(o['run/refs'].compression_opts,7);self.assertEqual(o['run/refs'].maxshape,(None,));self.assertEqual(o[o['run/refs'][0]].id,o['run/data'].id);self.assertFalse(o['run/refs'][1])
            self.assertEqual(o['run/data'].dims[0].label,'τ');self.assertEqual(o['run/data'].dims[0][0].id,o['time'].id);self.assertEqual(len(o['time'].attrs['REFERENCE_LIST']),1)
            self.assertEqual(o['time'].attrs['NAME'],b'');self.assertEqual(o[o['run'].attrs['cal']].id,o['time'].id)

    def test_exact_all_supported_payloads_and_attributes(self):
        from h5carry.scan import make_plan_native
        from h5carry.write import write_staging
        f=self.f;g=f.create_group('選択');g.attrs['bytes']=self.np.array([b'a\0b',b'z'],dtype='S8');g.attrs['ascii']=self.np.array([b'a',b'b'],dtype=self.h5.string_dtype('ascii'));g.attrs['utf8']=self.np.array(['µ','字'],dtype=self.h5.string_dtype('utf-8'));g.attrs['null']=self.h5.Empty('>f4');g.attrs.create('null_ref',self.h5.Empty(self.h5.ref_dtype))
        for index,dtype in enumerate(['i1','u1','>i2','<u4','>u8','f4','>f8','c8','>c16','?','S7']):
            a=self.np.zeros((2,3),dtype=dtype);a.flat[0]=1
            d=g.create_dataset('d'+str(index),data=a,chunks=(1,3),compression='gzip',shuffle=True,fletcher32=True,fillvalue=self.np.asarray(1,dtype=dtype))
            d.attrs['array']=a;d.attrs['scalar']=self.np.asarray(a.flat[0],dtype=dtype)
        bits=self.np.array([0x7ff8000000000012,0x8000000000000000,0x7ff0000000000000],dtype='u8');g.create_dataset('bits',data=bits.view('f8'))
        for name,shape in [('scalar',()),('empty',(0,3)),('null',None)]:g.create_dataset(name,shape=shape,dtype='>f8')
        for name,shape in [('scalar_ref',()),('empty_ref',(0,)),('null_ref',None)]:g.create_dataset(name,shape=shape,dtype=self.h5.ref_dtype)
        g.attrs.create('ref_scalar',self.h5.Reference(),dtype=self.h5.ref_dtype)
        source=f.filename;f.close();graph=make_plan_native(source,['/選択'],self.limits);output=self.tmp.name+'/out.h5';open(output,'wb').close();write_staging(source,graph,output,self.limits)
        with self.h5.File(source,'r') as s,self.h5.File(output,'r') as o:
            for obj in graph['objects']:
                path=obj['id'];self.assertEqual(self.p.describe_object(s[path],self.limits),self.p.describe_object(o[path],self.limits))
            self.assertEqual(o['選択/bits'][...].tobytes(),bits.tobytes())

    def test_repeated_scale_names_multidimension_and_reference_cycles(self):
        from h5carry.scan import make_plan_native
        from h5carry.write import write_staging
        f=self.f;a=f.create_group('a');b=f.create_group('b');a.attrs['b']=b.ref;b.attrs['a']=a.ref;d=a.create_dataset('data',data=self.np.ones((2,2)))
        s=f.create_dataset('s',data=3);t=f.create_dataset('t',data=self.np.arange(7).reshape(1,7));s.make_scale('same');t.make_scale('same');d.dims[0].attach_scale(s);d.dims[0].attach_scale(t);d.dims[1].attach_scale(s)
        source=f.filename;f.close();graph=make_plan_native(source,['/a'],self.limits);outpath=self.tmp.name+'/out.h5';open(outpath,'wb').close();write_staging(source,graph,outpath,self.limits)
        with self.h5.File(outpath,'r') as o:
            self.assertEqual(o[o['a'].attrs['b']].name,'/b');self.assertEqual(o[o['b'].attrs['a']].name,'/a')
            self.assertEqual(len(o['a/data'].dims[0]),2);self.assertEqual(len(o['s'].attrs['REFERENCE_LIST']),2);self.assertEqual(len(o['t'].attrs['REFERENCE_LIST']),1)
            self.assertEqual(o['a/data'].dims[1][0].id,o['s'].id)

    def test_maximum_attribute_payload_survives_reconstruction(self):
        from h5carry.scan import make_plan_native
        from h5carry.write import write_staging
        source=self.f.filename;self.f.close()
        payload=self.np.arange(self.limits.max_attribute_bytes,dtype='u1')
        with self.h5.File(source,'w',track_order=True) as f:
            f.attrs['large_root']=payload;f.create_group('g',track_order=True).attrs['large_group']=payload
        graph=make_plan_native(source,['/g'],self.limits);outpath=self.tmp.name+'/out.h5';open(outpath,'wb').close();write_staging(source,graph,outpath,self.limits)
        with self.h5.File(outpath,'r') as o:
            self.np.testing.assert_array_equal(o.attrs['large_root'],payload);self.np.testing.assert_array_equal(o['g'].attrs['large_group'],payload)

    def test_actual_source_payload_budget_is_enforced_before_staging(self):
        import copy
        from pathlib import Path
        from dataclasses import replace
        from h5carry.model import fingerprint
        from h5carry.scan import make_plan_native
        from h5carry.supervisor import run_native
        self.f.create_dataset('values',data=self.np.arange(1024,dtype='u1'))
        source=self.f.filename;self.f.close();graph=make_plan_native(source,['/values'],self.limits);before=fingerprint(source)
        lower=replace(self.limits,max_payload_bytes=1)
        for forged_shape in (False,True):
            with self.subTest(forged_shape=forged_shape):
                tampered=copy.deepcopy(graph);tampered['payload_bytes']=1
                if forged_shape:
                    next(item for item in tampered['objects'] if item['kind']=='dataset')['metadata']['shape']=[1]
                stage=Path(self.tmp.name)/('stage-'+str(forged_shape)+'.h5');stage.touch()
                response=run_native({'operation':'write','source':source,'graph':tampered,'staging':str(stage),'limits':lower.to_dict()})
                self.assertFalse(response['ok'],response)
                self.assertEqual(response['error']['code'],'RESOURCE',response)
                self.assertEqual(stage.stat().st_size,0,'Budget must be checked before opening/allocating staging output')
                self.assertFalse((Path(self.tmp.name)/'final.h5').exists())
        self.assertEqual(fingerprint(source),before)
