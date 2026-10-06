import importlib.util
from pathlib import Path
import unittest
from tests.test_scan_v2 import request,box,whole
from tests import test_scan_v2 as scan_cases
from tests.test_profile import NativeFixtureMixin

class SliceWriteTests(NativeFixtureMixin,unittest.TestCase):
    def build(self):return scan_cases.SliceScanTests.build(self)
    def plan(self,r):
        from h5carry.scan_v2 import make_plan_native
        from h5carry.model import fingerprint
        source=self.f.filename;self.f.flush(); self.f.close()
        result=make_plan_native(source,r,self.limits)
        return source,dict(format='h5carry-plan',version=2,source=fingerprint(source),request=r,limits=self.limits.to_dict(),**result)
    def write(self,source,plan):
        self.assertIsNotNone(importlib.util.find_spec('h5carry.write_v2'),'typed writer missing')
        from h5carry.write_v2 import write_staging
        path=self.tmp.name+'/out.h5';write_staging(source,plan,path,self.limits);return path
    def test_exact_crop_alias_scales_and_fixed_creation(self):
        m=self.build();source,p=self.plan(request(box('/A'),box('/B'),box('/alias'),mappings=m));path=self.write(source,p)
        with self.h5.File(path,'r') as o:
            self.assertEqual(o['A'][:].tolist(),[[101,102],[201,202],[301,302]])
            self.assertEqual(o['time'][:].tolist(),[10,20,30]);self.assertEqual(o['channel'][:].tolist(),[500,600]);self.assertNotIn('C',o)
            self.assertEqual(o['A'].id,o['alias'].id);self.assertEqual(o['A'].chunks,(3,2));self.assertEqual(o['A'].maxshape,(3,2));self.assertEqual(o['A'].compression,'gzip')
            self.assertEqual(o['A'].dims[0][0].id,o['B'].dims[0][0].id);self.assertEqual(len(o['time'].attrs['REFERENCE_LIST']),2)
    def test_empty_snapshot_and_outward_reference(self):
        m=self.build();gain=self.f.create_dataset('gain',data=[9]);self.f['A'].attrs['gain']=gain.ref
        source,p=self.plan(request(box('/A',(1,1),(1,3)),mappings=m[:2]));path=self.write(source,p)
        with self.h5.File(path,'r') as o:
            self.assertEqual(o['A'].shape,(0,2));self.assertEqual(o['A'].maxshape,(0,2));self.assertEqual(o['A'].chunks,(1,2));self.assertEqual(o[o['A'].attrs['gain']].id,o['gain'].id)
    def test_forged_plan_refused_before_staging(self):
        m=self.build();source,p=self.plan(request(box('/A'),mappings=m[:2]));p['graph']['objects'][-1]['metadata']['shape']=[1]
        with self.assertRaises(self.error):self.write(source,p)
        self.assertFalse(Path(self.tmp.name+'/out.h5').exists())
    def test_exact_float_endian_complex_payloads_and_fill(self):
        n=self.np
        for dtype,bits in [('>f8',[0x8000000000000000,0x7ff8000000000123,0x3ff0000000000000]),('>c16',[0x8000000000000000,0x7ff8000000000123,0,0x7ff0000000000000,0,0])]:
            values=n.array(bits,dtype='>u8').view(dtype)
            d=self.f.create_dataset('float' if dtype=='>f8' else 'complex',data=values,compression='gzip',shuffle=True,fletcher32=True,chunks=(3,),fillvalue=values[0]);d.attrs['bits']=values
        r=request(box('/float',(0,),(2,)),box('/complex',(0,),(2,)));source,p=self.plan(r);path=self.write(source,p)
        with self.h5.File(source,'r') as a,self.h5.File(path,'r') as b:
            for key in ('float','complex'):
                self.assertEqual(a[key][:2].tobytes(),b[key][:].tobytes());self.assertTrue(a[key].id.get_type().equal(b[key].id.get_type()))
                self.assertEqual(a[key].attrs['bits'].tobytes(),b[key].attrs['bits'].tobytes())
