"""Original graph fixtures exercising selection and rejection."""
import unittest
from tests.test_profile import NativeFixtureMixin

class ScanTests(NativeFixtureMixin, unittest.TestCase):
    # Fixture setup reused, profile cases are intentionally tested once only.
    def test_selection_forward_closure_aliases_ancestors(self):
        from h5carry.scan import make_plan_native
        f=self.f;rootref=f.create_dataset('root_dependency',data=[7]);f.attrs['root_ref']=rootref.ref
        a=f.create_group('runs/A');a.attrs['note']='α';d=a.create_dataset('signal',data=[1,2]);a['alias']=d
        scale=f.create_dataset('scales/time',data=[0,1]);scale.make_scale('time');d.dims[0].attach_scale(scale)
        c=f.create_dataset('runs/C/signal',data=[9001,9002]);c.dims[0].attach_scale(scale)
        gain=f.create_dataset('calibration/gain',data=2.);f['unused_gain_alias']=gain;a.attrs['gain']=gain.ref
        a['soft_a']=self.h5.SoftLink('/calibration/gain');a['soft_b']=self.h5.SoftLink('/runs/A/soft_a')
        source=f.filename;f.close();g=make_plan_native(source,['/runs/A'],self.limits)
        paths={x['path'] for x in g['links']};self.assertNotIn('/runs/C',paths);self.assertNotIn('/unused_gain_alias',paths);self.assertIn('/root_dependency',paths)
        links={x['path']:x for x in g['links']};self.assertEqual(links['/runs/A/signal']['target'],links['/runs/A/alias']['target'])
        self.assertEqual(len(g['scales']),1);self.assertEqual(g['scales'][0]['scale'],'/scales/time')
        self.assertTrue(any(x['owner']=='/' and x['target']=='/root_dependency' for x in g['references']))
        self.assertEqual(g,make_plan_native(source,['/runs/A'],self.limits))

    def test_reference_group_expands_children_and_nulls(self):
        from h5carry.scan import make_plan_native
        f=self.f;g=f.create_group('selected');dep=f.create_group('dependency');dep.create_dataset('nested/x',data=[1]);refs=g.create_dataset('refs',(2,),dtype=self.h5.ref_dtype);refs[:]=[dep.ref,self.h5.Reference()]
        source=f.filename;f.close();result=make_plan_native(source,['/selected'],self.limits)
        self.assertIn('/dependency/nested/x',{x['path'] for x in result['links']});self.assertTrue(any(x['target'] is None for x in result['references']))

    def test_global_external_link_refusal_without_target_open(self):
        from h5carry.scan import make_plan_native
        self.f.create_dataset('selected',data=[1]);self.f['hidden']=self.h5.ExternalLink('/definitely/missing','/x');source=self.f.filename;self.f.close()
        with self.assertRaises(self.error) as e:make_plan_native(source,['/selected'],self.limits)
        self.assertEqual(e.exception.code,'UNSUPPORTED');self.assertEqual(e.exception.path,'/hidden')

    def test_group_alias_and_true_soft_cycle_refused(self):
        from h5carry.scan import make_plan_native
        g=self.f.create_group('g');self.f['alias']=g;source=self.f.filename;self.f.close()
        with self.assertRaises(self.error):make_plan_native(source,['/g'],self.limits)
        with self.h5.File(source,'w') as f:f['a']=self.h5.SoftLink('/b');f['b']=self.h5.SoftLink('/a')
        with self.assertRaises(self.error):make_plan_native(source,['/a'],self.limits)

    def test_dangling_reference_and_malformed_reverse_refused(self):
        from h5carry.scan import make_plan_native
        d=self.f.create_dataset('d',data=[1]);s=self.f.create_dataset('s',data=[1]);s.make_scale('s');d.dims[0].attach_scale(s);r=s.attrs['REFERENCE_LIST'];r['dimension']=8;s.attrs.modify('REFERENCE_LIST',r);source=self.f.filename;self.f.close()
        with self.assertRaises(self.error):make_plan_native(source,['/d'],self.limits)

    def test_source_budgets_and_read_only(self):
        from dataclasses import replace
        from h5carry.model import fingerprint
        from h5carry.scan import make_plan_native
        d=self.f.create_dataset('d',data=self.np.arange(32,dtype='i8'));self.f.create_group('other');source=self.f.filename;self.f.close();before=fingerprint(source)
        for limits in [replace(self.limits,max_objects=1),replace(self.limits,max_edges=1),replace(self.limits,max_payload_bytes=8),replace(self.limits,max_plan_bytes=32)]:
            with self.assertRaises(self.error) as e:make_plan_native(source,['/d'],limits)
            self.assertEqual(e.exception.code,'RESOURCE')
        self.assertEqual(fingerprint(source),before)

    def test_dangling_object_reference(self):
        from h5carry.scan import make_plan_native
        d=self.f.create_dataset('refs',(1,),dtype=self.h5.ref_dtype);target=self.f.create_dataset('deleted',data=[1]);ref=target.ref;d[0]=ref;del self.f['deleted'];target.id.close();source=self.f.filename;self.f.close()
        with self.assertRaises(self.error):make_plan_native(source,['/refs'],self.limits)

    def test_selected_alias_not_unselected_lexical_name(self):
        from h5carry.scan import make_plan_native
        d=self.f.create_dataset('a_unselected',data=[1]);self.f['z_selected']=d;g=self.f.create_group('group');g.attrs['ref']=d.ref;source=self.f.filename;self.f.close()
        result=make_plan_native(source,['/group','/z_selected'],self.limits)
        self.assertNotIn('/a_unselected',{x['path'] for x in result['links']});self.assertEqual(result['references'][0]['target'],'/z_selected')

    def test_soft_dot_relative_group_chain_and_cycle(self):
        from h5carry.scan import make_plan_native
        d=self.f.create_dataset('g/d',data=[1]);self.f['g/s']=self.h5.SoftLink('./d');self.f['a']=self.h5.SoftLink('/g');self.f['b']=self.h5.SoftLink('/a');source=self.f.filename;self.f.close()
        result=make_plan_native(source,['/b'],self.limits)
        self.assertEqual({x['path'] for x in result['links']},{'/a','/b','/g','/g/d','/g/s'})

    def test_global_bad_reserved_metadata_outside_selection(self):
        from h5carry.scan import make_plan_native
        self.f.create_dataset('selected',data=[1]);self.f.create_dataset('excluded',data=[2]).attrs['NAME']='not a scale';source=self.f.filename;self.f.close()
        with self.assertRaises(self.error) as e:make_plan_native(source,['/selected'],self.limits)
        self.assertEqual(e.exception.path,'/excluded')

    def test_dimension_list_actual_reference_bytes_limit(self):
        from dataclasses import replace
        from h5carry.scan import make_plan_native
        d=self.f.create_dataset('data',data=[1])
        for index in range(4):
            s=self.f.create_dataset('s'+str(index),data=[1]);s.make_scale('s');d.dims[0].attach_scale(s)
        source=self.f.filename;self.f.close()
        with self.assertRaises(self.error) as e:make_plan_native(source,['/data'],replace(self.limits,max_attribute_bytes=24))
        self.assertEqual(e.exception.code,'RESOURCE')

    def test_discovery_deadline_and_external_open_spy(self):
        from dataclasses import replace
        from unittest.mock import patch
        from h5carry.scan import make_plan_native
        self.f.create_dataset('selected',data=[1]);self.f['hidden']=self.h5.ExternalLink('/not-a-target.h5','/data');source=self.f.filename;self.f.close()
        with patch('h5carry.scan.time.monotonic',side_effect=[0,2]):
            with self.assertRaises(self.error) as e:make_plan_native(source,['/selected'],replace(self.limits,discovery_seconds=1))
        self.assertEqual(e.exception.code,'RESOURCE')
        original=self.h5.Group.__getitem__
        def guarded(group,name):
            if name in ('hidden','/hidden'):raise AssertionError('external link was dereferenced')
            return original(group,name)
        with patch.object(self.h5.Group,'__getitem__',guarded):
            with self.assertRaises(self.error) as e:make_plan_native(source,['/selected'],self.limits)
        self.assertEqual(e.exception.code,'UNSUPPORTED')

    def test_native_soft_chain_limit_is_not_success(self):
        from h5carry.scan import make_plan_native
        self.f.create_dataset('target',data=[1])
        for index in range(20):self.f['s%02d'%index]=self.h5.SoftLink('/target' if index==19 else '/s%02d'%(index+1))
        source=self.f.filename;self.f.close()
        with self.assertRaises(self.error) as e:make_plan_native(source,['/s00'],self.limits)
        self.assertEqual(e.exception.code,'UNSUPPORTED')

    def test_non_utf8_name_rejected_without_native_name_lookup(self):
        from h5carry.scan import make_plan_native
        self.f.create_dataset('selected',data=[1]);self.f.create_dataset(b'bad\xff',data=[2]);source=self.f.filename;self.f.close()
        with self.assertRaises(self.error) as e:make_plan_native(source,['/selected'],self.limits)
        self.assertEqual(e.exception.code,'UNSUPPORTED')
