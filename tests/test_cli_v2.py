"""CLI, plan and staged publication tests for typed requests."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
from tests.test_profile import NativeFixtureMixin
from tests.test_scan_v2 import request,box

def cli_command():
    import h5carry
    package_root = str(Path(h5carry.__file__).resolve().parents[1])
    return [sys.executable,'-I','-c','import sys; sys.path.insert(0,'+repr(package_root)+'); from h5carry.cli import main; raise SystemExit(main())']

class TypedIntegrationTests(NativeFixtureMixin,unittest.TestCase):
    def setup_source(self):
        self.f.create_dataset('A',data=self.np.arange(24,dtype='i4').reshape((6,4)))
        source=Path(self.f.filename);self.f.close();return source
    def test_plan_cli_selection_json_and_mutual_exclusion(self):
        source=self.setup_source();root=Path(self.tmp.name);r=root/'request.json';r.write_text(json.dumps(request(box('/A'))));p=root/'plan.json'
        cli=cli_command()
        proc=subprocess.run(cli+['plan',str(source),'--selection-json',str(r),'--out',str(p)],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr);plan=json.loads(p.read_text());self.assertEqual(plan['version'],2)
        proc=subprocess.run(cli+['plan',str(source),'--select','/A','--selection-json',str(r),'--out',str(root/'bad.json')],capture_output=True,text=True)
        self.assertNotEqual(proc.returncode,0);self.assertFalse((root/'bad.json').exists())
    def test_native_operation_and_preflight_refusal_before_stage(self):
        source=self.setup_source()
        from h5carry.plan_v2 import make_selection_plan
        from h5carry.transaction import export_checked
        p=make_selection_plan(source,request(box('/A')))
        p['graph']['objects'][-1]['metadata']['payload_sha256']='0'*64
        with patch('h5carry.transaction._stage',side_effect=AssertionError('must not stage refused plan')):
            with self.assertRaises(self.error) as cm:export_checked(source,p,self.tmp.name+'/output.h5',self.tmp.name+'/report.json')
        self.assertEqual(cm.exception.code,'MISMATCH')
    def test_selection_request_is_distinct_from_plan_destination(self):
        source=self.setup_source();r=Path(self.tmp.name)/'request.json';content=json.dumps(request(box('/A')));r.write_text(content)
        proc=subprocess.run(cli_command()+['plan',str(source),'--selection-json',str(r),'--out',str(r)],capture_output=True,text=True)
        self.assertEqual(proc.returncode,2);self.assertEqual(r.read_text(),content)
    def test_export_verify_inspect_no_clobber_and_source_identity(self):
        from h5carry.model import fingerprint
        from h5carry.plan_v2 import make_selection_plan
        from h5carry.transaction import export_checked,verify_checked,inspect_checked
        source=self.setup_source();before=fingerprint(source);root=Path(self.tmp.name)
        p=make_selection_plan(source,request(box('/A')));out=root/'output.h5';report=root/'export.json'
        result=export_checked(source,p,out,report)
        self.assertEqual(result['status'],'verified',result);self.assertTrue(result['coverage']['source_equality'])
        self.assertEqual(verify_checked(source,out,p)['status'],'verified')
        inspected=inspect_checked(out);self.assertEqual(inspected['status'],'inspected');self.assertFalse(inspected['coverage']['source_equality'])
        with self.assertRaises(self.error):export_checked(source,p,out,root/'other-report.json')
        self.assertFalse((root/'other-report.json').exists());self.assertEqual(fingerprint(source),before)
        self.assertFalse(list(root.glob('.h5carry-*')))
    def test_partial_publication_preserved_and_interrupted_staging_cleaned(self):
        from h5carry.plan_v2 import make_selection_plan
        from h5carry import transaction as tx
        source=self.setup_source();root=Path(self.tmp.name);p=make_selection_plan(source,request(box('/A')))
        publish=tx._publish
        def partial(stage,destination):
            if str(destination).endswith('.json'):raise OSError('test report failure')
            publish(stage,destination)
        with patch.object(tx,'_publish',partial),self.assertRaises(tx.PublicationError) as cm:
            tx.export_checked(source,p,root/'partial.h5',root/'partial.json')
        self.assertEqual(cm.exception.publication,{'output_published':True,'report_published':False})
        self.assertTrue((root/'partial.h5').exists());self.assertFalse((root/'partial.json').exists());self.assertFalse(list(root.glob('.h5carry-*')))
        with patch.object(tx,'_publish',side_effect=KeyboardInterrupt),self.assertRaises(KeyboardInterrupt):
            tx.export_checked(source,p,root/'interrupt.h5',root/'interrupt.json')
        self.assertFalse((root/'interrupt.h5').exists());self.assertFalse(list(root.glob('.h5carry-*')))
