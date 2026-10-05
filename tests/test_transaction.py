import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_plan import empty_plan


class TransactionTests(unittest.TestCase):
    def setup_paths(self, root):
        from h5carry.model import fingerprint
        source=Path(root)/'source.h5'; source.write_bytes(b'original')
        plan=empty_plan(); plan['source']=fingerprint(source)
        return source, plan, Path(root)/'out.h5', Path(root)/'report.json'

    @staticmethod
    def native(request):
        if request['operation']=='write':
            Path(request['staging']).write_bytes(b'verified-output')
            return {'ok':True,'result':None}
        return {'ok':True,'result':{'status':'verified','diagnostics':[],'coverage':{},'runtime':{}}}

    def test_unit_verified_publication_never_clobbers(self):
        from h5carry.transaction import export_checked
        from h5carry.model import CarryError
        with tempfile.TemporaryDirectory() as d:
            s,p,o,r=self.setup_paths(d)
            with patch('h5carry.transaction.run_native', side_effect=self.native):
                result=export_checked(s,p,o,r)
                self.assertEqual(result['publication'], {'output_published':True,'report_published':True})
                self.assertEqual(o.read_bytes(),b'verified-output')
                self.assertEqual(json.loads(r.read_text())['source'],p['source'])
                with self.assertRaises(CarryError): export_checked(s,p,o,r)
            self.assertEqual(s.read_bytes(),b'original')
            self.assertFalse(list(Path(d).glob('.h5carry-*')))

    def test_path_aliases_and_existing_report_fail_before_native(self):
        from h5carry.transaction import export_checked
        from h5carry.model import CarryError
        with tempfile.TemporaryDirectory() as d:
            s,p,o,r=self.setup_paths(d)
            alias=Path(d)/'alias'; os.link(s,alias)
            with patch('h5carry.transaction.run_native') as native:
                for out,report in [(s,r),(alias,r),(o,s),(o,o)]:
                    with self.subTest(out=out,report=report),self.assertRaises(CarryError): export_checked(s,p,out,report)
                r.write_text('existing')
                with self.assertRaises(CarryError): export_checked(s,p,o,r)
                native.assert_not_called()

    def test_destination_race_has_no_overwrite(self):
        from h5carry.transaction import export_checked
        from h5carry.model import CarryError
        with tempfile.TemporaryDirectory() as d:
            s,p,o,r=self.setup_paths(d)
            def racing(request):
                result=self.native(request)
                if request['operation']=='verify': o.write_bytes(b'racer')
                return result
            with patch('h5carry.transaction.run_native', side_effect=racing),self.assertRaises(CarryError):
                export_checked(s,p,o,r)
            self.assertEqual(o.read_bytes(),b'racer'); self.assertFalse(r.exists())
            self.assertFalse(list(Path(d).glob('.h5carry-*')))

    def test_report_race_reports_partial_and_retains_verified_output(self):
        from h5carry.transaction import export_checked, PublicationError
        with tempfile.TemporaryDirectory() as d:
            s,p,o,r=self.setup_paths(d)
            real_link=os.link
            def race_link(src,dst,**kwargs):
                if Path(dst)==r: r.write_text('racer')
                return real_link(src,dst,**kwargs)
            with patch('h5carry.transaction.run_native', side_effect=self.native),patch('h5carry.transaction.os.link',side_effect=race_link):
                with self.assertRaises(PublicationError) as found: export_checked(s,p,o,r)
            self.assertEqual(found.exception.publication,{'output_published':True,'report_published':False})
            self.assertEqual(o.read_bytes(),b'verified-output'); self.assertEqual(r.read_text(),'racer')

    def test_interrupt_after_each_atomic_link_reports_actual_publication(self):
        from h5carry.transaction import export_checked, PublicationError
        for after_report in (False, True):
            with self.subTest(after_report=after_report),tempfile.TemporaryDirectory() as d:
                s,p,o,r=self.setup_paths(d)
                real_link=os.link
                def interrupted_link(src,dst,**kwargs):
                    real_link(src,dst,**kwargs)
                    if Path(dst)==(r if after_report else o): raise KeyboardInterrupt
                with patch('h5carry.transaction.run_native',side_effect=self.native),patch('h5carry.transaction.os.link',side_effect=interrupted_link):
                    with self.assertRaises(PublicationError) as found: export_checked(s,p,o,r)
                self.assertEqual(found.exception.publication,{'output_published':True,'report_published':after_report})
                self.assertTrue(o.exists()); self.assertEqual(r.exists(),after_report)
                self.assertFalse(list(Path(d).glob('.h5carry-*')))

    def test_source_change_mismatch_and_crash_leave_no_output(self):
        from h5carry.transaction import export_checked
        from h5carry.model import CarryError
        for variant in ('changed','mismatch','crash'):
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as d:
                s,p,o,r=self.setup_paths(d)
                def failing(request):
                    result=self.native(request)
                    if request['operation']=='verify':
                        if variant=='changed': s.write_bytes(b'changed')
                        if variant=='mismatch': result['result']['status']='mismatch'
                        if variant=='crash': raise RuntimeError('unexpected')
                    return result
                with patch('h5carry.transaction.run_native', side_effect=failing),self.assertRaises((CarryError,RuntimeError)):
                    export_checked(s,p,o,r)
                self.assertFalse(o.exists()); self.assertFalse(r.exists())
                self.assertFalse(list(Path(d).glob('.h5carry-*')))
