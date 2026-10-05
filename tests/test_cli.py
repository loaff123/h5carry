import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def test_help_version_and_native_free_import(self):
        result=subprocess.run([sys.executable,'-c',"import h5carry.cli,sys; assert 'h5py' not in sys.modules; assert 'numpy' not in sys.modules"],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,'-m','h5carry','--version'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('0.1.0a1',result.stdout)

    def test_missing_arguments_fail(self):
        from h5carry.cli import main
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as found: main(['export'])
        self.assertEqual(found.exception.code,2)

    def test_invalid_plan_json_is_explicit_invalid(self):
        from h5carry.cli import main
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source'; source.write_bytes(b'not opened')
            plan=Path(d)/'plan.json'; plan.write_text('{"x":NaN}')
            out=io.StringIO()
            with contextlib.redirect_stdout(out):
                code=main(['export',str(source),'--plan',str(plan),'--out',str(Path(d)/'out'),'--report',str(Path(d)/'report')])
            self.assertEqual(code,2)
            self.assertEqual(json.loads(out.getvalue())['diagnostics'][0]['code'],'INVALID')
            self.assertFalse((Path(d)/'out').exists())

    def test_limits_can_only_lower(self):
        from h5carry.cli import main
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            code=main(['plan','absent','--select','/','--out','absent-plan','--max-objects','10001'])
        self.assertEqual(code,2)
        self.assertEqual(json.loads(out.getvalue())['diagnostics'][0]['code'],'INVALID')
