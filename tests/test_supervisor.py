"""Original benign containment probes; none read external HDF5 input."""
import dataclasses
import importlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from h5carry import supervisor

PROBE = Path(__file__).with_name('supervisor_probe.py')
ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(supervisor.__file__).resolve().parents[1]


@unittest.skipUnless(sys.platform == 'linux', 'qualified Linux containment profile')
class SupervisorTests(unittest.TestCase):
    def probe(self, mode, *, limits=None, **kwargs):
        request = json.dumps({'mode': mode, **kwargs}).encode()
        return supervisor._run_worker([sys.executable, '-I', '-S', '-u', str(PROBE), '--package-root', str(PACKAGE_ROOT)], request,
            limits or supervisor.ProcessLimits(), capture=True)

    def test_default_limits_are_release_ceilings(self):
        limits = supervisor.ProcessLimits()
        self.assertEqual(limits.memory_bytes, 2 * 1024**3)
        self.assertEqual(limits.cpu_seconds, 120)
        self.assertEqual(limits.wall_seconds, 180)
        self.assertEqual(limits.file_bytes, 1024**3)
        self.assertEqual(limits.result_bytes, 8 * 1024**2)
        self.assertEqual(limits.stdout_bytes, 64 * 1024)
        self.assertEqual(limits.stderr_bytes, 64 * 1024)

    def test_limits_are_frozen_lowering_only_and_strict(self):
        default = supervisor.ProcessLimits()
        for field in dataclasses.fields(default):
            name = field.name
            for value in (False, True, 0, -1, float('inf'), float('nan'), '10', 10**1000, getattr(default, name) + 1):
                with self.subTest(name=name, value=value):
                    with self.assertRaises((ValueError, TypeError)):
                        supervisor.ProcessLimits(**{name: value})
        for name in ('memory_bytes', 'cpu_seconds', 'file_bytes', 'result_bytes', 'stdout_bytes', 'stderr_bytes'):
            with self.assertRaises((ValueError, TypeError)):
                supervisor.ProcessLimits(**{name: 1.5})
        self.assertEqual(supervisor.ProcessLimits(wall_seconds=.1).wall_seconds, .1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            default.cpu_seconds = 1

    def test_normal_protocol_and_output_capture(self):
        result = self.probe('echo')
        self.assertEqual(result['ok'], True)
        self.assertEqual(result['result'], {'answer': 42})
        self.assertIn('benign stdout', result['_stdout'])
        self.assertIn('benign stderr', result['_stderr'])

    def test_actual_limits_and_thread_environment_before_native_import(self):
        result = self.probe('native_environment', limits=supervisor.ProcessLimits(cpu_seconds=5, file_bytes=4096))
        self.assertTrue(result['ok'], result)
        value = result['result']
        self.assertEqual(value['limits']['as'], [2*1024**3, 2*1024**3])
        self.assertEqual(value['limits']['cpu'], [5, 6])
        self.assertEqual(value['limits']['fsize'], [4096, 4096])
        self.assertEqual(value['limits']['core'], [0, 0])
        self.assertTrue(value['pre_import_checked'])
        self.assertEqual(value['worker_file'], str(Path(supervisor.__file__).with_name('worker.py').resolve()))
        self.assertEqual(value['plugin_loading'], 0)
        self.assertEqual(value['plugin_paths'], 0)
        self.assertIsNone(value['plugin_environment_path'])
        self.assertEqual(value['plugin_preload'], '::')
        self.assertEqual(value['threads'], ['1'] * 4)

    def test_production_bootstrap_installs_controls_before_first_native_import(self):
        from h5carry.model import Limits
        request = {'operation':'inspect','output':'/tmp/original-probe-placeholder','limits':Limits().to_dict()}
        result = supervisor._run_worker([sys.executable,'-I','-S','-u',str(PROBE.with_name('supervisor_bootstrap_probe.py')), '--package-root',str(PACKAGE_ROOT)],
            json.dumps(request).encode(), supervisor.ProcessLimits(cpu_seconds=5,file_bytes=4096), capture=True)
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['result']['observed_imports'], ['h5py','numpy'])

    def test_startup_skips_sitecustomize_and_pth_before_and_after_limits(self):
        import sysconfig
        import venv
        from h5carry.model import Limits
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp/'original.h5'
            self.assertTrue(self.probe('create_fixture',path=str(source))['ok'])
            environment = tmp/'original-test-venv'
            venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
            site = Path(sysconfig.get_path('purelib', vars={'base':str(environment),'platbase':str(environment)}))
            site.mkdir(parents=True, exist_ok=True)
            site_marker, pth_marker = tmp/'site-hook-ran', tmp/'pth-hook-ran'
            (site/'sitecustomize.py').write_text('from pathlib import Path;Path('+repr(str(site_marker))+').write_text("ran")\n')
            (site/'original-hook.pth').write_text('import pathlib;pathlib.Path('+repr(str(pth_marker))+').write_text("ran")\n')
            # Confirm these ORIGINAL benign hooks genuinely execute under normal
            # isolated startup, before testing that production suppresses them.
            subprocess.run([str(environment/'bin/python'), '-I', '-c', 'pass'], check=True, timeout=5)
            self.assertTrue(site_marker.exists()); self.assertTrue(pth_marker.exists())
            site_marker.unlink(); pth_marker.unlink()
            original_get_path = sysconfig.get_path
            dependency_site = original_get_path('purelib')
            def controlled_path(name, *args, **kwargs):
                if name == 'purelib': return str(site)
                if name == 'platlib': return dependency_site
                return original_get_path(name,*args,**kwargs)
            with mock.patch.object(supervisor.sys,'executable',str(environment/'bin/python')), mock.patch.object(sysconfig,'get_path',side_effect=controlled_path):
                result = supervisor.run_native({'operation':'plan','source':str(source),'selections':['/values'],'limits':Limits().to_dict()})
            self.assertFalse(site_marker.exists(), 'sitecustomize ran during native worker startup')
            self.assertFalse(pth_marker.exists(), '.pth hook ran during native worker startup')
            self.assertTrue(result['ok'], result)
            self.assertEqual({x['id'] for x in result['result']['objects']}, {'/','/values'})

    def test_actual_memory_allocation_denial(self):
        result = self.probe('memory', limits=supervisor.ProcessLimits(memory_bytes=96*1024**2))
        self.assertFalse(result['ok'])
        self.assertEqual(result['error']['code'], 'RESOURCE')
        self.assertIn('memory', result['error']['message'].lower())

    def test_actual_cpu_exhaustion(self):
        start = time.monotonic()
        result = self.probe('cpu', limits=supervisor.ProcessLimits(cpu_seconds=1, wall_seconds=6))
        self.assertEqual(result['error']['code'], 'RESOURCE', result)
        self.assertIn('cpu', result['error']['message'].lower())
        self.assertLess(time.monotonic()-start, 5)

    def test_actual_wall_exhaustion(self):
        start = time.monotonic()
        result = self.probe('wall', limits=supervisor.ProcessLimits(wall_seconds=.15))
        self.assertEqual(result['error']['code'], 'RESOURCE', result)
        self.assertIn('wall', result['error']['message'].lower())
        self.assertLess(time.monotonic()-start, 3)

    def test_actual_file_size_exhaustion(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'limited.bin'
            result = self.probe('file', target=str(target), limits=supervisor.ProcessLimits(file_bytes=4096))
            self.assertEqual(result['error']['code'], 'RESOURCE', result)
            self.assertLessEqual(target.stat().st_size, 4096)
            self.assertIn('file', result['error']['message'].lower())

    def test_unexpected_crash_is_error_not_unsupported(self):
        result = self.probe('crash')
        self.assertEqual(result['error']['code'], 'ERROR', result)
        self.assertIn('signal', result['error']['message'].lower())

    def test_nonzero_exit_is_error_even_after_success_record(self):
        result = self.probe('success_then_fail')
        self.assertEqual(result['error']['code'], 'ERROR', result)

    def test_empty_truncated_duplicate_and_malformed_results_are_errors(self):
        for mode in ('empty', 'truncated', 'duplicate', 'malformed', 'nan', 'unknown_error', 'extra_key', 'two_results', 'invalid_utf8', 'overflow_float'):
            with self.subTest(mode=mode):
                result = self.probe(mode)
                self.assertFalse(result['ok'], result)
                self.assertEqual(result['error']['code'], 'ERROR', result)

    def test_declared_error_codes_survive_protocol(self):
        for code in ('INVALID', 'UNSUPPORTED', 'RESOURCE', 'SOURCE_CHANGED', 'MISMATCH', 'ERROR'):
            result = self.probe('error', code=code)
            self.assertEqual(result['error'], {'code': code, 'message': 'declared', 'path': '/x'})

    def test_result_stdout_and_stderr_are_bounded_separately(self):
        for mode, field in (('result_flood', 'result_bytes'), ('stdout_flood', 'stdout_bytes'), ('stderr_flood', 'stderr_bytes')):
            with self.subTest(mode=mode):
                result = self.probe(mode, limits=supervisor.ProcessLimits(**{field:1024}, wall_seconds=3))
                self.assertEqual(result['error']['code'], 'RESOURCE', result)
                self.assertIn(field.split('_')[0], result['error']['message'].lower())
                self.assertLessEqual(len(result.get('_stdout','').encode()), 1024)
                self.assertLessEqual(len(result.get('_stderr','').encode()), 1024)

    def assert_descendant_killed(self, pid):
        # SIGKILL delivery is asynchronous. Wait for the observed condition,
        # rather than requiring a descendant to be scheduled before waitpid
        # returns for its already-exited parent.
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                state = Path(f'/proc/{pid}/stat').read_text().split()[2]
            except (FileNotFoundError, ProcessLookupError):
                return
            if state == 'Z':
                return
            time.sleep(.005)
        self.fail(f'descendant {pid} remains live after SIGKILL (state {state})')

    def test_exit_with_live_descendant_cleans_entire_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp)/'child.pid'
            start = time.monotonic()
            result = self.probe('orphan', pidfile=str(pidfile), limits=supervisor.ProcessLimits(wall_seconds=2))
            self.assertTrue(result['ok'], result)
            self.assertLess(time.monotonic()-start, 1.5)
            pid = int(pidfile.read_text())
            self.assert_descendant_killed(pid)

    def test_wall_timeout_cleans_entire_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = Path(tmp)/'child.pid'
            result = self.probe('orphan_wall', pidfile=str(pidfile), limits=supervisor.ProcessLimits(wall_seconds=.3))
            self.assertEqual(result['error']['code'], 'RESOURCE', result)
            pid = int(pidfile.read_text())
            self.assert_descendant_killed(pid)

    def test_unsupported_platform_fails_closed_before_launch(self):
        with mock.patch.object(supervisor.sys, 'platform', 'darwin'), mock.patch.object(supervisor.subprocess, 'Popen') as launch:
            result = supervisor.run_native({'operation':'inspect','output':'/tmp/none','limits':{}})
            self.assertEqual(result['error']['code'], 'UNSUPPORTED', result)
            launch.assert_not_called()

    def test_no_arbitrary_operation_or_extra_fields(self):
        for request in ({'operation':'eval','code':'print(1)'}, {'operation':'inspect','output':'a','limits':{},'module':'os'}, {'operation':'inspect','output':'a\x00b','limits':{}}, {'operation':'inspect','output':'a','limits':{},'command':['sh']}):
            result = supervisor.run_native(request)
            self.assertEqual(result['error']['code'], 'INVALID', result)

    def test_request_cap_is_enforced_before_launch(self):
        with mock.patch.object(supervisor.subprocess, 'Popen') as launch:
            result = supervisor.run_native({'operation':'plan','source':'a','selections':['/'+'x'*(8*1024**2)],'limits':{}})
            self.assertEqual(result['error']['code'], 'RESOURCE', result)
            launch.assert_not_called()

    def test_actual_main_maps_memory_and_unexpected_errors(self):
        for mode, expected in [('main_memory','RESOURCE'), ('main_unexpected','ERROR'), ('main_oserror','RESOURCE'), ('main_unsupported','UNSUPPORTED')]:
            with self.subTest(mode=mode):
                result = self.probe(mode, limits=supervisor.ProcessLimits(memory_bytes=256*1024**2))
                self.assertEqual(result['error']['code'], expected, result)

    def test_copied_native_test_harness_uses_selected_package_without_source_tree(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_dir = root/'tests'
            test_dir.mkdir()
            (test_dir/'__init__.py').write_text('')
            for filename in ('native_runner.py','native_bootstrap.py'):
                shutil.copyfile(ROOT/'tests'/filename, test_dir/filename)
            (test_dir/'original_import_probe.py').write_text('from h5carry import worker;print(worker.__file__)\n')
            package_root = str(Path(supervisor.__file__).resolve().parents[1])
            result = subprocess.run([sys.executable,'-m','tests.native_runner','tests.original_import_probe'],
                cwd=root, env=dict(os.environ,PYTHONPATH=package_root), capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn(str(Path(supervisor.__file__).with_name('worker.py').resolve()),result.stdout)

    def test_unexpected_transport_exception_is_error_not_unsupported(self):
        with mock.patch.object(supervisor.selectors.DefaultSelector, 'select', side_effect=ValueError('original benign transport bug')):
            result = self.probe('echo')
        self.assertEqual(result['error']['code'], 'ERROR', result)

    def test_unavailable_limit_setup_fails_closed_with_diagnostic(self):
        result = self.probe('main_setup_denied')
        self.assertEqual(result['error']['code'], 'UNSUPPORTED', result)
        self.assertNotIn('native imported', result['_stdout'])

    def test_main_invalid_request_is_rejected_before_native_import(self):
        for mode in ('main_invalid','main_duplicate'):
            result = self.probe(mode)
            self.assertEqual(result['error']['code'], 'INVALID', result)
            self.assertNotIn('native imported', result['_stdout'])

    def test_blocked_request_pipe_is_subject_to_wall_deadline(self):
        start = time.monotonic()
        result = supervisor._run_worker([sys.executable, '-I', '-S', '-u', str(PROBE), '--package-root', str(PACKAGE_ROOT), '--ignore-input'],
            b'x'*(8*1024**2), supervisor.ProcessLimits(wall_seconds=.1))
        self.assertEqual(result['error']['code'], 'RESOURCE', result)
        self.assertIn('wall', result['error']['message'].lower())
        self.assertLess(time.monotonic()-start, 3)

    def test_hostile_native_and_python_environment_is_removed(self):
        with mock.patch.dict(os.environ, {'HDF5_PLUGIN_PATH':'/tmp/not-a-plugin-path',
            'HDF5_DRIVER':'remote-driver', 'HDF5_VOL_CONNECTOR':'unexpected',
            'PYTHONPATH':'/tmp/not-a-python-path', 'PYTHONSTARTUP':'/tmp/no.py',
            'OPENBLAS_NUM_THREADS':'200', 'OMP_NUM_THREADS':'200'}):
            result = self.probe('environment')
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['result'], {'HDF5_PLUGIN_PRELOAD':'::','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})

    def test_supervisor_import_does_not_import_native_libraries(self):
        code = 'import sys;sys.path.insert(0,'+repr(str(PACKAGE_ROOT))+');import h5carry.supervisor;assert "h5py" not in sys.modules;assert "numpy" not in sys.modules'
        result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
