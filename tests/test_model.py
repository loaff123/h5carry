import hashlib
import tempfile
import unittest
from pathlib import Path


class ModelTests(unittest.TestCase):
    def test_canonical_json_and_fingerprint(self):
        from h5carry.model import canonical_json, fingerprint
        self.assertEqual(canonical_json({'b': 1, 'a': 'λ'}), b'{"a":"\xce\xbb","b":1}')
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'source'; p.write_bytes(b'abc')
            self.assertEqual(fingerprint(p), {'sha256': hashlib.sha256(b'abc').hexdigest(), 'size': 3})

    def test_limits_strict_lowering_only(self):
        from h5carry.model import CarryError, Limits
        default = Limits()
        self.assertEqual(Limits.from_dict(default.to_dict()), default)
        for key, value in [('max_objects', True), ('max_objects', 0), ('max_objects', 10001), ('discovery_seconds', float('nan'))]:
            data = default.to_dict(); data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(CarryError): Limits.from_dict(data)
        with self.assertRaises(CarryError): Limits.from_dict({'max_objects': 10})

    def test_hdf5_path_strict_and_unicode(self):
        from h5carry.model import CarryError, validate_path
        for p in ['/', '/runs/λ', '/a.../b']:
            self.assertEqual(validate_path(p), p)
        for p in ['', 'relative', '//a', '/a/', '/a/../b', '/a/./b', '/a\0b']:
            with self.subTest(p=p), self.assertRaises(CarryError): validate_path(p)

    @unittest.skipUnless(__import__('sys').platform=='linux','FIFO check uses Linux')
    def test_nonregular_source_and_plan_fail_without_waiting_for_writer(self):
        import h5carry, os, subprocess, sys
        package_root = Path(h5carry.__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as d:
            fifo=Path(d)/'fifo'; os.mkfifo(fifo)
            for function in ('fingerprint','load_plan'):
                # Source-only and installed-artifact runs must test the same package as this process.
                code = ('import sys\nfrom pathlib import Path\n'
                        'root = Path(sys.argv[1]).resolve()\nsys.path.insert(0, str(root))\n'
                        'import h5carry, h5carry.model, h5carry.plan\n'
                        'assert Path(h5carry.__file__).resolve() == root / "h5carry" / "__init__.py"\n'
                        'assert Path(h5carry.model.__file__).resolve() == root / "h5carry" / "model.py"\n'
                        'assert Path(h5carry.plan.__file__).resolve() == root / "h5carry" / "plan.py"\n'
                        'from h5carry.model import CarryError, fingerprint\nfrom h5carry.plan import load_plan\n'
                        'try: '+function+'(sys.argv[2])\nexcept CarryError as e: assert e.code=="INVALID"\n'
                        'else: raise AssertionError("FIFO accepted")')
                result=subprocess.run([sys.executable,'-I','-S','-c',code,str(package_root),str(fifo)],
                                      capture_output=True,text=True,timeout=1)
                self.assertEqual(result.returncode,0,result.stderr)

    def test_errors_and_no_native_import(self):
        from h5carry.model import CarryError
        e = CarryError('INVALID', 'bad input', '/x')
        self.assertEqual(e.to_dict(), {'code':'INVALID', 'message':'bad input', 'path':'/x'})
