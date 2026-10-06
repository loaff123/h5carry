import importlib.util
import unittest
from tests.test_profile import NativeFixtureMixin

class NativePropertyTests(NativeFixtureMixin,unittest.TestCase):
    def test_reads_exact_native_chunk_options(self):
        self.assertIsNotNone(importlib.util.find_spec('h5carry.native_properties'),'native option reader missing')
        from h5carry.native_properties import chunk_options
        import ctypes
        p=self.h5.h5p.create(self.h5.h5p.DATASET_CREATE);p.set_chunk((4,))
        self.assertEqual(chunk_options(p),0)
        library=ctypes.CDLL(self.h5.h5p.__file__);fn=library.H5Pset_chunk_opts
        fn.argtypes=(ctypes.c_int64,ctypes.c_uint);fn.restype=ctypes.c_int
        with self.h5._objects.phil:self.assertEqual(fn(p.id,2),0)
        self.assertEqual(chunk_options(p),2)
        clone=p.copy();clone.set_chunk((4,))
        self.assertTrue(p.equal(clone));self.assertEqual(chunk_options(clone),0)
    def test_missing_symbol_fails_closed(self):
        self.assertIsNotNone(importlib.util.find_spec('h5carry.native_properties'),'native option reader missing')
        from h5carry.native_properties import chunk_options
        from unittest.mock import patch
        p=self.h5.h5p.create(self.h5.h5p.DATASET_CREATE);p.set_chunk((1,))
        with patch('ctypes.CDLL',side_effect=OSError('missing')),self.assertRaises(self.error) as cm:chunk_options(p)
        self.assertEqual(cm.exception.code,'UNSUPPORTED')
