"""Read-only native property gap in the two pinned Linux h5py builds.

This module is native-only and runs within H5Carry's supervised child. It loads
only the already imported extension, never searches for another HDF5 library.
"""
import ctypes
import h5py
from .model import CarryError


def chunk_options(dcpl):
    """H5Pget_chunk_opts is missing from the qualified h5py Python surface.

    H5Pequal does not compare these options in the qualified builds, so equality
    of creation lists cannot replace this check. hid_t is signed 64-bit there.
    """
    if (h5py.__version__, h5py.version.hdf5_version) not in {('3.14.0','1.14.6'),('3.16.0','2.0.0')}:
        raise CarryError('UNSUPPORTED','Native chunk-option reader requires a qualified h5py/HDF5 build')
    try:
        library=ctypes.CDLL(h5py.h5p.__file__)
        function=library.H5Pget_chunk_opts
        function.argtypes=(ctypes.c_int64,ctypes.POINTER(ctypes.c_uint))
        function.restype=ctypes.c_int
        value=ctypes.c_uint()
        with h5py._objects.phil:
            status=function(dcpl.id,ctypes.byref(value))
        if status < 0:
            raise CarryError('UNSUPPORTED','Native chunk-option query failed')
        return value.value
    except (OSError,AttributeError,TypeError) as exc:
        raise CarryError('UNSUPPORTED','Native chunk-option query is unavailable in this build') from exc
