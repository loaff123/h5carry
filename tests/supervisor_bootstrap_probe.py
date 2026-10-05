"""Audit the actual production bootstrap without preinstalling any limits."""
import builtins
import json
import os
from pathlib import Path
import resource
import sys

sys.path.insert(0, sys.argv[sys.argv.index('--package-root') + 1])
from h5carry import worker
assert 'h5py' not in sys.modules and 'numpy' not in sys.modules
expected = json.loads(os.environ['H5CARRY_PROCESS_LIMITS'])
original_import = builtins.__import__
checked = set()

def guarded_import(name, *args, **kwargs):
    if name in ('h5py', 'numpy'):
        assert resource.getrlimit(resource.RLIMIT_AS)[1] <= expected['memory_bytes']
        assert resource.getrlimit(resource.RLIMIT_CPU)[0] <= expected['cpu_seconds']
        assert resource.getrlimit(resource.RLIMIT_FSIZE)[1] <= expected['file_bytes']
        assert resource.getrlimit(resource.RLIMIT_CORE) == (0,0)
        assert os.environ['HDF5_PLUGIN_PRELOAD'] == '::'
        assert 'HDF5_PLUGIN_PATH' not in os.environ
        assert os.environ['OPENBLAS_NUM_THREADS'] == '1'
        checked.add(name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import

def dispatch(request):
    assert {'h5py','numpy'} <= checked
    import h5py
    assert h5py.h5pl.size() == 0
    return {'observed_imports':sorted(checked)}

worker._dispatch = dispatch
raise SystemExit(worker.main())
