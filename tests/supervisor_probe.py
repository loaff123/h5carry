"""Benign test helpers, never a production native operation."""
import builtins
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

sys.path.insert(0, sys.argv[sys.argv.index('--package-root') + 1])
from h5carry.worker import _install_limits, _prepare_native, _send_result

values = json.loads(os.environ['H5CARRY_PROCESS_LIMITS'])
_install_limits(values)
fd = int(os.environ['H5CARRY_RESULT_FD'])
if '--ignore-input' in sys.argv:
    time.sleep(30)
request = json.loads(sys.stdin.buffer.read())
mode = request['mode']

def send(result):
    _send_result(result, fd, values['result_bytes'])

if mode == 'echo':
    print('benign stdout')
    print('benign stderr', file=sys.stderr)
    send({'ok':True,'result':{'answer':42}})
elif mode == 'create_fixture':
    _prepare_native()
    import h5py
    with h5py.File(request['path'], 'w') as file:
        file.create_dataset('values', data=[1,2,3])
    send({'ok':True,'result':None})
elif mode == 'environment':
    names=['HDF5_PLUGIN_PATH','HDF5_DRIVER','HDF5_VOL_CONNECTOR','HDF5_PLUGIN_PRELOAD','PYTHONPATH','PYTHONSTARTUP','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']
    send({'ok':True,'result':{name:os.environ[name] for name in names if name in os.environ}})
elif mode.startswith('main_'):
    import errno
    import io
    from h5carry import worker
    from h5carry.model import CarryError, Limits
    raw = json.dumps({'operation':'inspect','output':'/tmp/none','limits':Limits().to_dict()}).encode()
    if mode == 'main_invalid': raw = b'{"operation":"eval"}'
    if mode == 'main_duplicate': raw = b'{"operation":"inspect","operation":"eval"}'
    sys.stdin = io.TextIOWrapper(io.BytesIO(raw))
    old_prepare = worker._prepare_native
    def prepare():
        print('native imported')
        return old_prepare()
    worker._prepare_native = prepare
    def dispatch(request):
        if mode == 'main_memory': return bytearray(512*1024**2)
        if mode == 'main_unexpected': raise RuntimeError('original benign injected bug')
        if mode == 'main_oserror': raise OSError(errno.ENOMEM, 'original benign allocation error')
        if mode == 'main_unsupported': raise CarryError('UNSUPPORTED','original benign unsupported feature','/x')
        raise AssertionError('native dispatch must not be reached')
    worker._dispatch = dispatch
    if mode == 'main_setup_denied':
        def denied(values): raise PermissionError('controlled setup denial')
        worker._install_limits = denied
    raise SystemExit(worker.main())
elif mode == 'native_environment':
    original_import = builtins.__import__
    checked = []
    def guarded_import(name, *args, **kwargs):
        if name in ('h5py', 'numpy'):
            assert resource.getrlimit(resource.RLIMIT_AS)[1] <= values['memory_bytes']
            assert resource.getrlimit(resource.RLIMIT_FSIZE)[1] <= values['file_bytes']
            assert os.environ['HDF5_PLUGIN_PRELOAD'] == '::'
            assert 'HDF5_PLUGIN_PATH' not in os.environ
            checked.append(name)
        return original_import(name, *args, **kwargs)
    builtins.__import__ = guarded_import
    loading_state = _prepare_native()
    import h5py
    send({'ok':True,'result':{
        'limits':{name:list(resource.getrlimit(kind)) for name,kind in [('as',resource.RLIMIT_AS),('cpu',resource.RLIMIT_CPU),('fsize',resource.RLIMIT_FSIZE),('core',resource.RLIMIT_CORE)]},
        'pre_import_checked':bool(checked), 'worker_file':str(Path(_prepare_native.__code__.co_filename).resolve()), 'plugin_loading':loading_state, 'plugin_paths':h5py.h5pl.size(),
        'plugin_environment_path':os.environ.get('HDF5_PLUGIN_PATH'), 'plugin_preload':os.environ.get('HDF5_PLUGIN_PRELOAD'),
        'threads':[os.environ.get(x) for x in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']]}})
elif mode == 'memory':
    try:
        bytearray(256*1024**2)
    except MemoryError:
        send({'ok':False,'error':{'code':'RESOURCE','message':'memory allocation denied'}})
    else:
        send({'ok':True,'result':'unexpected allocation'})
elif mode == 'cpu':
    while True:
        pass
elif mode == 'wall':
    time.sleep(30)
elif mode == 'file':
    with open(request['target'],'wb',buffering=0) as stream:
        for _ in range(10):
            stream.write(b'x'*4096)
elif mode == 'crash':
    os.kill(os.getpid(), signal.SIGABRT)
elif mode == 'success_then_fail':
    send({'ok':True,'result':None})
    sys.exit(17)
elif mode in ('orphan','orphan_wall'):
    pid = os.fork()
    if pid == 0:
        time.sleep(30)
        os._exit(0)
    Path(request['pidfile']).write_text(str(pid))
    if mode == 'orphan_wall':
        time.sleep(30)
    send({'ok':True,'result':None})
    os._exit(0)
elif mode == 'error':
    send({'ok':False,'error':{'code':request['code'],'message':'declared','path':'/x'}})
elif mode.endswith('_flood'):
    target = {'result_flood':fd,'stdout_flood':1,'stderr_flood':2}[mode]
    while True:
        os.write(target,b'x'*65536)
else:
    payloads = {
        'empty':b'', 'truncated':b'{"ok":true,"result":',
        'duplicate':b'{"ok":true,"ok":false,"result":null}', 'malformed':b'junk',
        'nan':b'{"ok":true,"result":NaN}',
        'overflow_float':b'{"ok":true,"result":1e999}',
        'unknown_error':b'{"ok":false,"error":{"code":"OOPS","message":"x"}}',
        'extra_key':b'{"ok":true,"result":null,"extra":1}',
        'two_results':b'{"ok":true,"result":null}{"ok":true,"result":null}',
        'invalid_utf8':b'{"ok":true,"result":"\xff"}',
    }
    os.write(fd,payloads[mode])
