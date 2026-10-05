"""Private fixed-operation child. Never import h5py/numpy before installing limits."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import signal
import sys


def _install_limits(values: dict) -> None:
    if sys.platform != 'linux':
        raise RuntimeError('qualified Linux containment unavailable')
    import resource
    from h5carry.supervisor import ProcessLimits
    limits = ProcessLimits(**values)

    def lower(kind, soft, hard):
        old_soft, old_hard = resource.getrlimit(kind)
        if old_hard != resource.RLIM_INFINITY:
            hard = min(hard, old_hard)
        if old_soft != resource.RLIM_INFINITY:
            soft = min(soft, old_soft)
        resource.setrlimit(kind, (min(soft, hard), hard))

    lower(resource.RLIMIT_AS, limits.memory_bytes, limits.memory_bytes)
    lower(resource.RLIMIT_CPU, limits.cpu_seconds, limits.cpu_seconds + 1)
    lower(resource.RLIMIT_FSIZE, limits.file_bytes, limits.file_bytes)
    lower(resource.RLIMIT_CORE, 0, 0)
    signal.signal(signal.SIGXCPU, signal.SIG_DFL)
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
    os.environ.pop('HDF5_PLUGIN_PATH', None)
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'


def _install_dependency_paths() -> None:
    # Directly add only the parent's trusted interpreter directories. In
    # particular, never import site or call addsitedir (both execute hooks).
    raw = os.environ.get('H5CARRY_PYTHON_CONTEXT', '')
    if not raw or len(raw) > 16384:
        raise RuntimeError('missing bounded Python dependency context')
    context = json.loads(raw)
    if not isinstance(context, dict) or set(context) != {'paths','prefix','exec_prefix'}:
        raise RuntimeError('invalid Python dependency context')
    paths = context['paths']
    if not isinstance(paths, list) or not 1 <= len(paths) <= 2:
        raise RuntimeError('invalid Python dependency paths')
    if any(not isinstance(p,str) or not os.path.isabs(p) or '\x00' in p for p in [*paths,context['prefix'],context['exec_prefix']]):
        raise RuntimeError('invalid Python dependency directory')
    # Python <3.14 defers virtualenv prefix handling to site. Preserve the actual
    # launching interpreter's prefix without executing its startup hooks. This
    # also keeps nested original test subprocesses in the selected environment.
    sys.prefix, sys.exec_prefix = context['prefix'], context['exec_prefix']
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)


def _prepare_native() -> int:
    _install_dependency_paths()
    import h5py
    # PRELOAD='::' disables dynamic loading even during import; explicitly lock
    # state down and clear HDF5's compiled/default path list before any file open.
    import ctypes
    # h5py exposes the path API but not loading-state access. Resolve the exact
    # already-imported extension's linked HDF5, never a guessed system library.
    library = ctypes.CDLL(h5py.h5pl.__file__)
    set_state = library.H5PLset_loading_state
    set_state.argtypes = [ctypes.c_uint]
    set_state.restype = ctypes.c_int
    get_state = library.H5PLget_loading_state
    get_state.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    get_state.restype = ctypes.c_int
    state = ctypes.c_uint(0xFFFFFFFF)
    if set_state(0) < 0 or get_state(ctypes.byref(state)) < 0:
        raise RuntimeError('could not establish native plugin loading state')
    for _ in range(h5py.h5pl.size()):
        h5py.h5pl.remove(0)
    if state.value != 0 or h5py.h5pl.size() != 0:
        raise RuntimeError('could not disable native plugin loading')
    return state.value


def _send_result(result: dict, result_fd: int, cap: int) -> None:
    # Fully encode under a cap before writing so an encoding error cannot leave a
    # plausible but truncated success envelope. There is exactly one record.
    chunks, size = [], 0
    try:
        for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(',', ':'), sort_keys=True).iterencode(result):
            if len(chunk) > cap:
                raise OverflowError
            chunk = chunk.encode('utf-8')
            size += len(chunk)
            if size > cap:
                raise OverflowError
            chunks.append(chunk)
        data = b''.join(chunks)
    except OverflowError:
        data = b'{"ok":false,"error":{"code":"RESOURCE","message":"Native result exceeds the result byte limit"}}'
    except (ValueError, TypeError, UnicodeError, RecursionError):
        data = b'{"ok":false,"error":{"code":"ERROR","message":"Native result is not finite JSON"}}'
    offset = 0
    while offset < len(data):
        offset += os.write(result_fd, data[offset:offset + 65536])


def _dispatch(request: dict):
    from h5carry.model import Limits
    limits = Limits.from_dict(request['limits'])
    operation = request['operation']
    if operation == 'plan':
        from h5carry.scan import make_plan_native
        return make_plan_native(request['source'], request['selections'], limits)
    if operation == 'write':
        from h5carry.write import write_staging
        return write_staging(request['source'], request['graph'], request['staging'], limits)
    if operation == 'verify':
        from h5carry.verify import verify_export
        return verify_export(request['source'], request['output'], request['plan'], limits)
    if operation == 'inspect':
        from h5carry.verify import inspect_output
        return inspect_output(request['output'], limits)
    raise ValueError('unknown native operation')


def main() -> int:
    # Only the installed/source package containing this fixed script is added.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        result_fd = int(os.environ['H5CARRY_RESULT_FD'])
        process_limits = json.loads(os.environ['H5CARRY_PROCESS_LIMITS'])
    except (KeyError, ValueError, TypeError):
        return 70
    try:
        _install_limits(process_limits)
    except (OSError, ValueError, RuntimeError):
        # A verified unavailable containment mechanism is distinct from a crash.
        # No native import or file open occurs on this path.
        _send_result({'ok': False, 'error': {'code': 'UNSUPPORTED',
                     'message': 'Required Linux resource limits could not be established'}},
                     result_fd, process_limits['result_bytes'])
        return 0
    except Exception:
        return 70
    from h5carry.model import CarryError
    from h5carry.supervisor import REQUEST_BYTES, _parse_json, _validate_request
    try:
        data = sys.stdin.buffer.read(REQUEST_BYTES + 1)
        if len(data) > REQUEST_BYTES:
            raise CarryError('RESOURCE', 'Native request exceeds the request byte limit')
        try:
            request = _parse_json(data)
            _validate_request(request)
        except CarryError:
            raise
        except (ValueError, TypeError, UnicodeError, RecursionError, KeyError) as exc:
            raise CarryError('INVALID', 'Malformed native request') from exc
        _prepare_native()
        result = {'ok': True, 'result': _dispatch(request)}
    except CarryError as exc:
        result = {'ok': False, 'error': exc.to_dict()}
    except MemoryError:
        result = {'ok': False, 'error': {'code': 'RESOURCE', 'message': 'Native worker exhausted its memory limit'}}
    except OSError as exc:
        code = 'RESOURCE' if exc.errno in (errno.ENOMEM, errno.EFBIG, errno.ENOSPC, errno.EDQUOT) else 'ERROR'
        result = {'ok': False, 'error': {'code': code, 'message': f'Native operation failed ({type(exc).__name__}, errno={exc.errno})'}}
    except Exception as exc:
        result = {'ok': False, 'error': {'code': 'ERROR', 'message': f'Native operation failed unexpectedly ({type(exc).__name__})'}}
    try:
        _send_result(result, result_fd, process_limits['result_bytes'])
    except (MemoryError, OSError):
        return 71
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
