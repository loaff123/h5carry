"""Bounded Linux subprocess transport for trusted, quiescent local HDF5 files.

No native scientific library is imported here. The wall deadline starts immediately
before process creation and covers startup, request delivery and result collection.
It excludes request validation/encoding. On expiry the entire process group is
killed; OS reaping can add up to two seconds. An uninterruptible kernel operation
cannot be guaranteed to finish by a userspace deadline. This is resource
containment, not a filesystem/network or native-memory-safety sandbox.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

REQUEST_BYTES = 8 * 1024 * 1024
_ERROR_CODES = frozenset({'INVALID', 'UNSUPPORTED', 'RESOURCE', 'SOURCE_CHANGED', 'MISMATCH', 'ERROR'})


@dataclass(frozen=True)
class ProcessLimits:
    """Lowering-only release limits; CPU hard limit is one second above soft."""
    memory_bytes: int = 2 * 1024**3
    cpu_seconds: int = 120
    wall_seconds: float = 180
    file_bytes: int = 1024**3
    result_bytes: int = 8 * 1024**2
    stdout_bytes: int = 64 * 1024
    stderr_bytes: int = 64 * 1024

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            expected = (int, float) if field.name == 'wall_seconds' else (int,)
            if isinstance(value, bool) or not isinstance(value, expected):
                raise TypeError(f'{field.name} must be a positive number of the declared type')
            if value <= 0 or value > field.default or not math.isfinite(value):
                raise ValueError(f'{field.name} must be positive and no greater than {field.default}')

    def to_dict(self):
        return asdict(self)


def _error(code: str, message: str) -> dict:
    return {'ok': False, 'error': {'code': code, 'message': message}}


def _supported() -> bool:
    if sys.platform != 'linux' or not all(hasattr(os, name) for name in ('killpg', 'waitid', 'WNOWAIT', 'WEXITED', 'WNOHANG')):
        return False
    try:
        import resource
    except ImportError:
        return False
    return all(hasattr(resource, name) for name in ('RLIMIT_AS', 'RLIMIT_CPU', 'RLIMIT_FSIZE', 'RLIMIT_CORE'))


class _EnvironmentUnavailable(RuntimeError):
    pass


def _environment(limits: ProcessLimits, result_fd: int) -> dict[str, str]:
    # Python's isolated mode also ignores PYTHON*; remove these as defense in depth.
    env = {k: v for k, v in os.environ.items() if not k.startswith(('PYTHON', 'LD_', 'DYLD_', 'HDF5_', 'H5CARRY_', 'OPENBLAS_', 'MKL_', 'OMP_', 'NUMEXPR_'))}
    import sysconfig
    # Derive dependency directories from this trusted interpreter, never from a
    # request or inherited PYTHONPATH. -S prevents .pth/sitecustomize execution.
    paths = list(dict.fromkeys(sysconfig.get_path(name) for name in ('purelib', 'platlib')))
    context = {'paths': paths, 'prefix': sys.prefix, 'exec_prefix': sys.exec_prefix}
    encoded_context = json.dumps(context, ensure_ascii=True, separators=(',', ':'))
    if len(encoded_context) > 16384 or any(not isinstance(p, str) or not os.path.isabs(p) or '\x00' in p for p in [*paths,sys.prefix,sys.exec_prefix]):
        raise _EnvironmentUnavailable('trusted Python dependency context is invalid or oversized')
    env.update({'H5CARRY_PYTHON_CONTEXT': encoded_context,
                'HDF5_PLUGIN_PRELOAD': '::', 'OPENBLAS_NUM_THREADS': '1',
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
                'H5CARRY_RESULT_FD': str(result_fd),
                'H5CARRY_PROCESS_LIMITS': json.dumps(limits.to_dict(), separators=(',', ':'))})
    return env


def _encode_request(request: dict) -> bytes:
    # iterencode prevents collecting an arbitrarily large serialized request.
    chunks, size = [], 0
    for chunk in json.JSONEncoder(ensure_ascii=False, allow_nan=False, separators=(',', ':'), sort_keys=True).iterencode(request):
        if len(chunk) > REQUEST_BYTES:
            raise OverflowError('request exceeds byte limit')
        encoded = chunk.encode('utf-8')
        size += len(encoded)
        if size > REQUEST_BYTES:
            raise OverflowError('request exceeds byte limit')
        chunks.append(encoded)
    return b''.join(chunks)


def _validate_request(request: dict) -> None:
    from .model import Limits
    if not isinstance(request, dict):
        raise ValueError('request must be an object')
    operation = request.get('operation')
    variants = {
        'plan': {'operation', 'source', 'selections', 'limits'},
        'plan_v2': {'operation', 'source', 'request', 'limits'},
        'write_v2': {'operation', 'source', 'plan', 'staging', 'limits'},
        'write': {'operation', 'source', 'graph', 'staging', 'limits'},
        'verify': {'operation', 'source', 'output', 'plan', 'limits'},
        'inspect': {'operation', 'output', 'limits'},
    }
    if not isinstance(operation, str) or operation not in variants or set(request) != variants[operation]:
        raise ValueError('unknown operation or unexpected/missing request fields')
    for name in ('source', 'output', 'staging'):
        if name in request and (not isinstance(request[name], str) or not request[name] or '\x00' in request[name]):
            raise ValueError(f'{name} must be a nonempty local pathname without NUL')
    Limits.from_dict(request['limits'])
    if operation == 'plan':
        from .model import validate_path
        if not isinstance(request['selections'], list) or not request['selections']:
            raise ValueError('selections must be a nonempty list')
        for selection in request['selections']:
            validate_path(selection)
    if operation == 'plan_v2':
        from .selection import validate_selection
        validate_selection(request['request'], Limits.from_dict(request['limits']))
    for name in ('plan', 'graph'):
        if name in request and not isinstance(request[name], dict):
            raise ValueError(f'{name} must be an object')


def _duplicate_free(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def _parse_json(data: bytes):
    def nonfinite(value):
        raise ValueError('nonfinite JSON number')
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('nonfinite JSON number')
        return result
    return json.loads(data.decode('utf-8'), object_pairs_hook=_duplicate_free,
                      parse_constant=nonfinite, parse_float=finite_float)


def _parse_result(data: bytes) -> dict:
    try:
        result = _parse_json(data)
        if not isinstance(result, dict) or type(result.get('ok')) is not bool:
            raise ValueError('result is not an envelope')
        if result['ok']:
            if set(result) != {'ok', 'result'}:
                raise ValueError('invalid success envelope')
        else:
            if set(result) != {'ok', 'error'} or not isinstance(result['error'], dict):
                raise ValueError('invalid error envelope')
            error = result['error']
            if set(error) not in ({'code', 'message'}, {'code', 'message', 'path'}):
                raise ValueError('invalid error fields')
            if not isinstance(error['code'], str) or error['code'] not in _ERROR_CODES or not isinstance(error['message'], str):
                raise ValueError('invalid error diagnostic')
            if 'path' in error and not isinstance(error['path'], str):
                raise ValueError('invalid diagnostic path')
        return result
    except (ValueError, UnicodeError, RecursionError, TypeError):
        return _error('ERROR', 'Native worker returned missing, malformed, or incomplete result protocol')


def _kill_group(process: subprocess.Popen) -> None:
    # start_new_session creates a dedicated group. Retain the leader's zombie
    # with waitid(WNOWAIT) until this call so its PID cannot be recycled.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _run_worker(command: list[str], request: bytes, limits: ProcessLimits, *, capture: bool = False) -> dict:
    """Private bounded runner; production exposes only run_native's fixed worker.

    Original test helpers may call this with a trusted executable. No request
    field, configuration file, or CLI argument chooses an executable/module.
    """
    if not _supported():
        return _error('UNSUPPORTED', 'Native execution requires the qualified Linux resource-limit profile')
    if not isinstance(limits, ProcessLimits):
        return _error('INVALID', 'process_limits must be a ProcessLimits value')
    if not isinstance(request, bytes):
        return _error('INVALID', 'encoded request must be bytes')
    if len(request) > REQUEST_BYTES:
        return _error('RESOURCE', 'Native request exceeds the request byte limit')
    read_fd, write_fd = os.pipe()
    process = None
    selector = selectors.DefaultSelector()
    buffers = {name: bytearray() for name in ('result', 'stdout', 'stderr')}
    caps = {'result': limits.result_bytes, 'stdout': limits.stdout_bytes, 'stderr': limits.stderr_bytes}
    failure = None
    returncode = None
    group_killed = False
    deadline = time.monotonic() + limits.wall_seconds
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=_environment(limits, write_fd),
                                   pass_fds=(write_fd,), start_new_session=True, close_fds=True,
                                   bufsize=0)
        os.close(write_fd)
        write_fd = -1
        for stream, name in ((process.stdout, 'stdout'), (process.stderr, 'stderr'), (read_fd, 'result')):
            fd = stream if isinstance(stream, int) else stream.fileno()
            os.set_blocking(fd, False)
            selector.register(fd, selectors.EVENT_READ, name)
        os.set_blocking(process.stdin.fileno(), False)
        selector.register(process.stdin.fileno(), selectors.EVENT_WRITE, 'request')
        offset = 0
        while True:
            observed = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            if observed is not None and not group_killed:
                _kill_group(process)
                group_killed = True
                # Close input even if a helper exited before accepting it.
                if not process.stdin.closed:
                    try:
                        selector.unregister(process.stdin.fileno())
                    except KeyError:
                        pass
                    process.stdin.close()
            if observed is not None and not selector.get_map():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = _error('RESOURCE', 'Native worker exceeded its wall deadline')
                break
            for key, mask in selector.select(min(remaining, .025)):
                fd, name = key.fd, key.data
                if name == 'request':
                    try:
                        count = os.write(fd, request[offset:offset + 65536])
                        offset += count
                    except BrokenPipeError:
                        offset = len(request)
                    except BlockingIOError:
                        continue
                    if offset == len(request):
                        selector.unregister(fd)
                        process.stdin.close()
                else:
                    # Read at most remaining capacity plus one byte to detect excess.
                    available = caps[name] - len(buffers[name])
                    try:
                        chunk = os.read(fd, min(65536, available + 1))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(fd)
                    elif len(chunk) > available:
                        buffers[name].extend(chunk[:available])
                        failure = _error('RESOURCE', f'Native worker exceeded the {name} byte limit')
                        break
                    else:
                        buffers[name].extend(chunk)
            if failure is not None:
                break
    except _EnvironmentUnavailable:
        failure = _error('UNSUPPORTED', 'Trusted Python dependency directories could not be established')
    except OSError as exc:
        failure = _error('ERROR', f'Native worker transport failed ({type(exc).__name__}, errno={exc.errno})')
    except Exception as exc:
        failure = _error('ERROR', f'Native worker transport failed unexpectedly ({type(exc).__name__})')
    finally:
        if process is not None:
            try:
                _kill_group(process)
            except OSError as exc:
                failure = _error('ERROR', f'Native worker group cleanup failed (errno={exc.errno})')
            try:
                returncode = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                failure = _error('ERROR', 'Native worker was killed but OS reaping did not complete within two seconds')
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        selector.close()
        os.close(read_fd)
        if write_fd != -1:
            os.close(write_fd)
    if failure is None:
        if returncode == -signal.SIGXCPU:
            failure = _error('RESOURCE', 'Native worker exhausted its CPU limit')
        elif returncode == -signal.SIGXFSZ:
            failure = _error('RESOURCE', 'Native worker exhausted its output file-size limit')
        elif returncode is not None and returncode < 0:
            failure = _error('ERROR', f'Native worker terminated unexpectedly by signal {-returncode}')
        elif returncode != 0:
            failure = _error('ERROR', f'Native worker exited unexpectedly with status {returncode}')
    result = failure if failure is not None else _parse_result(bytes(buffers['result']))
    if capture:
        result = dict(result, _stdout=bytes(buffers['stdout']).decode('utf-8', 'replace'),
                      _stderr=bytes(buffers['stderr']).decode('utf-8', 'replace'))
    return result


def run_native(request: dict, process_limits: ProcessLimits | None = None) -> dict:
    """Execute exactly one fixed native operation, returning a bounded envelope."""
    if not _supported():
        return _error('UNSUPPORTED', 'Native execution requires the qualified Linux resource-limit profile')
    limits = process_limits if process_limits is not None else ProcessLimits()
    if not isinstance(limits, ProcessLimits):
        return _error('INVALID', 'process_limits must be a ProcessLimits value')
    try:
        data = _encode_request(request)
    except OverflowError:
        return _error('RESOURCE', 'Native request exceeds the request byte limit')
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return _error('INVALID', 'Native request is not bounded finite JSON')
    try:
        _validate_request(request)
    except Exception as exc:
        from .model import CarryError
        if isinstance(exc, CarryError):
            return {'ok': False, 'error': exc.to_dict()}
        if isinstance(exc, (ValueError, TypeError, KeyError)):
            return _error('INVALID', str(exc))
        return _error('ERROR', f'Native request validation failed ({type(exc).__name__})')
    return _run_worker([sys.executable, '-I', '-S', '-u', str(Path(__file__).with_name('worker.py'))], data, limits)
