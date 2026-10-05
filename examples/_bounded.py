"""Linux-only bounded launcher for the shipped, original benign examples.

This is not a file-format sandbox. Only the fixed example and qualification modules can be
launched, and each applies limits before importing NumPy or h5py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import sysconfig
import time

MODULES = frozenset({
    'examples.three_run_archive', 'examples.analyze_selected',
    'qualification.run_baselines', 'qualification.check_frozen',
})
MEMORY = 2 * 1024 ** 3
CPU = 120
FILE_SIZE = 1024 ** 3
WALL = 180
RESULT_SIZE = 2 * 1024 ** 2


def require_native_limits():
    """Refuse native imports outside a constrained Linux child."""
    if not sys.platform.startswith('linux'):
        raise RuntimeError('Examples require the qualified Linux process-limit profile')
    import resource
    for key, maximum in ((resource.RLIMIT_AS, MEMORY), (resource.RLIMIT_CPU, CPU),
                         (resource.RLIMIT_FSIZE, FILE_SIZE)):
        soft, _ = resource.getrlimit(key)
        if soft == resource.RLIM_INFINITY or soft <= 0 or soft > maximum:
            raise RuntimeError('Run this example through its bounded CLI wrapper')
    if os.environ.get('HDF5_PLUGIN_PRELOAD') != '::':
        raise RuntimeError('Dynamic HDF5 plugins must be disabled before native imports')


def launch(module, arguments):
    if module not in MODULES:
        raise ValueError('Unknown example module')
    if not sys.platform.startswith('linux'):
        print(json.dumps({'status': 'incomplete', 'error': 'Examples require Linux process limits'}))
        return 2
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('PYTHON', 'LD_', 'DYLD_', 'HDF5_', 'H5CARRY_'))}
    environment['HDF5_PLUGIN_PRELOAD'] = '::'
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '1'
    roots = list(dict.fromkeys(sysconfig.get_path(name) for name in ('purelib', 'platlib')))
    context = json.dumps({'paths': roots, 'prefix': sys.prefix, 'exec_prefix': sys.exec_prefix})
    if len(context.encode('utf-8')) > 16384:
        raise ValueError('Example dependency context exceeds byte ceiling')
    environment['H5CARRY_EXAMPLE_CONTEXT'] = context
    command = [sys.executable, '-I', '-S', str(Path(__file__).resolve()), '--child', module, *arguments]
    deadline = time.monotonic() + WALL
    process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, 'stdout')
    selector.register(process.stderr, selectors.EVENT_READ, 'stderr')
    buffers = {'stdout': bytearray(), 'stderr': bytearray()}
    failure = None
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = 'Example worker exceeded its wall deadline'
                break
            for key, _ in selector.select(min(remaining, 0.1)):
                part = os.read(key.fileobj.fileno(), 65536)
                if not part:
                    selector.unregister(key.fileobj)
                    continue
                buffers[key.data].extend(part)
                if sum(map(len, buffers.values())) > RESULT_SIZE:
                    failure = 'Example worker exceeded its bounded result size'
                    break
            if failure:
                break
        if failure:
            os.killpg(process.pid, signal.SIGKILL)
        code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        failure = 'Example worker exceeded its wall deadline'
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        code = process.wait()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    if failure:
        print(json.dumps({'status': 'incomplete', 'error': failure}))
        return 2
    sys.stderr.write(buffers['stderr'].decode('utf-8', errors='replace'))
    if code not in (0, 1, 2):
        print(json.dumps({'status': 'incomplete', 'error': 'Example worker terminated unexpectedly', 'returncode': code}))
        return 2
    try:
        record = json.loads(buffers['stdout'].decode('utf-8'))
        expected = {'examples.three_run_archive': 'generated',
                    'examples.analyze_selected': 'verified',
                    'qualification.run_baselines': 'complete',
                    'qualification.check_frozen': 'verified'}[module]
        if not isinstance(record, dict) or (code == 0 and record.get('status') != expected):
            raise ValueError('incomplete result')
    except (ValueError, UnicodeError):
        print(json.dumps({'status': 'incomplete', 'error': 'Example worker returned no complete result record'}))
        return 2
    print(json.dumps(record, sort_keys=True))
    return code


def _child(module, arguments):
    if module not in MODULES or not sys.platform.startswith('linux'):
        raise ValueError('Unknown or unsupported example worker')
    import resource
    for key, limit in ((resource.RLIMIT_AS, MEMORY), (resource.RLIMIT_CPU, CPU),
                       (resource.RLIMIT_FSIZE, FILE_SIZE)):
        resource.setrlimit(key, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.environ['HDF5_PLUGIN_PRELOAD'] = '::'
    os.environ.pop('HDF5_PLUGIN_PATH', None)
    require_native_limits()
    serialized = os.environ.get('H5CARRY_EXAMPLE_CONTEXT', '')
    if len(serialized.encode('utf-8')) > 16384:
        raise ValueError('Example dependency context exceeds byte ceiling')
    context = json.loads(serialized)
    if not isinstance(context, dict) or set(context) != {'paths', 'prefix', 'exec_prefix'}:
        raise ValueError('Invalid example dependency context')
    roots = context['paths']
    if not isinstance(roots, list) or not 1 <= len(roots) <= 2:
        raise ValueError('Invalid example dependency roots')
    for value in [*roots, context['prefix'], context['exec_prefix']]:
        if not isinstance(value, str) or '\x00' in value or not os.path.isabs(value):
            raise ValueError('Example dependency roots must be absolute local paths')
    sys.prefix = context['prefix']
    sys.exec_prefix = context['exec_prefix']
    sys.path.extend(roots)
    from importlib import import_module
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    target = import_module(module)
    return target.worker_main(arguments)


if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[1] != '--child':
        raise SystemExit('This module is a fixed example worker launcher')
    try:
        raise SystemExit(_child(sys.argv[2], sys.argv[3:]))
    except Exception as exc:
        print(json.dumps({'status': 'incomplete', 'error_type': type(exc).__name__, 'error': str(exc)}))
        raise SystemExit(2)
