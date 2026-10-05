"""Bounded execution of ORIGINAL local test/example modules only.

Not imported by the product. This test utility intentionally accepts a Python
module name; production run_native never accepts arbitrary code or module names.
"""
from pathlib import Path
import sys
import h5carry
from h5carry.supervisor import ProcessLimits, _run_worker


def run_module(module, args=(), process_limits=None):
    """Run a trusted test module; return {ok,result:{exit_code},_stdout,_stderr}."""
    if not isinstance(module, str) or not module or not all(part.isidentifier() for part in module.split('.')):
        raise ValueError('module must be a local dotted Python module name')
    package_root = str(Path(h5carry.__file__).resolve().parents[1])
    command = [sys.executable, '-I', '-S', '-u', str(Path(__file__).with_name('native_bootstrap.py')), package_root, module, *map(str,args)]
    return _run_worker(command, b'{}', process_limits or ProcessLimits(), capture=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m tests.native_runner MODULE [ARGS...]')
    result = run_module(sys.argv[1], sys.argv[2:])
    sys.stdout.write(result.pop('_stdout', ''))
    sys.stderr.write(result.pop('_stderr', ''))
    if not result['ok']:
        sys.stderr.write(str(result) + '\n')
        raise SystemExit(2)
    raise SystemExit(result['result']['exit_code'])
