"""Test-only module execution after production containment controls."""
import json
import os
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
package_root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(package_root), str(root)]
from h5carry import worker
if Path(worker.__file__).resolve().parents[1] != package_root:
    raise RuntimeError('test bootstrap imported an unexpected product package')
from h5carry.worker import _install_limits, _prepare_native, _send_result
values = json.loads(os.environ['H5CARRY_PROCESS_LIMITS'])
_install_limits(values)
_prepare_native()
os.environ['H5CARRY_TEST_NATIVE_CHILD'] = '1'
module = sys.argv[2]
sys.argv = [module, *sys.argv[3:]]
try:
    runpy.run_module(module, run_name='__main__')
    exit_code = 0
except SystemExit as exc:
    exit_code = exc.code or 0
    if not isinstance(exit_code, int):
        print(exit_code, file=sys.stderr)
        exit_code = 1
_send_result({'ok': True, 'result': {'exit_code': exit_code}}, int(os.environ['H5CARRY_RESULT_FD']), values['result_bytes'])
