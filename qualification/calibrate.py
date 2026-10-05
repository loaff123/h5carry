"""Original benign ceiling calibration, run only through tests.native_runner.

Examples:
  python -m tests.native_runner qualification.calibrate objects qualification/runs/new-run
  python -m tests.native_runner qualification.calibrate payload qualification/runs/new-run

Each mode gets its own complete native-process resource envelope. Synthetic
zero payloads measure bounded accounting and correctness, not real-file speed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    if os.environ.get('H5CARRY_TEST_NATIVE_CHILD') != '1':
        raise SystemExit('Run calibration through python -m tests.native_runner qualification.calibrate')
    import h5py
    import numpy as np
    from h5carry.model import Limits, canonical_json, fingerprint
    from h5carry.plan import validate_plan
    from h5carry.scan import make_plan_native
    from h5carry.write import write_staging
    from h5carry.verify import verify_export

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['objects', 'edges', 'payload', 'attribute'])
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=False)
    source, output = args.directory/'original.h5', args.directory/'output.h5'
    limits = Limits()
    started = time.monotonic()
    with h5py.File(source, 'w', track_order=True) as file:
        if args.mode == 'objects':
            for index in range(limits.max_objects - 1):
                file.create_group('g%05d' % index)
        elif args.mode == 'edges':
            file.create_dataset('references', shape=(limits.max_edges - 1,), dtype=h5py.ref_dtype)
        elif args.mode == 'payload':
            file.create_dataset('payload', shape=(limits.max_payload_bytes // 8,), dtype='f8')
        else:
            file.attrs['maximum_attribute'] = np.arange(limits.max_attribute_bytes, dtype='u1')
    built = time.monotonic()
    before = fingerprint(source)
    graph = make_plan_native(str(source), ['/'], limits)
    plan = {'format':'h5carry-plan', 'version':1, 'source':before,
            'selections':['/'], 'limits':limits.to_dict(), 'graph':graph}
    validate_plan(plan)
    planned = time.monotonic()
    output.touch()
    write_staging(str(source), graph, str(output), limits)
    written = time.monotonic()
    report = verify_export(str(source), str(output), plan, limits)
    verified = time.monotonic()
    unchanged = fingerprint(source) == before
    result = {'mode':args.mode, 'status':report['status'], 'source_unchanged':unchanged,
              'limits':limits.to_dict(), 'objects':len(graph['objects']),
              'edges':sum(len(graph[key]) for key in ('links','references','scales')),
              'logical_payload_bytes':graph['payload_bytes'], 'plan_bytes':len(canonical_json(plan)),
              'source_file_bytes':source.stat().st_size, 'output_file_bytes':output.stat().st_size,
              'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'seconds':{'fixture':built-started,'plan':planned-built,'write':written-planned,
                         'verify':verified-written,'total':verified-started},
              'runtime':{'python':sys.version.split()[0],'h5py':h5py.__version__,
                         'hdf5':h5py.version.hdf5_version,'numpy':np.__version__},
              'diagnostics':report.get('diagnostics', [])}
    (args.directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True))
    if report['status'] != 'verified' or not unchanged:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
