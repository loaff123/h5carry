# Rectangular snapshot journey replay

This public, original reproducer exercises strict request → CLI plan/export/verify/inspect, followed by an output-only consumer in a fresh bounded process. It preserves the 14 original case identities, original source coordinates and independently authored literal/formula expectations. It is not a copying recipe. Existing H5Carry baseline installations cannot silently count as the candidate: every subprocess asserts the exact requested package origin.

## Portable commands

From the source tree with one qualified Linux Python environment containing the pinned dependencies:

```sh
python qualification/run_snapshots.py --out /tmp/snapshot-original-replay
python qualification/run_snapshots.py --contract-variants --out /tmp/snapshot-reference-contract
python qualification/run_snapshots.py --resources --out /tmp/snapshot-resource-replay
```

Use new output directories. Commands, complete stdout/stderr, source/output hashes, plan transformations and results are saved there. No source input is overwritten. Frozen original files can be replayed without modification with `--frozen-root DIRECTORY` containing CASE/source.h5. Add `--frozen-manifest MANIFEST.json --profile baseline` (or current) to require pre-recorded source hashes. Without these flags the shipped original fixture creator generates portable sources.

An extracted or installed artifact can be checked with `--project SOURCE_TREE --package-root DIRECTORY_CONTAINING_H5CARRY`. The explicit artifact package must exist, and its resolved import origin and production/schema hashes are recorded. This option does not select an arbitrary preinstalled baseline package by default.

For a small subset, repeat `--only-case NAME`. For the independent checker alone:

```sh
PYTHONPATH=src:. python -m tests.native_runner qualification.check_snapshot /tmp/output.h5 --case shared_coherent
```

The observer imports no product planner, scanner, writer, verifier, request resolver or block mapper, and checks that those modules are absent. It reads no source or plan. Low-level primitive types, storage/filter order/flags/parameters, normal attribute encodings, exact namespaces, aliases, ordinary refs, null refs, labels and raw/high-level forward/reverse scale edges have independent expectations. Original nonempty payloads use original-coordinate literals; resource payloads use bounded independent formulas. Empty crops and scales are checked without payload reads.

## Important original prototype discrepancy

The original matrix intentionally retains a measured red `reference_outbound` result. Its prototype observer expects both `/gain` and `/gain_alias`, but its request explicitly retains only `/gain_alias`. Under established v1 retained-path closure, ordinary references require the whole gain identity and can be remapped to the already-retained alias; they do not force another unrequested canonical source pathname. The product output contains `/gain_alias`, and its ordinary attribute and dataset refs target that same whole identity. The unchanged original checker therefore rejects its exact namespace. Do not describe the original 14-case matrix as all passing.

Two separately named contract controls are provided, without changing that original request or checker:

- `reference_outbound_retained_alias`: the exact original typed request, checked against the documented retained-path namespace (`/gain_alias` only), including whole values and actual reference identities
- `reference_outbound_explicit_both_aliases`: a strengthened request explicitly includes whole `/gain`; the unchanged original checker then establishes both aliases and their shared identity

The strict typed adapter also deduplicates identical positional scale assertions through the known `/A`/`/A_alias` identity. Both explicit alias box selections remain. The frozen original recipe request is recorded separately, and all 14 original requests are preserved exactly in the adapter's evidence. Canonical typed object/mapping list order is explicit.

## Resources and limits

The separate resource suite covers original-coordinate offset/crossing chunks, a wide-row crop, rank-32, contiguous and rank-32 empty crops, a 2-byte crop from a sparse 2^35×4 source, exactly 256 MiB selected logical payload, 128 coherent shared-scale consumers, and compressed/uncompressed selected-payload overcap refusals. The exact-max fixture is fill-heavy compressed data, not a simultaneous-all-maxima or high-entropy performance claim. Full-source hashing remains extra I/O outside the native-operation deadline.

Every native fixture/checker/mutation process is supervised by the existing test native runner. Product CLI native operations use the product supervisor. Limits precede NumPy/h5py import: 2 GiB address space, 120 s CPU, 180 s wall per native process, 1 GiB file size; dynamic HDF5 plugins are disabled. No limits are raised. A CLI export may perform several separately limited native operations; its total elapsed time is not promised to be 180 s.

For every successful journey, the output is relocated, the operation's original source pathname is renamed to a held file, and a fresh output-only checker asserts that the original pathname is unavailable. The held source remains reachable elsewhere and the original frozen file remains unchanged. This proves pathname-independent consumption and reference/scale identity integrity, not OS filesystem isolation, hostile-file safety, or absence of access to every source copy.

Existing output, source destination and source-hardlink destination protection controls must refuse without modifying bytes or publishing replacement reports. Value, alias, ref and scale mutations must be detected. Keep red logs and corrected reruns separately.
