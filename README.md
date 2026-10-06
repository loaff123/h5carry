# H5Carry

Choose whole objects or explicit rectangular snapshots from a trusted local HDF5 archive, inspect what dependencies come along, and create a new file whose data and retained graph are independently checked.

H5Carry is an alpha command-line workflow for native HDF5. It preserves selected dataset aliases, supported object references, internal soft links, normal attributes, and dimension-scale relationships. It follows forward dependencies and rebuilds reverse scale relationships from retained consumers, so a shared scale alone does not select unrequested sibling runs.

## Quick start

Use Linux, a clean self-contained Python virtual environment, and a stable source file closed to all writers. The qualified Python/runtime combinations and exact test evidence are described in [qualification](qualification/README.md). For the current pinned dependency profile:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -c qualification/constraints-current.txt .
h5carry plan archive.h5 --select /runs/A --select /runs/B --out plan.json
h5carry export archive.h5 --plan plan.json --out shared.h5 --report export.json
h5carry verify archive.h5 shared.h5 --plan plan.json --report verification.json
h5carry inspect shared.h5 --report inspection.json
```

The plan lists included objects, links, explicit references, scale edges, creation properties, normal attributes, logical data digests and reasons. Review it before exporting: an explicit reference on a selected object or ancestor can legitimately require another run. The source fingerprint binds the plan to the whole file. Plans contain retained metadata, which may itself be sensitive; treat plans and reports as scientific data.

All destinations must be new. H5Carry never overwrites an existing plan, output or report. Output and report are individually published with atomic no-clobber hard-link creation on their local filesystems. They are **not an atomic pair**. If output succeeds and report publication fails, the verified output remains and the command reports that partial state. It checks staging/final inode identity after an interrupted publication; a null publication field means filesystem state could not be established. A crash can likewise leave only the verified output; compare its fingerprint with a new `verify` report. Power-loss durability is not promised.

## Rectangular snapshots

Use `h5carry plan source.h5 --selection-json selection.json --out plan.json` for rank-preserving, bounded start/stop boxes and explicit positional scale assertions. Proper crops have fixed maximum shape and deterministic clamped chunks. Shared scales and aliases must agree on original coordinates; ordinary references require whole targets. Empty rectangles are supported. Read [the exact request and resource contract](docs/rectangular-snapshots.md) before using this mode.

Version 0.2.0a1 also intentionally narrows whole-object admission and strengthens creation-property verification after a demonstrated 0.1.0a1 omission. Nonempty `FILL_TIME_NEVER` payload reads are refused; see [the compatibility and reliability notes](docs/release-0.2.0a1.md).

## Practical example

The source distribution includes an entirely generated three-run measurement archive and a standalone consumer. No external scientific dataset is downloaded.

```sh
python examples/three_run_archive.py demo-source.h5 --demo
h5carry plan demo-source.h5 --select /runs/selected --select /runs/other --out demo-plan.json
h5carry export demo-source.h5 --plan demo-plan.json --out shared.h5 --report demo-export.json
mkdir delivery
mv shared.h5 delivery/shared.h5
mv demo-source.h5 source-held-aside.h5
python examples/analyze_selected.py delivery/shared.h5
```

The consumer reads the delivered output without a source argument. It checks retained aliases, calibration and scale relationships, excluded-marker absence, and calibrated totals 173 + 3173 = 3346. Moving the original source path is a standalone-file demonstration, **not** an OS-enforced denial of access to every source copy.

## Supported contract

- Complete groups/datasets, scalar, empty and null dataspaces; exact absolute path selections
- Canonical fixed-width integer, boolean, float32/64, complex64/128 and fixed byte-string encodings
- Numeric/fixed-string attributes, canonical variable-length text attributes and simple object-reference attributes
- Simple object-reference datasets, including null references; selected dataset hard aliases
- A tree of hard-linked groups; internal soft links resolvable with the qualified native reader’s default traversal limit; standard dimension-scale names, labels and attachment sets
- Contiguous or chunked datasets with supported deflate, shuffle and Fletcher32 creation properties
- Exact logical values, including signed zero and NaN payload bytes exposed by the qualified reader

External links anywhere in the scanned namespace, external storage, VDS, region/nested references outside recognized scale metadata, group hard aliases/cycles, compact layout, unknown filters, general compound/enum/named types, dataset variable-length data and malformed reserved metadata are rejected. Conventional h5py boolean and complex encodings are explicit narrowly checked exceptions. See [the profile](docs/profile.md).

This is not netCDF/CF validation, domain-schema validation or a guarantee of scientific completeness. Physical file layout, object addresses, timestamps, metadata creation order, file size and compression ratio are outside the contract. The verifier is a separate closure/traversal implementation; it still uses the same h5py/HDF5 parser and primitive type descriptions.

## Resource and trust boundary

Every product native open runs in a supervised Linux child, with limits applied before importing h5py. Worker startup uses isolated no-site mode; dependency directories are supplied explicitly without running .pth or sitecustomize hooks. Environments that rely on inherited system-site packages are not supported. Dynamic HDF5 plugin loading is disabled. Defaults cap address space at 2 GiB, CPU at 120 seconds, wall time at 180 seconds and each written file at 1 GiB. Unsupported platforms fail closed. The supervisor caps both protocol data and diagnostic streams and cleans up the child process group.

Application ceilings are 10,000 objects, 50,000 edges, an 8 MiB plan, 256 MiB selected logical payload, 1 MiB payload-read blocks, 256 KiB individual attributes, 64 path levels and 1,024-byte paths. Flags on `plan` and `inspect` can lower ceilings. Larger profiles are not qualified. Dataset/reference and metadata accounting are distinct from native allocation behavior.

Use **trusted, local, quiescent files only**. HDF5 parses metadata before all application checks are possible. Process limits reduce resource damage; they are not a malicious-file sanitizer, native memory-safety sandbox, filesystem isolation or network isolation. No arbitrary remote HDF5 payload was used for qualification.

Whole-file SHA-256 hashing adds full-source I/O even for a tiny selection. The parent hashes before/after native operations outside the child deadline; the verifier also rehashes inside its own child deadline. Kernel I/O latency and total command completion time are not bounded by the native deadline. CPU limits use a 120-second soft ceiling and 121-second hard ceiling; the wall timer includes child startup and protocol collection, while termination/reaping can add up to two seconds. Fingerprints detect observed changes but cannot prove absence of every concurrent-write race. Keep all writers closed and normal HDF5 locking enabled.

## Status and diagnostics

- Exit 0: planned, exported/verified, or output-only inspected successfully
- Exit 1: verified mismatch
- Exit 2: invalid/unsupported input, source change, resource stop, incomplete verification or unexpected error

JSON diagnostics distinguish `INVALID`, `UNSUPPORTED`, `SOURCE_CHANGED`, `RESOURCE`, `MISMATCH` and `ERROR`. Unexpected crashes are errors, never evidence that a file is simply unsupported. Inspection alone cannot prove equality with a source.

## Existing tools and evidence

h5py/h5copy already copy objects and expand references; h5repack, h5diff, NCO, schema-aware exporters and workload-driven subsetting tools solve important adjacent problems. H5Carry adds an inspectable, source-bound selection contract and checked reconstruction workflow. No algorithm novelty, speed advantage or broad adoption is claimed. [Comparison and scoped baseline results](docs/comparison.md) credit successful ordinary-copy controls and limit demonstrated copying failures to four original fixture families.

MIT licensed. Runtime dependencies and their licenses are documented in [third-party notices](docs/third-party.md).
