# Qualification and reproduction

The existing QUALIFICATION.json, REVIEW.md and frozen whole-object evidence describe the historical 0.1.0a1 release. The 0.2.0a1 snapshot/reliability gates are recorded separately in SNAPSHOT-QUALIFICATION.json and SNAPSHOT-REVIEW.md. Read [the compatibility correction](../docs/release-0.2.0a1.md) before treating old creation-property claims as current guarantees.

The release profile is Linux x86-64 in clean, self-contained CPython 3.12.14 virtual environments:

- Baseline: h5py 3.14.0, embedded HDF5 1.14.6, NumPy 2.3.5
- Current tested wheel: h5py 3.16.0, embedded HDF5 2.0.0, NumPy 2.3.5

Other Python versions, operating systems, HDF5 builds and dependency combinations are unqualified. Windows/macOS native execution is refused. The observed embedded HDF5 version is recorded from each actual runtime, not inferred only from the h5py version. “Current” names the tested pinned wheel, not a promise to track future releases.

Only original benign synthetic HDF5 fixtures are executed. No remote HDF5 input, malicious parser corpus, real user data, competitor implementation or external test suite is used. The independent verifier shares primitive descriptors and the native parser with the exporter; it independently traverses links, references and dimension-scale relationships and derives source selection closure.

## Reproduce from source

Create a fresh directory for every generated run. Install one of the pinned profiles in its own clean virtual environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -c qualification/constraints-current.txt build setuptools wheel h5py numpy
python -m pip install --no-deps --no-build-isolation -e .
python -m tests.native_runner unittest discover -s tests -v
python qualification/run_baselines.py qualification/runs/baselines
python qualification/run_product_families.py --out qualification/runs/frozen-product
python qualification/run_demo.py --out qualification/runs/demo
python qualification/run_snapshots.py --help
python -m tests.native_runner qualification.calibrate objects qualification/runs/objects
python -m tests.native_runner qualification.calibrate edges qualification/runs/edges
python -m tests.native_runner qualification.calibrate payload qualification/runs/payload
python -m tests.native_runner qualification.calibrate attribute qualification/runs/attribute
python -m build --no-isolation
```

Use constraints-baseline.txt in a separate environment to reproduce the baseline profile. Native test fixtures refuse unbounded direct execution; use the test runner shown above. Pure model/parser/publication tests are separate from actual native integration tests. Publication-fault tests inject interruptions or races around real filesystem operations; they do not count a mocked native result as scientific preservation evidence.

The resource tests exercise actual benign allocation denial, CPU/wall stops, file-size failure, signal crashes, protocol truncation/invalid records, stream caps, startup-hook suppression and process-group cleanup. No security or arbitrary-malicious-file safety claim follows from these mechanism tests. A userspace deadline cannot bound uninterruptible kernel I/O or prove native parser memory safety.

## Ceiling calibration

Separate workloads reach the release ceiling for objects (10,000), edges (50,000), selected logical bytes (256 MiB), and one attribute (256 KiB). Each performs plan, reconstruction and independent verification under the native process limits. They are not simultaneous maxima, benchmark claims or representative real-user performance. The serialized 8 MiB plan and parent JSON allocation/depth guards remain additional independent limits.

## Clean package checks

The release evidence includes wheel installation and extracted-sdist build/installation in fresh environments, with tests run from copied test/example/qualification trees that contain no product src directory. The imported package path is checked to be the installed wheel with `python qualification/check_install.py`. The standalone workflow is then rerun with installed commands. Source-tree tests alone are not counted as an installed-package pass.

## What the evidence establishes

Read QUALIFICATION.json for exact completed gates and runtime versions, and REVIEW.md for the independent review. For 0.2.0a1, the accompanying evidence contains selected path-normalized public records and original generated fixtures; historical 0.1.0a1 records remain separately retained. Baseline observations are limited to the four defined families; plain numeric copy controls are retained. A relocated consumer computes 173 + 3173 = 3346, with the original source pathname absent, while a copy remains held aside. This is a standalone-file demonstration, not filesystem isolation.

The artifact has not been published to a package registry or repository by these scripts. Repository CI status is a separate publication-time check.

## Hosted and portable full qualification

The GitHub workflow runs the same portable gate in both pinned profiles:

```sh
python qualification/check_ci_gate.py
python qualification/run_ci.py --profile current --out /tmp/h5carry-ci-current
```

Use a self-contained source-profile virtual environment with the constraints above,
and a new output directory outside input trees (or under `qualification/runs/`). Run from the complete repository checkout, including its retained `evidence/` fixtures and manifest. The driver builds twice in separate clean copies,
checks identical wheel bytes and normalized sdist transport bytes, preserves the
original unnormalized sdist bytes, and checks production/schema byte identity and
retention of every test, fixture definition and qualification input in the sdist.
Only archive timestamps and owner metadata are normalized; file contents are not.

All four surfaces (source, installed wheel, extracted sdist, installed sdist) run
exactly 208 product tests, the original 14-case snapshot matrix, the two separate
reference-contract variants, the 10 resource cases, four frozen whole-object
families, two relocated demos and the original h5py copy baselines. Interpreter,
resolved package origin, actual NumPy/h5py/HDF5 versions and native limits are
checked. Installed tests run outside the source tree with no product `src`
directory. Original snapshot source bytes are retained and reused unchanged for
subsequent artifact replays. All 112 retained historical HDF5 fixtures and 12 held sources must match the original public manifest before and after qualification. Each surface also verifies the same four unchanged historical baseline v1 plan/source/output triples. Native process limits remain unchanged.

The original matrix must still return exit 1 with exactly six successes, seven
expected refusals and the sole `reference_outbound` namespace discrepancy, plus
three detected mutations and 14 language refusals. Its raw failed result and
observer traceback remain intact. A different failure, stage, case identity,
control set or count fails qualification. The separate variants require two
successes and one detected reference mutation; resources require eight successes
and two selected-payload refusals. The overall receipt says
`passed-with-retained-known-negative`, never that the original matrix passed.

The gate's 12 standard-library self-checks are separate from the 208 product tests.
The driver normally creates clean wheel/sdist virtual environments and installs
the pinned dependencies. To reuse prepared, self-contained environments locally,
pass `--wheel-python /absolute/wheel-venv/bin/python` and
`--sdist-python /absolute/sdist-venv/bin/python`; only the locally built package is
then reinstalled, without dependency resolution or index access.

The output's `evidence/` contains complete command stdout/stderr, raw JSON,
original synthetic fixtures, outputs, distributions, checksums and the final
receipt. Its `work/` contains disposable build copies and environments. Hosted CI
uploads `evidence/` even after a failed gate, with a 30-day retention period.
Download it before expiration if a durable release record is needed; the presence
of this workflow is not evidence that a particular hosted commit has passed.
