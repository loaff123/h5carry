# Qualification and reproduction

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

Read QUALIFICATION.json for exact completed gates and runtime versions, and REVIEW.md for the independent review. The accompanying evidence archive contains raw public-safe run records and generated original fixtures. Baseline observations are limited to the four defined families; plain numeric copy controls are retained. A relocated consumer computes 173 + 3173 = 3346, with the original source pathname absent, while a copy remains held aside. This is a standalone-file demonstration, not filesystem isolation.

The artifact has not been published to a package registry or repository by these scripts. Repository CI status is a separate publication-time check.
