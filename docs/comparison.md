# What the comparison establishes

H5Carry makes a complete-object selection and its dependencies inspectable, then
writes and independently verifies that declared graph. HDF5's existing copy
operations already handle many copying tasks. This project does not claim a new
copying algorithm, a general HDF5 copying defect, superior speed, or scientific
schema validation.

## Measured original-fixture comparison

Fresh runs on 2026-10-05 used these exact Linux profiles:

- Python 3.12.14, NumPy 2.3.5, h5py 3.14.0, HDF5 1.14.6
- Python 3.12.14, NumPy 2.3.5, h5py 3.16.0, HDF5 2.0.0

The comparison uses `h5py.Group.copy` with ordinary defaults and with
`expand_soft=True, expand_external=True, expand_refs=True`. There are no external
links in these original fixtures. Command-line `h5copy` and `h5diff` were not run;
the runner records their availability separately and does not equate an API run
with testing those executables.

For **each runtime**, after closing and reopening:

- All four original sources pass manually specified array, namespace, alias,
  calibration-reference and forward/reverse dimension-scale checks. The sources
  are checked again after copying and their full-file SHA-256 hashes are unchanged
- All eight copies, four ordinary and four expanded, preserve the selected
  ordinary values and the selected intensity hard alias
- All eight copies exhibit at least one scale-iteration error on these fixtures
- Expanded copies resolve the selected calibration link to the expected values.
  In the explicit-reference fixture, expanded copies retain dataset and attribute
  calibration targets and the null reference
- Ordinary copies leave the calibration soft link dangling, because its target
  lies outside the selected group. In the explicit-reference fixture, ordinary
  copying turns the non-null calibration references into null references
- No excluded marker is found among inspected named and reachable objects. Some
  copied references do not resolve, so those scans report incomplete traversal.
  These fixtures do not establish unwanted payload expansion through reverse
  references, and the report does not convert an incomplete scan into proof of
  absence
- Both separate plain-copy controls pass all seven checks: values, big-endian
  dtype, dataset/group attributes, hard alias, soft-target values and gzip/chunks.
  Expanded soft links can become hard links; this does not establish preservation
  of soft-link text

Eight files per runtime is not eight tests across both runtimes. These are small,
original benign cases, not a representative corpus or parser-security test.
No arbitrary remote HDF5 input was used. Reproduce with a new destination:

```sh
python qualification/run_baselines.py baseline-evidence
```

`baseline-results.json` records exact versions, every source check, ordinary and
expanded successes, observed errors, source/output hashes and the plain controls.
The runner has no expectation that copying must fail: it records actual outcomes.
Every native operation runs in a Linux child with 2 GiB address-space, 120-second
CPU, 180-second wall and 1 GiB per-file limits. Dynamic HDF5 plugin loading is
disabled before import. Workers start with isolated Python and site startup
disabled; dependencies must be installed directly in a self-contained virtual
environment. These controls are resource limits, not an operating
system sandbox or a guarantee about native-parser memory safety.

## Fixture definitions and supplemental demo

The checked-in `qualification/fixture-manifest.json` records original data,
selection differences and generator digest. The four frozen families share
original three-run arrays: selected values 0–11,
other values 100–111 and excluded values 200–211, each reshaped to 4 × 3. Gains
are `[1.5, 2.5, 3.5]`; time and channel scales have the same display name while
remaining distinct objects. All families include a selected intensity alias and
an internal calibration soft link.

1. Select one run with shared scales
2. Select two runs with shared scales
3. Select one run with a non-null and a null object reference, an attribute
   reference, and an unselected calibration alias
4. Select one run with an empty array, scalar negative zero, multiple scales on
   one axis, and a square array sharing a scale across two axes

Generate the frozen two-run family with:

```sh
python examples/three_run_archive.py archive.h5 --family 2
```

The **supplemental** `--demo` variant extends family 2 with explicit reference
arrays/attributes and root/ancestor metadata. It does not replace or retroactively
change any frozen family. The public walkthrough can exercise it as follows,
using an installed H5Carry package:

```sh
python examples/three_run_archive.py archive.h5 --demo
h5carry plan archive.h5 --select /runs/selected --select /runs/other --out plan.json
h5carry export archive.h5 --plan plan.json --out shared.h5 --report export-report.json
h5carry verify archive.h5 shared.h5 --plan plan.json --report verification.json
mkdir relocated source-hold
mv shared.h5 relocated/shared.h5
mv archive.h5 source-hold/held-source.bin
python examples/analyze_selected.py relocated/shared.h5 --require-demo
```

Both the frozen family-2 and supplemental variants completed the real product
plan/export/verify/inspect/relocation workflow on both exact profiles above.
All four runs produced calibrated sums 173 and 3173, totaling **3346**. The
consumer imports no H5Carry planner, writer or verifier. It opens only the
provided output, checks selected alias/calibration identities, explicit references
and nulls in the supplemental variant, shared scales, forward/reverse consumers,
ancestor metadata and excluded-marker absence, including reference-reachable
objects. Without `--require-demo`, it also accepts the frozen family-2 export.

This walkthrough makes the original source path unavailable and relocates the
output into a new directory. Source bytes are held aside, not made inaccessible
by an OS access policy. The consumer's code does not open them; this is not a
claim that its process is denied access to every possible source copy. Follow the
qualification receipt for actual end-to-end results on the packaged candidate.

The complete original-fixture and relocation sequences are reproducible with
these standard-library drivers, each writing fresh evidence directories:

```sh
python qualification/run_product_families.py --out product-family-evidence
python qualification/run_demo.py --out relocated-demo-evidence
```

`run_product_families.py` checks each source before export, invokes all four
product commands, then runs `check_frozen.py` against the reopened output. That
observer has manually specified expectations and imports no product planner,
scanner, writer or verifier. All four families passed on both profiles.

## Existing tools remain useful

- [h5py copying](https://docs.h5py.org/en/stable/high/group.html) and
  [h5copy](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__c_p__u_g.html)
  already copy objects and offer link/reference expansion. Use them for plain
  copies when their semantics match the job
- [h5diff](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__d_f__u_g.html)
  provides existing comparison support and can supplement a workflow
- [h5repack](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_t_o_o_l__r_p__u_g.html)
  supports repacking and external-link merging; self-containment alone is not
  this project's distinction
- [NCO](https://nco.sourceforge.net/nco.html) provides mature netCDF selection;
  [PyNWB export](https://pynwb.readthedocs.io/en/latest/export.html) is appropriate
  when domain-schema knowledge matters
- [SCALPEL](https://www.csl.sri.com/users/gehani/papers/eScience-2025.SCALPEL.pdf)
  addresses workload-accessed elements/chunks. H5Carry selects complete objects
  under an explicit static graph contract

The [HDF5 dimension-scale guide](https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d_s__u_g.html)
describes application-managed scale relationships. H5Carry's narrowly supported
profile and explicit reverse-consumer reconstruction address that bounded
workflow; they do not certify every valid HDF5 file or domain schema.
