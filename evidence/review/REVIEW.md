# Independent implementation review: H5Carry

Reviewed 2026-10-05. This review used only original, locally generated benign fixtures. Native HDF5 operations ran in bounded Linux children. No external HDF5 input or public publication was used.

## Verdict

**QUALIFIED for the bounded Linux alpha contract on the two exact runtime profiles below.** No blocking finding remains in the reviewed implementation. All reproduced findings were repaired and rechecked. This verdict covers the reviewed runtime bytes and stated original-fixture workflow; it does not authorize publication or extend the compatibility and safety claims beyond the documented profile.

## Scope and independence

I read all production modules, the structural plan schema, the design/build requirements, the current bounded profile, the CLI/documentation, and relevant tests and qualification scripts. This is a fresh implementation review, not approval inherited from the disposable feasibility prototype.

The verifier has its own namespace traversal, soft-link resolution, dependency fixed point, reference decoding and scale-relation checks. It imports neither scanner, planner nor writer closure code. It shares primitive descriptors and the same h5py/HDF5 implementation; this is algorithmic separation, not an independently implemented HDF5 parser.

I wrote an additional original source fixture, manually checked its expected output, and deliberately corrupted outputs and a persisted selection contract. Those checks supplement the implementers' tests rather than merely counting an already-green suite.

## Independently observed evidence

- The full source-tree suite passed 94 tests on each clean pinned Linux environment: Python 3.12.14 / NumPy 2.3.5 with h5py 3.14.0 / HDF5 1.14.6, and h5py 3.16.0 / HDF5 2.0.0
- The original 9-case manual/mutation fixture also passed against installed wheels on both profiles, from a test directory without a product source tree. All 12 installed production modules plus the schema matched the reviewed source bytes; the parent import check did not load h5py or NumPy
- The wheel contains the reviewed production code and schema. The source archive includes required tests, examples, qualification drivers, pinned constraints and license, with no generated HDF5 files or bytecode
- My original fixture preserves an explicitly enumerated namespace, dataset alias identity, big-endian integer values, scalar negative zero, root and group attributes, fixed strings containing embedded NUL bytes, null attributes/references, compressed/chunked reference arrays, shape/maxshape/fill/filter settings, soft-link text, shared scales and rebuilt reverse consumers
- Seven additional mutations are rejected on both environments: split alias, redirected reference with an extra equal-valued target, signed-zero change, missing root attribute, unexpected marker data, changed soft target, and a selected soft child removed from both plan and output
- The mutually consistent plan/output omission still fails because the source-derived required closure is checked separately. The fixture's full-file source fingerprint remains unchanged
- A finite selected path that traverses the same completed soft link twice resolves natively and verifies on both environments. A completed alias is not confused with an active resolution cycle
- Output-only inspection reports `inspected` and `source_equality: false`; it does not claim a source comparison
- The repaired actual-source byte-budget preflight rejects forged descriptors before opening staging on both profiles
- Pure fault probes confirm the repaired publication state after interruption immediately following each successful hard-link syscall. Existing final entries are retained; output/report publication booleans agree with observed regular-file inode identity
- The exact-byte plan ceiling roundtrips after removal of an unaccounted trailing newline. Non-string hard/reference/scale IDs are rejected as `INVALID`, rather than escaping as incidental dictionary-membership errors

Test counts are regression evidence, not a completeness metric. Raw per-case results, suite logs and the reviewed production checksum list accompany this review.

## Findings resolved during review

### Publication interruption state

Initially, export recorded `output_published` and `report_published` only after the corresponding publication call returned. Interrupting immediately after a successful output link left a real output with no reported publication state; interruption after report linking incorrectly reported that the report was absent.

The transaction now reconciles staging/final regular-file identities when publication does not finish normally, without deleting final paths. If the filesystem prevents establishing a state, it uses an explicit unknown value rather than inventing success or absence. The two original reproduction cases now report the actual states.

### Plan byte accounting and malformed identifiers

A plan exactly at its active serialized-byte ceiling was accepted and then written one byte larger due to a trailing newline. Serialization now stays within the validated ceiling. Graph endpoint types are validated before dictionary membership; the original malformed-ID cases now yield anticipated invalid-input diagnostics.

### Actual-source payload accounting

Initially, a forged plan could understate its logical payload, pass the parent codec and cause staging writes exceeding a lowered application byte ceiling. Independent final verification rejected it, so no false final success was observed. The parent codec now recomputes logical payload from descriptors, and the writer independently counts actual source dataspaces before opening or changing staging. My contradictory-total and forged-shape cases now fail as `RESOURCE` on both profiles; a preexisting staging sentinel stays byte-identical.

### Native interpreter startup

The final native launch uses `-I -S`. Resource limits are applied before native imports, and its explicit dependency context is installed afterward without processing startup hooks. The fixed product package must be importable for limit setup; native libraries are imported only after that setup. Neither `.pth` processing nor `sitecustomize` is executed in the native child. The full suites include the original startup-hook sentinel and first-native-import resource checks.

The qualified environments are clean, self-contained installations. An older environment that inherited NumPy from system-site-packages no longer works through this stricter startup route; it was replaced for qualification, not silently counted as a pass.

## Boundaries and residual risk

- This is a narrow Linux alpha for trusted, local, quiescent HDF5 files. It is not malicious-file validation, a native memory-safety sandbox, filesystem/network isolation or scientific/domain-schema validation
- Application limits are distinct from native allocations. In particular, HDF5 must parse metadata, and variable-length metadata can be allocated before its full logical payload is known. The child process memory/CPU/wall/file caps remain the containment boundary
- The current release uses lowering-only ceilings: 10,000 objects, 50,000 accounted edges, an 8 MiB plan, 256 MiB selected logical payload, 1 MiB fixed-width read blocks and 256 KiB individual attribute payload. Separate maximum-size examples do not establish that every simultaneous combination of maxima completes
- Source fingerprints detect observed change; they cannot prove quiescence against all concurrent mutation or replacement races. Whole-source hashing adds I/O. Parent-side fingerprint passes are outside the native-child deadline; the verifier also rehashes inside its bounded child. The native deadline is not a total-command deadline. Source files must remain closed to writers
- Output and report publication are individually no-clobber operations, not an atomic pair. Process interruption or a report failure can leave the verified output. Power-loss durability is not promised
- Unsupported low-level types/layouts/filters, external destinations, group hard aliases/cycles and malformed reserved metadata are rejected. These exclusions cover many otherwise valid HDF5 files
- The exact tested runtimes are recorded above. Other h5py/HDF5 builds, Python versions, architectures and operating systems are not established by this review
- The baseline and relocated examples support their stated original-fixture claims only. They establish neither general copying failure nor representative real-user adoption, speed superiority, or universal HDF5 compatibility

## Packaged qualification

I inspected successful clean wheel and extracted-source build/install records on both pinned profiles. Each of those four installed-package combinations ran all 94 tests, both relocated demo variants, and all four original fixture families from copied test/example/qualification trees containing no product `src` directory. The demos each returned 3346, and all family checks passed.

Separately, I verified every installed production module and schema in both extracted-source installations against the reviewed source hashes, and checked the imported package was in site-packages with native libraries absent from the parent import check. My independently authored manual/mutation suite also ran directly against both installed wheels, as recorded above. The broader installed-package suites and workflow records were inspected, not recounted as my own independent reruns.

Documentation-only final packaging may change archive hashes. The final receipt must verify that runtime members still match the reviewed production checksum list and record the actual final archive hashes.

## Other qualification evidence

I inspected the supplied four-family export and relocated-consumer result records and their checking code. They report successful reopened source/output checks and calibrated totals of 173 + 3173 = 3346 on both pinned profiles. The source path is moved aside for the consumer; the process is not denied access to every possible source copy. These are accurately scoped supplementary observations, not additional independent reruns by this reviewer.

The comparison credits ordinary-copy successes and limits observed scale-iteration failures to the original fixtures. It does not claim the h5copy/h5diff executables were tested when the baseline actually used h5py APIs. Dependency notices distinguish original MIT code from separately distributed runtime binaries.
