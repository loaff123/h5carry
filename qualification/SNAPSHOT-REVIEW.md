# 0.2.0a1 qualification and independent review

The reviewed candidate extends the existing H5Carry workflow; it is not a new project or a novel slicing algorithm. Both exact runtime profiles are listed in SNAPSHOT-QUALIFICATION.json. Reproduce the new cases with [SNAPSHOT-REPLAY.md](SNAPSHOT-REPLAY.md), and read the explicit [compatibility correction](../docs/release-0.2.0a1.md).

## Independent review outcomes

Strict request/v2-plan codec review found no blocking defect. Supplemental type controls checked 1,779 mutations. Its minor public-schema maximum-shape discrepancy and installed-test import/resource-path assumptions were corrected and independently rechecked. The v1 decoder body and v1 schema remain unchanged.

Typed native review found no blocking defect after 17 new probes per profile. These included coordinated equal-shape/empty request mutations, root self-reference closure, alias-only dependency targets, raw Boolean/fixed-string values, exact native properties, unsupported accessor behavior, no-clobber races and interruption after publication's link syscall. The verifier independently derives source closure and rectangle coordinates without importing planner/writer resolution or tiling.

The separate v1 correction received independent spec and quality approval. Five additional review methods per profile covered reference/empty chunk flags, alias-only whole exports, unselected NEVER data, output-reference pre-read refusal and unavailable native queries. Another 64 scoped reliability/codec tests ran per profile. This deliberately narrows v1 read admission while preserving its schema and closure; it is not described as entirely unchanged v1 behavior.

## Preserved disagreement and corrected controls

The original frozen snapshot matrix has six passing positives, seven expected policy refusals and one preserved negative result, on each profile. `reference_outbound`'s prototype observer demanded an extra unrequested `/gain` alias. Established closure correctly retains only requested `/gain_alias` when that path already retains the whole reference target identity. Independent review checked the actual attribute/dataset refs and both namespaces directly.

The original request/checker failure remains visible. Two separately named controls pass: the same request against the documented retained-path expectation, and an explicit request for both aliases against the original observer. Do not combine these into a claim that every original frozen positive passed unchanged.

Fourteen strict language/mapping controls refuse without plan or staging output. Value, alias and scale mutations are detected in original journeys; reference mutation is detected in the separate contract variant. All 28 frozen source hashes remain unchanged.

## Tests and native resources

Both profiles pass 208 source tests, including all 94 original tests (the version assertion and subprocess import routing are updated) and 18 v1 reliability regression tests. Installed-wheel, installed-sdist and extracted-sdist runs each use the same 208 tests with explicit package-origin and production/schema identity checks. Installed whole-object frozen families and relocated consumers are rerun. Existing 0.1.0a1 source/plan/export triples from all four original families also verify unchanged with this candidate on both profiles.

The separate resource matrix has eight successful journeys and two expected selected-payload refusals per profile. It covers offsets/crossing chunks, wide rows, rank 32, empty shapes, a tiny crop of a huge sparse logical extent, 128 shared-scale consumers and exactly 256 MiB selected data. The max-payload fixture is fill-heavy, not a high-entropy benchmark. Actual read instrumentation and pre-read native expansion controls are separate from this throughput observation. Caps were not raised; simultaneous maxima are not promised.

Native property construction experiments covered 36 reopened creation combinations and 1,248 dtype/filter combinations per profile. They exposed the chunk-option/H5Pequal gap and undefined NEVER reads. A live no-attributes hint did not persist after reopening; the mistaken negative-test precondition and correcting observation were retained, with no persisted mismatch claimed.

## Preserved deadline observation and bounded row coalescing

A later prepublication current-profile run of the unchanged 256 MiB resource fixture stopped with `RESOURCE` at the existing 60-second discovery deadline after 1,029,358 reads. The original result is preserved in `evidence/snapshots/resource-deadline-before-coalescing-current.json`. One unchanged serial observation completed all 1,048,576 reads in 57.343 seconds; it is diagnostic evidence of small-read cost and timing sensitivity, not replacement qualification or a reason to discard the failed run.

The verifier now coalesces adjacent segments within a C-order row under independently calculated logical and aggregate native-chunk caps for both inputs. The four logical passes, separate planner/verifier implementations and pre-read guards remain. On this exact fixture the deterministic read count is 262,144, with 4 KiB logical reads and at most 128 KiB native-chunk exposure below the unchanged 1 MiB cap. No cross-row batching, higher deadline, cache enabling or skipped verification pass was introduced.

Seven additional regression methods check independent coordinate order and caps, deterministic read counts, actual chunked big-endian signed-zero/NaN bytes, mutation detection and refusal before over-budget native reads. Historical 201-test logs are retained as earlier candidate observations; the final source and distribution suites contain 208 tests. Improved local completion does not establish a universal wall-time or throughput guarantee.

## Reproduction and evidence

Selected source test logs and original synthetic journey reports are in `evidence/snapshots/` in the repository/source bundle. Distributable JSON records replace only the local workspace prefix with `<workspace>`. The public evidence bundle contains path-normalized copies of selected original records, including initial negative results; original raw records are retained separately. Only documented path substitutions are applied: runtime/source identities, command statuses and assertions are unchanged. Artifact hashes and repeated-build identity checks are supplied in the separate receipt.

The wheel and sdist are built twice from fixed source with SOURCE_DATE_EPOCH. Sdist gzip/tar transport metadata is normalized deterministically; original unnormalized build bytes are retained separately. Rebuilding and reinstalling from the extracted sdist is part of qualification. No package registry publication, tag, GitHub Release or repository change is performed by these scripts.

The configured hosted workflow runs both pinned profiles across source, installed wheel, extracted sdist and installed sdist. Every surface runs the 208-test suite, original/contract snapshot journeys, resource matrix, existing copy baselines, four frozen whole-object families, two relocated demos and verification of four unchanged historical v1 artifact triples. It checks import/runtime identities, repeated distribution builds and preservation of the 112 original HDF5 fixtures plus 12 held sources. Twelve separate gate self-checks reject altered records, including resource or signal errors appended to the retained original namespace discrepancy. Hosted CI for the eventual public commit is a separate publication gate.

No verification here proves scientific validity, positional correspondence, historic initialization, source quiescence, arbitrary native-parser safety, physical layout or future HDF5 property behavior. Native process limits are not a malicious-file sanitizer. Relocation demonstrates source-path independence, not filesystem isolation.
