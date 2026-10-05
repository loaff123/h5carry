# H5Carry release qualification evidence

This archive records original benign-fixture qualification of H5Carry 0.1.0a1 on the exact Linux/Python/h5py/HDF5/NumPy profiles in QUALIFICATION.json. It contains no user data or external HDF5 payloads.

- original-workflows/: raw four-family ordinary/expanded-copy baselines, successful numeric controls, product outputs/plans/reports, and relocated standalone analysis
- installed-packages/: fresh wheel and extracted-sdist installations on both profiles; each runs all 94 tests plus the two relocated demos and all four product families from an asset tree without product source
- verifier/: 42 implementation acceptance scenarios per profile, including independent source-closure omission and data/graph/metadata tampering
- review/: separate review, original additional fixture/mutations, repaired boundary probes and installed-code identity evidence
- calibration/: four separate maximum-budget synthetic workloads per profile, each verified; large zero-filled generated files are reproducible with qualification/calibrate.py and omitted from this evidence archive
- resource/: actual benign resource/transport/startup/cleanup test logs
- production-hashes.json: runtime/schema hashes of the reviewed implementation
- license-audit.json: metadata/license files observed in the installed dependencies

The review/ scripts with native imports must run through the source distribution's tests.native_runner, never directly. To reproduce those supplemental scripts, copy them into a fresh source-tree review_probes/ directory and invoke, for example:

    python -m tests.native_runner review_probes.native_adversarial_suite fresh-review-output
    python -m tests.native_runner review_probes.native_budget_fixed_probe fresh-budget-output

The ordinary fault/codec probes do not import native libraries and can run directly with the package installed. All result directories should be new. Full standard reproduction commands and pinned constraints are in the accompanying source distribution's qualification/README.md.

Install logs retain their original output with local setup paths normalized to placeholders. Scientific result records, reported statuses, hashes and test counts are unchanged. Successful suite logs require no path rewriting beyond that normalization. SHA256SUMS covers every file except itself.

Limits apply to trusted native processing, not native-parser memory safety. No filesystem/network sandbox, malicious-file guarantee, general copy failure, netCDF/CF semantic promise, performance superiority, or new algorithm is established. Output/report publication is individually no-clobber, not an atomic pair. The relocated consumer has no source pathname; held source bytes were not denied by OS policy. Remote repository publication and CI are separate and were not performed here.
