# 0.2.0a1 compatibility and reliability changes

This alpha adds explicit rectangular snapshots as a new strict typed request and plan version 2. Version 1 remains structurally readable, with the same whole-object plan schema and closure behavior. The minor-version increment marks a new public workflow and the conservative admission changes below.

## Deliberate v1 reliability narrowing

An independent audit of 0.1.0a1 found two precise gaps on both qualified native builds:

1. Nonempty datasets with `FILL_TIME_NEVER` and unallocated storage were accepted and reported source-equal even though those source bytes are undefined. This did not establish corruption of defined values; the verifier was making an unsupported equality claim over unspecified bytes
2. A fully populated source/output pair differing only in fill-time policy was reported verified. The old graph descriptor omitted that DCPL property, so equal data and recorded descriptors did not establish complete creation-policy equality

Version 0.2.0a1 refuses nonempty `FILL_TIME_NEVER` dataset payload reads, including fully populated ones. Reliable proof that every logical element was initialized is outside scope. Null and empty data do not need payload reads and remain supported. Metadata-only inspection while considering an empty typed crop does not itself read the undefined payload.

Source-bound verification now compares each retained dataset's low-level type and full creation property list. Native chunk options are read and compared separately because H5Pequal ignores that property on the two tested builds. A missing/failed qualified property query refuses rather than claiming verification. These are deliberate compatibility narrowings: an old v1 plan may load correctly but its operation can now refuse.

The original positive fixtures, graph semantics and supported type exclusions remain. The precise prior guarantee about omitted creation properties and undefined reads is superseded; this does not invalidate all old fixture results. Raw historical 0.1.0a1 qualification records remain preserved and are identified as historical. New synthetic regression controls distinguish original failures from corrected behavior.

The supported native query remains specific to CPython 3.12.14 / NumPy 2.3.5 with h5py 3.14.0–HDF5 1.14.6 or h5py 3.16.0–HDF5 2.0.0 on the qualified Linux environment. Other builds are not inferred correct from the package dependency range.
