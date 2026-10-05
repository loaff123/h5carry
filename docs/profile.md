# Supported profile and selection semantics

H5Carry selects whole objects, never slices elements, merges files or renames objects. Selecting a group includes its descendants. Selecting a dataset name does not select every inbound alias. A dependency outside selected groups uses the first source hard path; all explicitly retained aliases remain one object. Group hard aliases and cycles are rejected. Soft-link chains must also be usable through the native reader’s default traversal limit; merely lexical termination is insufficient.

Follow simple reference values in datasets and normal attributes, internal soft-link targets and forward dimension-scale attachments until closure. A referenced group is fully selected. Add ancestor groups and root, preserve their normal attributes, and follow their references; ancestor status alone does not select siblings. Do not follow the scale convention's reverse REFERENCE_LIST to select consumers. Verify both raw forward and reverse scale relations and regenerate reverse relations from retained dataset/axis pairs.

The saved plan contains no portable HDF5 addresses: object IDs are lexicographically first retained hard paths. Export independently verifies source-derived closure against both plan and output. Removing a required child or dependency from both plan and output is a mismatch, even if those two agree.

Canonical low-level HDF5 encodings are checked, including integer precision/offset/padding/order and exact bool/complex conventions. Equality of NumPy dtype text alone is insufficient. Supported lossless filter pipeline settings, shape/maxshape/chunks and fill values are compared. Reference datasets are allocated with source creation properties before references are remapped. Fixed-width data are read in bounded C-order blocks and compared exactly.

Unknown reserved metadata is rejected, never silently dropped. Standard scale metadata is a declared graph transformation, not raw attribute-byte preservation. Scale lengths/ranks need not match consumer dimensions; HDF5 permits this and H5Carry imposes no domain rule.

This alpha deliberately rejects many valid HDF5 files. Native parsing remains trusted-input-only. Neither successful inspection nor verification establishes scientific, netCDF, CF, NWB or any other domain-schema validity.
