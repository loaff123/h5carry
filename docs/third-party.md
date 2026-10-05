# Third-party notices

Original H5Carry implementation, synthetic fixtures, tests and documentation are MIT licensed. No competitor implementation is vendored.

Runtime dependencies are distributed separately:

- h5py: BSD 3-Clause license, https://github.com/h5py/h5py/blob/master/LICENSE
- NumPy: BSD 3-Clause license, https://github.com/numpy/numpy/blob/main/LICENSE.txt
- HDF5, embedded in the tested official h5py wheels: HDF Group BSD-style license, https://github.com/HDFGroup/hdf5/blob/develop/COPYING

The NumPy and h5py wheels may bundle additional libraries with their own notices. Consult the installed wheel's license files; this package does not relicense or redistribute those binaries.

Build tooling (setuptools, wheel and build) is not imported by the runtime package. Pinned qualification constraints describe reproducibility, not vendored dependencies.
