# Overview

saltatlas_h5_io is a header-only library and depends on only HDF5 (C/C++, tested with HDF5 v1.10.7).

All header files are located in [./include/saltatlas_h5_io](./include/saltatlas_h5_io).

# Getting Started (on LC)

We describe how to install HDF5 and build example and test programs.

## Install HDF5

```bash
module load gcc/8.3.1
spack compiler add

# Install HDF5 v1.10.7 with C++ support and without MPI support
# using GCC v8.3.1
spack install hdf5@1.10.7%gcc@8.3.1+cxx~mpi
```

### Specify Installed HDF5 Location

There are two ways to set the path of the HDF5 to use:
using `spack load` and manually setting `CMAKE_PREFIX_PATH`.

If ones call `spack load hdf5`,
the paths of the libraries which HDF5 depends on are also put into `CMAKE_PREFIX_PATH`.
Spack installs an MPI along with HDF5 if the MPI support is enabled (enabled by default).
If ones want to use an MPI installed in another location,
we recommend the instruction in Section [2. Set CMAKE_PREFIX_PATH Manually](#2-set-cmake_prefix_path-manually).

#### 1. Use spack load

```bash
spack load hdf5
```

#### 2. Set CMAKE_PREFIX_PATH Manually

```bash
# Get the path to an installed HDF5
spack find --path hdf5

export CMAKE_PREFIX_PATH=/path/to/hdf5:${CMAKE_PREFIX_PATH}
```

## Build Example and Test Programs

saltatlas_h5_io is a header-only library.
Build is not required.
Here we describe how to build our example and test programs.

Before following the instruction below,
load the path of an installed HDF5 as described in Section [Specify Installed HDF5 Location](#specify-installed-hdf5-location).

```bash
module load gcc/8.3.1

# In a directory of this repository
mkdir build
cd build
cmake ../
make
```

# License
saltatlas_h5_io is distributed under the MIT license.

All new contributions must be made under the MIT license.

See [LICENSE-MIT](LICENSE-MIT), [NOTICE](NOTICE), and [COPYRIGHT](COPYRIGHT) for
details.

SPDX-License-Identifier: MIT

# Release
LLNL-CODE-833039
