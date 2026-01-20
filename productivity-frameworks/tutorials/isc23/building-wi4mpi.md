# How to install Wi4MPI

## Learning objectives

With these hands-on exercises, participants will learn:
 - how to configure their environment in preparation for Wi4MPI installation
 - how to install Wi4MPI using the Spack package manager
 - how to configure Wi4MPI in their environment to efficiently use the library

Once participants complete these exercises, they will have a working Wi4MPI environment ready to use.

## Installing Wi4MPI using Spack

Wi4MPI is available through Spack and the Spack package is updated with each new version.

### (For Reference) Setup

To install Wi4MPI, one needs to first install a compiler supporting C, C++ and Fortran, given that these compilers are needed for MPI compilation. For example:

 - Debian-like distro:
 
```bash
(sudo) % apt install gcc-11 g++-11 gfortran-11
```

 - RHEL-like distro:
 
```bash
(sudo) % yum install gcc gcc-gc++ gcc-gfortran
```

### Spack installation

Spack can be installed and configured using the compilers previously installed. In general, this would
be done as follows:

```bash
git clone https://github.com/spack/spack
source spack/share/spack/setup-env.sh
```
However, since we have already cloned Spack for you, simply run:

```bash
. /home/tutorial/spack/share/spack/setup-env.sh
```

Now we're ready to create a Spack environment and tell Spack where to find things:

```bash
mkdir myenv
cd myenv
spack env create -d .
spack env activate .
cd ~
spack compiler find
spack external find
module load openmpi
spack external find openmpi
module unload openmpi
module load mpi/mpich-x86_64
spack external find mpich
module unload mpi/mpich-x86_64
```

Unfortunately there is one dependency (hwloc) which we will need to add manually. We can do so
as follows:

```bash
spack cd --env
cat << EOF >> spack.yaml
    hwloc:
      externals:
      - spec: hwloc@2.2.0
        prefix: /usr
EOF
cd ~
```

Once Spack is configured, you can install Wi4MPI and the benchmarks and proxy applications to use
later:

```bash
spack add wi4mpi amg quicksilver gromacs osu-micro-benchmarks^mpich
srun -N 1 spack install
```

Note: We're using Spack Environments here because it will simulate the steps necessary if you
were to do this on your own cluster without cloning Spack dozens of times. We still recommend
using environments with Spack to keep builds reproducible and separate.

### (For Reference) Wi4MPI configuration

Once Wi4MPI and the different MPI implementations are installed, Wi4MPI needs to be configured. To do so, the paths to the available MPI implementations need to be specified:

```bash
spack load wi4mpi
echo 'OPENMPI_DEFAULT_ROOT="/opt/amazon/openmpi"' >> $WI4MPI_ROOT/etc/wi4mpi.cfg
echo 'MPICH_DEFAULT_ROOT="/usr/lib64/mpich"' >> $WI4MPI_ROOT/etc/wi4mpi.cfg
```

The above paths need to be updated with the actual paths on your system.

By the way, Spack has some neat features that allow you to find where things were installed...  
For example, if you want to go to the OpenMPI prefix, you can just use `spack cd -i openmpi`.

However, we have already done this for you on this system, so the above doesn't need to be run here.
