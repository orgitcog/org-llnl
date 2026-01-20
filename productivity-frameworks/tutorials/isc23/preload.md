# Translating MPI dynamically using Preload mode

## Learning objectives

With these hands-on exercises, participants will learn:
 - how to execute simple programs using Wi4MPI in preload mode
 - in particular, how to switch between different MPI implementations at runtime thanks to Wi4MPI
 - how to execute benchmarks and measure Wi4MPI performance
 - how to use Wi4MPI preload mode with Slurm

With these exercises, participants will also observe ABI incompatibility directly.

Once participants complete these exercises, they will be able to launch programs using Wi4MPI in their own environments.

## Running MPI "Hello world"

We will use a simple "Hello world!" program written in C:

```C
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    char library_version[MPI_MAX_LIBRARY_VERSION_STRING];
    MPI_Get_library_version(library_version, &name_len);
 
    printf("Hello world from processor %s, rank %d out of %d processors with MPI library %s\n",
           processor_name, rank, size, library_version);

    MPI_Finalize();
}

```

Note the `MPI_Get_library_version` function which will give the real MPI library used.

### Compilation and first execution

The previous program may be compiled using MPICH:

```bash
spack unload -a
spack load mpich
mpicc hello.c -o hello
```

and executed:

```bash
srun --mpi=pmi2 -n 4 ./hello
```

### Switching MPI implementations using Wi4MPI

You can check that ABI-compatibility is still an issue (!):

```bash
spack unload -a
chrpath -c hello
module load openmpi
export LD_PRELOAD=/opt/amazon/openmpi/lib64/libmpi.so
srun --mpi=pmix -n 4 ./hello
unset LD_PRELOAD
```

It is then possible to use OpenMPI to run this program successfully with Wi4MPI; to do so, you'll need to specify the `-F` and `-T` options to `mpirun`. `-F` (*from*) specifies which MPI implementation was used to compiled the program whereas `-T` (*to*) specifies which MPI implementation to use:

```bash
spack unload -a
spack load wi4mpi
srun --mpi=pmix -n 4 wi4mpi -f mpich -t openmpi ./hello
```

## OSU Micro Benchmarks

Now, let's load and use the OSU Micro Benchmarks:

```bash
spack unload -a
spack load osu-micro-benchmarks
```

### First execution

Several benchmarks may be executed to assert the usability and overhead of Wi4MPI:

```bash
srun --mpi=pmi2 -n 4 osu_init
srun --mpi=pmi2 -N 2 osu_bibw
srun --mpi=pmi2 -N 2 -n 4 osu_allreduce
```

You can note the results as reference for the next execution using Wi4MPI.

```bash
# OSU MPI Init Test v5.6.2
nprocs: 4, min: 67 ms, max: 67 ms, avg: 67 ms

# OSU MPI Bi-Directional Bandwidth Test v5.6.2
# Size      Bandwidth (MB/s)
1                       2.10
2                       5.52
4                      12.39
8                      28.51

# OSU MPI Allreduce Latency Test v5.6.2
# Size       Avg Latency(us)
4                       1.40
8                       1.08
16                      1.18
32                      0.92
```

### Wi4MPI Preload mode with Slurm

Wi4MPI is able to translate MPI implementations without its own mpirun or wi4mpi wrappers.
Using the Slurm srun command directly on your executable is a powerful tool.

### Switching MPI implementations using Wi4MPI

```bash
spack unload -a
spack load osu-micro-benchmarks
spack load wi4mpi

srun --mpi=pmix -n 4 wi4mpi -f mpich -t openmpi osu_init
srun --mpi=pmix -n 2 wi4mpi -f mpich -t openmpi osu_bibw
srun --mpi=pmix -n 4 wi4mpi -f mpich -t openmpi osu_allreduce
```

Note the performance using Wi4MPI switching from one MPI implementation to the other.

```bash
You are using Wi4MPI-3.6.0 with the mode preload From MPICH To OMPI

# OSU MPI Init Test v5.6.2
nprocs: 4, min: 540 ms, max: 542 ms, avg: 541 ms

# OSU MPI Bi-Directional Bandwidth Test v5.6.2
# Size      Bandwidth (MB/s)
1                       2.55
2                       5.03
4                      12.76
8                      29.26

# OSU MPI Allreduce Latency Test v5.6.2
# Size       Avg Latency(us)
4                       0.85
8                       0.62
16                      0.65
32                      0.66
64                      0.67
```

### What the `wi4mpi` wrapper really do

The following lines show you how to run it with OpenMPI via srun without the `wi4mpi` wrapper:

```bash
spack unload -a
spack load osu-micro-benchmarks
spack load wi4mpi

export OPENMPI_ROOT=/opt/amazon/openmpi/

export WI4MPI_FROM=MPICH
export WI4MPI_TO=OMPI
export WI4MPI_RUN_MPI_C_LIB=${OPENMPI_ROOT}/lib64/libmpi.so
export WI4MPI_RUN_MPI_F_LIB=${OPENMPI_ROOT}/lib64/libmpi_mpifh.so
export WI4MPI_RUN_MPIIO_C_LIB=${WI4MPI_RUN_MPI_C_LIB}
export WI4MPI_RUN_MPIIO_F_LIB=${WI4MPI_RUN_MPI_F_LIB}
export LD_PRELOAD=${WI4MPI_ROOT}/libexec/wi4mpi/libwi4mpi_${WI4MPI_FROM}_${WI4MPI_TO}.so:${WI4MPI_RUN_MPI_C_LIB}

srun --mpi=pmix -n 4 osu_hello
```

The different `WI4MPI_*` variables are listed in the documentation.

If the translation works, you should have this kind of output:

```bash
You are using Wi4MPI-3.6.4 with the mode preload From MPICH To OMPI
# OSU MPI Hello World Test v7.0
This is a test with 4 processes
```

### (For Reference) From OpenMPI to MPICH by srun

To install a version with OpenMPI:

Without getting into too many details about the Spack concretizer, Spack prefers to only have a
single MPI in an environment. Therefore, if you simply attempt to add the OSU Micro Benchmarks to
your environment, built with OpenMPI, you'll hit errors because our existing environment so far
has everything built with MPICH. Let's quickly override that behavior:

```bash
spack cd --env
sed -i 's/unify: true/unify: when_possible/' spack.yaml
sed -i 's/view: true/view: false/' spack.yaml
cd ~
```

We have now told Spack to allow building the same package twice in our environment with two
different MPIs, and since the default filesystem view would give these two installations the same
location, we have disabled filesystem views for the time being as well.

Now, we're ready to install the new package in our environment.

```bash
spack add osu-micro-benchmarks ^openmpi
srun -N 1 spack install
```

With an OMB suite compiled with OpenMPI, the following lines show you how to run it with MPICH via srun.

```bash
spack unload -a
spack load osu-micro-benchmarks ^openmpi
spack load wi4mpi

export MPICH_ROOT=/usr/lib64/mpich

export WI4MPI_FROM=OMPI
export WI4MPI_TO=MPICH
export WI4MPI_RUN_MPI_C_LIB=${MPICH_ROOT}/lib/libmpi.so
export WI4MPI_RUN_MPI_F_LIB=${MPICH_ROOT}/lib/libmpifort.so
export WI4MPI_RUN_MPIIO_C_LIB=${WI4MPI_RUN_MPI_C_LIB}
export WI4MPI_RUN_MPIIO_F_LIB=${WI4MPI_RUN_MPI_F_LIB}
export LD_PRELOAD=${WI4MPI_ROOT}/libexec/wi4mpi/libwi4mpi_${WI4MPI_FROM}_${WI4MPI_TO}.so:${WI4MPI_RUN_MPI_C_LIB}

srun --mpi=pmi2 -n 4 osu_hello
```

The different `WI4MPI_*` variables are listed in the documentation.

If the translation works, you should have this kind of output:

```bash
You are using Wi4MPI-3.6.4 with the mode preload From OMPI To MPICH
# OSU MPI Hello World Test v7.0
This is a test with 4 processes
```
