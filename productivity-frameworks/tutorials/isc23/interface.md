# Translating MPI dynamically using Interface mode

## Learning objectives

With these hands-on exercises, participants will learn:
 - how to build and execute simple programs using Wi4MPI in interface mode
 - in particular, how to switch between different MPI implementations at runtime thanks to Wi4MPI
 - how to execute benchmarks and measure Wi4MPI performance
 - how to use Wi4MPI interface mode with Slurm

Once participants complete these exercises, they will be able to build and launch programs using Wi4MPI in their own environments.

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

### Compilation and first execution

The previous program may be compiled using Wi4MPI:

```bash
spack unload -a
spack load wi4mpi
mpicc hello.c -o hello
```

and executed:

```bash
srun -n 4 ./hello
```

Here you will get an error saying he can't find `libmpi.so`. 

### Switching MPI implementations using Wi4MPI

A program built with Wi4MPI cannot be executed by itself.
It is then possible to use OpenMPI or MPICH to run this program successfully with Wi4MPI; to do so, you'll need to add the `wi4mpi` launcher and specify the `-t` option (*to*) specifies which MPI implementation to use:

```bash
spack unload -a
spack load wi4mpi
srun --mpi=pmix -n 1 wi4mpi -t openmpi ./hello
```

or:

```bash
spack unload -a
spack load wi4mpi
srun --mpi=pmix -n 1 wi4mpi -t mpich ./hello
```

## OSU Micro Benchmarks

First, you will need to get the OSU Micro Benchmarks:

```bash
wget https://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-7.1-1.tar.gz
tar xf osu-micro-benchmarks-7.1-1.tar.gz
```

### Compiling OMB with Wi4MPI

It is straightforward to compile the OSU Micro Benchmarks and run it with MPICH or OpenMPI:

```bash
spack unload -a
spack load wi4mpi
cd osu-micro-benchmarks-7.1-1
wi4mpi -t openmpi ./configure CC=mpicc CXX=mpicxx
make -j
```

As said, a program built with WI4MPI cannot be executed by itself.

### Wi4MPI Preload mode with Slurm

Wi4MPI is able to translate MPI implementations without its own mpirun or wi4mpi wrappers.
Using the Slurm srun command directly on your executable is a powerful tool.

### Switching MPI implementations using Wi4MPI

With OpenMPI:

```bash
srun --mpi=pmix -n 4 wi4mpi -t openmpi c/mpi/startup/osu_init
srun --mpi=pmix -n 2 wi4mpi -t openmpi c/mpi/pt2pt/osu_bibw
srun --mpi=pmix -n 4 wi4mpi -t openmpi c/mpi/collective/osu_allreduce
```

or with MPICH:

```bash
srun -n 4 wi4mpi -t openmpi c/mpi/startup/osu_init
srun -n 2 wi4mpi -t openmpi c/mpi/pt2pt/osu_bibw
srun -n 4 wi4mpi -t openmpi c/mpi/collective/osu_allreduce
```

Note the performance using Wi4MPI switching from one MPI implementation to the other.

### What the `w4mpirun` wrapper really do

Here an example of what the `w4mpi` wrapper really do, if you need to set manually the variables:

```bash
spack unload -a
spack load openmpi
spack load wi4mpi

export OPENMPI_ROOT=/opt/amazon/openmpi

export LD_LIBRARY_PATH=${WI4MPI_ROOT}/lib:${LD_LIBRARY_PATH}
export WI4MPI_TO=OMPI
export WI4MPI_RUN_MPI_C_LIB=${OPENMPI_ROOT}/lib64/libmpi.so
export WI4MPI_RUN_MPI_F_LIB=${OPENMPI_ROOT}/lib64/libmpi_mpifh.so
export WI4MPI_RUN_MPIIO_C_LIB=${WI4MPI_RUN_MPI_C_LIB}
export WI4MPI_RUN_MPIIO_F_LIB=${WI4MPI_RUN_MPI_F_LIB}
export WI4MPI_WRAPPER_LIB=${WI4MPI_ROOT}/lib_${WI4MPI_TO}/libwi4mpi_${WI4MPI_TO}.so

srun --mpi=pmix -n 4 c/mpi/startup/osu_hello
```

The different `WI4MPI_*` variables are listed in the documentation.

And for MPICH:

```bash
spack unload -a
spack load mpich
spack load wi4mpi

export MPICH_ROOT=/usr/lib64/mpich

export LD_LIBRARY_PATH=${WI4MPI_ROOT}/lib:${LD_LIBRARY_PATH}
export WI4MPI_TO=MPICH
export WI4MPI_RUN_MPI_C_LIB=${MPICH_ROOT}/lib/libmpi.so
export WI4MPI_RUN_MPI_F_LIB=${MPICH_ROOT}/lib/libmpifort.so
export WI4MPI_RUN_MPIIO_C_LIB=${WI4MPI_RUN_MPI_C_LIB}
export WI4MPI_RUN_MPIIO_F_LIB=${WI4MPI_RUN_MPI_F_LIB}
export WI4MPI_WRAPPER_LIB=${WI4MPI_ROOT}/lib_${WI4MPI_TO}/libwi4mpi_${WI4MPI_TO}.so

srun -n 4 c/mpi/startup/osu_hello
```
