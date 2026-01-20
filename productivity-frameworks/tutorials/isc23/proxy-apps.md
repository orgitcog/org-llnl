# Running Proxy Apps with Wi4MPI

## Learning objectives

With these hands-on exercises, participants will learn:
 - how to switch between MPI implementation at runtime thanks to Wi4MPI on a real HPC application
 - how to execute benchmarks and measure performance with a common HPC benchmark

## GROMACS

[GROMACS](https://www.gromacs.org/) is a versatile package to perform molecular dynamics, i.e. simulate the Newtonian equations of motion for systems with hundreds to millions of particles. It is primarily designed for biochemical molecules like proteins, lipids and nucleic acids that have a lot of complicated bonded interactions, but since GROMACS is extremely fast at calculating the nonbonded interactions (that usually dominate simulations) it is also used for research on non-biological systems, e.g. polymers and fluid dynamics.

GROMACS is a molecular dynamics application designed to simulate Newtonian equations of motion for systems with hundreds to millions of particles. GROMACS is designed to simulate biochemical molecules like proteins, lipids, and nucleic acids that have a lot of complicated bonded interactions.

### What is GROMACS?

[GROMACS](https://www.gromacs.org/) is a versatile package to perform molecular dynamics, i.e. simulate the Newtonian equations of motion for systems with hundreds to millions of particles. It is primarily designed for biochemical molecules like proteins, lipids and nucleic acids that have a lot of complicated bonded interactions, but since GROMACS is extremely fast at calculating the nonbonded interactions (that usually dominate simulations) it is also used for research on non-biological systems, e.g. polymers and fluid dynamics.

GROMACS is a molecular dynamics application designed to simulate Newtonian equations of motion for systems with hundreds to millions of particles. GROMACS is designed to simulate biochemical molecules like proteins, lipids, and nucleic acids that have a lot of complicated bonded interactions.

### Obtaining the benchmark

```
wget https://www.mpinat.mpg.de/benchMEM
unzip benchMEM
```

This benchmark simulates a membrane channel protein embedded in a lipid bilayer surrounded by water and ions. With its size of ~80 000 atoms, it serves as a prototypical example for a large class of setups used to study all kinds of membrane-embedded proteins.For some more information see here.

This benchmark is attributed to the Dept. of Theoretical and Computational Biophysics, Max Planck Institute for Multidisciplinary Sciences, Göttingen, https://www.mpinat.mpg.de/grubmueller/bench.

### First execution

You can load the default GROMACS version installed in your environment and launch it:

 - load GROMACS:
```
spack load gromacs
```
 - launch the test case:
```
export OMP_NUM_THREADS=2
srun -n 32 -c 2 gmx_mpi mdrun -v -resethway -nsteps 10000 -ntomp ${OMP_NUM_THREADS} -s benchMEM.tpr
```

There are some parameters can be modified launching GROMACS:
 - `-nsteps` specifies the number of steps and can be used to run more or less long benchmark
 - `-resethway` resets the performance timers halfway through the run, removing the overhead of initialization and load balancing from the timings
 - `-ntomp` defines the number of OpenMP threads used

You can expect an output that looks like this:
```
10000 steps,     20.0 ps.
step 0
step 100, remaining wall clock time:    25 s          
step 200, remaining wall clock time:    24 s          
[...]
step 9800, remaining wall clock time:     0 s          
step 9900, remaining wall clock time:     0 s          
vol 0.70  imb F  1% pme/F 0.70 
step 10000, remaining wall clock time:     0 s          

               Core t (s)   Wall t (s)        (%)
       Time:      721.804       11.279     6399.6
                 (ns/day)    (hour/ns)
Performance:       76.618        0.313
```

You can note the performance obtained, in ns/day.

### Switching MPI with Wi4MPI preload mode

You can now easily switch between MPI versions with Wi4MPI, for example switching to Open MPI:

```bash
spack unload -a
spack load openmpi
spack load wi4mpi
spack load gromacs

export OPENMPI_ROOT=/opt/amazon/openmpi

export WI4MPI_FROM=MPICH
export WI4MPI_TO=OMPI
export WI4MPI_RUN_MPI_C_LIB=${OPENMPI_ROOT}/lib64/libmpi.so
export WI4MPI_RUN_MPI_F_LIB=${OPENMPI_ROOT}/lib64/libmpi_mpifh.so
export WI4MPI_RUN_MPIIO_C_LIB=${WI4MPI_RUN_MPI_C_LIB}
export WI4MPI_RUN_MPIIO_F_LIB=${WI4MPI_RUN_MPI_F_LIB}
export LD_PRELOAD=${WI4MPI_ROOT}/libexec/wi4mpi/libwi4mpi_${WI4MPI_FROM}_${WI4MPI_TO}.so:${WI4MPI_RUN_MPI_C_LIB}

```

And then run the app:

```bash
export OMP_NUM_THREADS=2
srun -n 32 -c 2 gmx_mpi mdrun -v -resethway -nsteps 10000 -ntomp ${OMP_NUM_THREADS} -s benchMEM.tpr
```

You can note the performance, in ns/day.

### Compiling with WI4MPI interface

You can install Gromacs with Wi4MPI:

```bash
wget https://ftp.gromacs.org/gromacs/gromacs-2022.5.tar.gz
tar xvf gromacs-2022.5.tar.gz
cd gromacs-2022.5
mkdir build
cd build
spack load wi4mpi
srun -n 1 -c 1 -- cmake -DCMAKE_INSTALL_PREFIX=${HOME}/gromacs/install -DGMX_MPI=on -DGMX_BUILD_OWN_FFTW=ON -DMPI_CXX_COMPILER=mpicxx -DCMAKE_CXX_COMPILER=mpicxx ..
srun -n 1 -c 4 make -j 4
srun -n 1 -c 4 make -j 4 install
```
Note that we have to compile on compute node to get the best optimization.
And then run the app with Openmpi:

```bash
spack load wi4mpi
export PATH=${HOME}/gromacs/install/bin:$PATH

export OPENMPI_ROOT=/opt/amazon/openmpi
export LD_LIBRARY_PATH=${WI4MPI_ROOT}/lib:${LD_LIBRARY_PATH}
export WI4MPI_TO=OMPI
export WI4MPI_RUN_MPI_C_LIB=${OPENMPI_ROOT}/lib64/libmpi.so
export WI4MPI_RUN_MPI_F_LIB=${OPENMPI_ROOT}/lib64/libmpi_mpifh.so
export WI4MPI_RUN_MPIIO_C_LIB=${WI4MPI_RUN_MPI_C_LIB}
export WI4MPI_RUN_MPIIO_F_LIB=${WI4MPI_RUN_MPI_F_LIB}
export WI4MPI_WRAPPER_LIB=${WI4MPI_ROOT}/lib_${WI4MPI_TO}/libwi4mpi_${WI4MPI_TO}.so

export OMP_NUM_THREADS=2
srun -n 2 -c 2 gmx_mpi mdrun -v -resethway -nsteps 1000 -ntomp ${OMP_NUM_THREADS} -s benchMEM.tpr

```

Or with MPICH:

```bash
spack load wi4mpi
export PATH=${HOME}/gromacs/install/bin:$PATH

export MPICH_ROOT=/usr/lib64/mpich
export LD_LIBRARY_PATH=${WI4MPI_ROOT}/lib:${LD_LIBRARY_PATH}
export WI4MPI_TO=MPICH
export WI4MPI_RUN_MPI_C_LIB=${MPICH_ROOT}/lib/libmpi.so
export WI4MPI_RUN_MPI_F_LIB=${MPICH_ROOT}/lib/libmpifort.so
export WI4MPI_RUN_MPIIO_C_LIB=${WI4MPI_RUN_MPI_C_LIB}
export WI4MPI_RUN_MPIIO_F_LIB=${WI4MPI_RUN_MPI_F_LIB}
export WI4MPI_WRAPPER_LIB=${WI4MPI_ROOT}/lib_${WI4MPI_TO}/libwi4mpi_${WI4MPI_TO}.so

export OMP_NUM_THREADS=2
srun -n 2 -c 2 gmx_mpi mdrun -v -resethway -nsteps 1000 -ntomp ${OMP_NUM_THREADS} -s benchMEM.tpr
```

## AMG

Algebraic MultiGrid (AMG) is an LLNL proxy app that solves systems of equations using data
decomposition. It is an SPMD code which uses MPI and OpenMP threading within MPI ranks/tasks.

It's a highly-synchronous, strong-scaling code demonstrating the surface-to-volume relationship
common in many HPC codes. Hence, the common use case is as follows:

```
amg -P <Px> <Py> <Pz> : define processor topology per part
                        Note that for test problem 1, which has 8 parts
			this leads to 8*Px*Py*Pz MPI processes!
			For all other test problems, the total amount of
			MPI processes is Px*Py*Pz.
```

Let's run it:

```bash
spack load amg
export OMP_NUM_THREADS=2
srun --mpi=pmi2 -N 2 -n 32 amg -P 4 4 2
```

Wait a second! What MPI did Spack even use to build this?!
Well, we can run:

```bash
spack find -l amg
```

to retrieve the Spack short hash of that installed package.  
And then we can run:

```bash
spack graph /mok3c2u
```

to see what it was built with. (Your hash and MPI may vary).

Let's run it with a different MPI without rebuilding, for example, to go from MPICH to OpenMPI:

```bash
spack load wi4mpi
export OPENMPI_ROOT=/opt/amazon/openmpi
export LD_LIBRARY_PATH=${WI4MPI_ROOT}/lib:${LD_LIBRARY_PATH}
export WI4MPI_TO=OMPI
export WI4MPI_RUN_MPI_C_LIB=${OPENMPI_ROOT}/lib64/libmpi.so
export WI4MPI_RUN_MPI_F_LIB=${OPENMPI_ROOT}/lib64/libmpi_mpifh.so
export WI4MPI_RUN_MPIIO_C_LIB=${WI4MPI_RUN_MPI_C_LIB}
export WI4MPI_RUN_MPIIO_F_LIB=${WI4MPI_RUN_MPI_F_LIB}
export WI4MPI_WRAPPER_LIB=${WI4MPI_ROOT}/lib_${WI4MPI_TO}/libwi4mpi_${WI4MPI_TO}.so
srun --mpi=pmix -N 2 -n 32 amg -P 4 4 2
```

## Quicksilver

Quicksilver is a Monte Carlo code using both MPI and OpenMP. It contains latency-bound table
lookups and poor vectorization potential.

### Translating the Benchmark

Again, we can leave wi4mpi loaded with the same settings to translate the application without
rebuilding.

```bash
spack load quicksilver
spack cd -i quicksilver
cd Examples/NoFission
srun --mpi=pmix -N 2 -n 32 qs -i noFission.inp
```

And finally, unsetting all the above environment variables would give the original quicksilver.
