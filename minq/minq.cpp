// Copyright 2020 Lawrence Livermore National Security, LLC and other
// minq developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "run.hpp"

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <mpi.h>
#include <complex>

int main(int argc, char ** argv){

  int provided = 0;
  auto err = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided );
  assert( err == 0 );
  assert( provided == MPI_THREAD_MULTIPLE );

  int size = 0;
  int rank = 0;

  err = MPI_Comm_size( MPI_COMM_WORLD, &size );
  assert( err == 0 );
  err = MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  assert( err == 0 );

  if(rank == 0){  
    if(argc != 2){
      std::cerr << "Please use '" << argv[0] << " <natoms>'" << std::endl; 
      exit(1);
    }
  }
    
  long const natoms = atoi(argv[1]);

  if(rank == 0){  
    if(natoms < 0) {
      std::cerr << "The number of atoms must be positive. The value received was " << natoms << std::endl; 
      exit(1);
    }
  }

  long const nstates = 4*natoms;
  
  double const vol_per_atom = 125.0;
  double const volume = vol_per_atom*natoms;
  double const cell_size = cbrt(volume);
  double const ecut = 40.0;
  double const spacing = M_PI*sqrt(0.5/ecut);
  long const npoints = pow(ceil(cell_size/spacing), 3);

  int dims[2] = {0, 0};
  
  err = MPI_Dims_create(size, 2, dims);
  assert(err == 0);
  assert(dims[0]*dims[1] == size);

  int periods[2] = {1, 1};
  MPI_Comm cart_comm;
  err = MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
  assert(err == 0);

  if(rank == 0){

    std::cout << "minq input:" << std::endl;
    std::cout << "  natoms            = " << natoms << std::endl;    
    std::cout << "  nstates           = " << nstates << std::endl;
    std::cout << "  npoints           = " << npoints << std::endl;
    std::cout << "  wavefunction size = " << nstates*npoints*sizeof(std::complex<double>)/(1024.0*1024.0*1024.0) << " GB" << std::endl;
    std::cout << "  total procs       = " <<  size   << std::endl;
    std::cout << "  states procs      = " << dims[0] << std::endl;
    std::cout << "  points procs      = " << dims[1] << std::endl;
    std::cout << std::endl;

  }

  minq::run<std::complex<double>>(nstates, npoints, cart_comm);
  
  err = MPI_Finalize();
  assert( err == 0 );

}
