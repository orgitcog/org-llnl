#ifndef MINQ__RUN
#define MINQ__RUN

// Copyright 2020 Lawrence Livermore National Security, LLC and other
// minq developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "randomize.hpp"
#include "orthogonalize.hpp"
#include "subspace_diagonalization.hpp"

#include <mpi.h>
#include <slate/slate.hh>

namespace minq {

template <class Type>
void run(long nstates, long npoints, MPI_Comm comm){

  int rank;
  auto ierr = MPI_Comm_rank(comm, &rank);
  assert(ierr == 0);
  
  using matrix = slate::Matrix<Type>;
  
  int nprocs[2];
  int periods[2];
  int coords[2];

  auto err = MPI_Cart_get(comm, 2, nprocs, periods, coords);
  assert(err == 0);

  //These are the blocksizes, essentially we need one block per
  //process because of the limitation of other operations like a 3D
  //FFT
  auto nbs = (nstates + nprocs[0] - 1)/nprocs[0];
  auto nbp = (npoints + nprocs[1] - 1)/nprocs[1];

  if(rank == 0) {
    std::cout << "Allocating wave functions       :";
    std::cout.flush();
  }
  matrix wavefunction(nstates, npoints, nbs, nbp, nprocs[0], nprocs[1], comm);
  wavefunction.insertLocalTiles();
  matrix hwavefunction(nstates, npoints, nbs, nbp, nprocs[0], nprocs[1], comm);
  hwavefunction.insertLocalTiles();
  MPI_Barrier(comm);
  if(rank == 0) std::cout << "    [  DONE  ]" << std::endl;
  
  if(rank == 0) {
    std::cout << "Randomizing wave functions      :";
    std::cout.flush();
  }
  
  aux::randomize(wavefunction);
  aux::randomize(wavefunction);

  MPI_Barrier(comm);
  if(rank == 0) std::cout << "    [  DONE  ]" << std::endl;

  if(rank == 0) {
    std::cout << "Orthogonalizing wave functions  :";
    std::cout.flush();
  }

  orthogonalize(wavefunction);

  MPI_Barrier(comm);
  if(rank == 0) std::cout << "    [  DONE  ]" << std::endl;

  aux::check_orthogonalization(wavefunction);

  if(rank == 0) {
    std::cout << "Subspace diagonalizations       :";
    std::cout.flush();
  }

  subspace_diagonalization(hwavefunction, wavefunction);
  MPI_Barrier(comm);
  if(rank == 0) std::cout << "    [  DONE  ]" << std::endl;
  
}

}

#endif

