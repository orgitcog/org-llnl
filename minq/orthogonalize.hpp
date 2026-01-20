#ifndef MINQ__ORTHOGONALIZE
#define MINQ__ORTHOGONALIZE

// Copyright 2020 Lawrence Livermore National Security, LLC and other
// minq developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "overlap.hpp"

#include <mpi.h>
#include <slate/slate.hh>

#include <iostream>
namespace minq {

template <class Type>
auto orthogonalize(slate::Matrix<Type>  & wavefunction){

  int nprocs[2];
  int periods[2];
  int coords[2];
  
  auto err = MPI_Cart_get(wavefunction.mpiComm(), 2, nprocs, periods, coords);
  assert(err == 0);  
  
  auto nstates = wavefunction.m();
  auto nbs = (nstates + nprocs[0] - 1)/nprocs[0];
  
  slate::HermitianMatrix<Type> overlap(slate::Uplo::Lower, nstates, nbs, nprocs[0], nprocs[1], wavefunction.mpiComm());
  overlap.insertLocalTiles();

  minq::overlap(wavefunction, overlap);

  slate::potrf(overlap);

  auto cholesky = slate::TriangularMatrix<Type>(slate::Diag::NonUnit, overlap);

  slate::trsm(slate::Side::Left, Type(1.0), cholesky, wavefunction);

}

namespace aux {

template <class Type>
auto check_orthogonalization(slate::Matrix<Type>  & wavefunction){

  int rank;
  auto ierr = MPI_Comm_rank(wavefunction.mpiComm(), &rank);
  assert(ierr == 0);

  if(rank == 0) {
    std::cout << "Checking orthogonalization      :";
    std::cout.flush();
  }
  
  int nprocs[2];
  int periods[2];
  int coords[2];
  
  auto err = MPI_Cart_get(wavefunction.mpiComm(), 2, nprocs, periods, coords);
  assert(err == 0);  

  MPI_Barrier(wavefunction.mpiComm());
  
  auto nstates = wavefunction.m();
  auto nbs = (nstates + nprocs[0] - 1)/nprocs[0];
  
  slate::HermitianMatrix<Type> overlap(slate::Uplo::Lower, nstates, nbs, nprocs[0], nprocs[1], wavefunction.mpiComm());
  overlap.insertLocalTiles();


  for(long jj = 0; jj < overlap.nt(); jj++) {
    for(long ii = 0; ii < overlap.mt(); ii++) {
      if(not overlap.tileExists(ii, jj)) continue;
      auto tile = overlap(ii, jj);
      
      for(long jtile = 0; jtile < tile.nb(); jtile++){
        for(long itile = 0; itile < tile.mb(); itile++){
          tile.data()[itile + tile.stride()*jtile] = NAN;
        }
      }
    }
  }
 
  minq::overlap(wavefunction, overlap);

  double diff = 0.0;
  
  for(long jj = 0; jj < overlap.nt(); jj++) {
    for(long ii = 0; ii < overlap.mt(); ii++) {
      if(not overlap.tileExists(ii, jj)) continue;
      auto tile = overlap(ii, jj);
      
      for(long jtile = 0; jtile < tile.nb(); jtile++){
        for(long itile = 0; itile < tile.mb(); itile++){

          if(ii == jj and itile < jtile) continue;
          
          if(itile == jtile and ii == jj){
            diff += fabs(tile.data()[itile + tile.stride()*jtile] - 1.0);
          } else {
            diff += fabs(tile.data()[itile + tile.stride()*jtile]);
          }
          
        }
      }

    }
  }
  
  err = MPI_Allreduce(MPI_IN_PLACE, &diff, 1, MPI_DOUBLE, MPI_SUM, wavefunction.mpiComm());
  assert(err == 0);
                   
  if(diff <= 1e-10){
    if(rank == 0) std::cout << "    [   OK   ] (diff = " << diff << ") " << std::endl;
  } else {
    if(rank == 0) std::cout << "    [  FAIL  ] (diff = " << diff << ") " << std::endl;
  }
  
}

}

}

#endif
