#ifndef MINQ__SUBSPACE_DIAGONALIZATION
#define MINQ__SUBSPACE_DIAGONALIZATION

// Copyright 2020 Lawrence Livermore National Security, LLC and other
// minq developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "overlap.hpp"

#include <slate/slate.hh>

#include <iostream>
namespace minq {

template <class Type>
auto subspace_diagonalization(slate::Matrix<Type>  & hwavefunction, slate::Matrix<Type>  & wavefunction){

  int nprocs[2];
  int periods[2];
  int coords[2];
  
  auto err = MPI_Cart_get(wavefunction.mpiComm(), 2, nprocs, periods, coords);
  assert(err == 0);  
  
  auto nstates = wavefunction.m();
  auto nbs = (nstates + nprocs[0] - 1)/nprocs[0];

  slate::Matrix<Type> temp(nstates, nstates, nbs, nbs, nprocs[0], nprocs[1], wavefunction.mpiComm());
  temp.insertLocalTiles();

  auto trans = conj_transpose(hwavefunction);
  
  slate::gemm(Type(0.1), wavefunction, trans, Type(0.0), temp);

  slate::HermitianMatrix<Type> hermitian_hamiltonian(slate::Uplo::Lower, temp);

  std::vector<double> eigenvalues(nstates);

  slate::heev(lapack::Job::Vec, hermitian_hamiltonian, eigenvalues, temp);

  auto wavefunction_copy = wavefunction;
  wavefunction_copy.insertLocalTiles();

  copy(wavefunction, wavefunction_copy);
  
  slate::gemm(Type(1.0), temp, wavefunction_copy, Type(0.0), wavefunction);
  
}

}


#endif
