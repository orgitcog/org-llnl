#ifndef MINQ__RANDOMIZE
#define MINQ__RANDOMIZE

// Copyright 2020 Lawrence Livermore National Security, LLC and other
// minq developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <slate/slate.hh>

#include <cstdlib>
#include <complex>

namespace minq {
namespace aux {

template <class Type>
Type random();

template <>
double random(){
  return 2.0*(drand48() - 0.5);
}

template <>
std::complex<double> random(){
  return std::complex<double>(random<double>(), random<double>());
}

//Randomize the wavefunction. This is just to initialize the
//matrix and it is not really representative of an intensive
//operation in a DFT code.

template <class Type>
void randomize(slate::Matrix<Type> & matrix){

  int rank;
  auto ierr = MPI_Comm_rank(matrix.mpiComm(), &rank);
  assert(ierr == 0);

  // we don't care about the quality of the numbers, just that they are different in every processor
  srand48(rank);
  
  for (long jj = 0; jj < matrix.nt(); ++jj) {
    for (long ii = 0; ii < matrix.mt(); ++ii) {
      if(not matrix.tileExists(ii, jj)) continue;

      auto tile = matrix(ii, jj);
      
      for(long jtile = 0; jtile < tile.nb(); jtile++){
        for(long itile = 0; itile < tile.mb(); itile++){
          tile.data()[itile + tile.stride()*jtile] = random<Type>();
        }
      }
      
    }
  }
  
}

}
}

#endif
