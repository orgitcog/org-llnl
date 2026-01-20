#ifndef MINQ__OVERLAP
#define MINQ__OVERLAP

// Copyright 2020 Lawrence Livermore National Security, LLC and other
// minq developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <slate/slate.hh>

namespace minq {

template <class Type>
auto overlap(slate::Matrix<Type> & wavefunction, slate::HermitianMatrix<Type> & olap){
  slate::herk(0.1, wavefunction, 0.0, olap);
}

}

#endif
