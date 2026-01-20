/*BHEADER**********************************************************************
 *
 * Copyright (c) 2025, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * LLNL-CODE-2008147. All Rights reserved. See file COPYRIGHT for details.
 *
 * This file is part of matred. For more information and source code
 * availability, see https://www.github.com/LLNL/matred.
 *
 * matred is free software; you can redistribute it and/or modify it under the
 * terms of the BSD-3 license.
 *
 ***********************************************************************EHEADER*/

/**
   @file utilities.hpp
   @brief Utility functions.
*/

#ifndef __MATRED_UTILITIES_HPP
#define __MATRED_UTILITIES_HPP

#include <vector>
#include <mpi.h>

#include "_hypre_utilities.h"
#include "HYPRE_utilities.h"

using namespace std;

namespace matred
{

/// Map a certain type T to the corresponding MPI_Datatype
template <typename T>
MPI_Datatype MPI_Type()
{
   if constexpr (std::is_same_v<T, bool>)
   {
      return MPI_C_BOOL;
   }
   else if constexpr (std::is_same_v<T, double>)
   {
      return MPI_DOUBLE;
   }
}

/// Wrapper for MPI_Allreduce
template<typename T>
T AllReduce(T local_value, MPI_Op op, MPI_Comm comm)
{
   T global_value;
   MPI_Allreduce(&local_value, &global_value, 1, MPI_Type<T>(), op, comm);
   return global_value;
}

/// Wrapper for hypre memory allocation
template<typename T>
T* HypreCTAlloc(size_t num_Ts)
{
   if (num_Ts == 0) { return NULL; }
   return hypre_CTAlloc(T, num_Ts, HYPRE_MEMORY_HOST);
}

/// Generate the parallel offsets array for hypre parallel matrices
vector<HYPRE_Int> GenerateOffsets(MPI_Comm comm, HYPRE_Int loc_size);

} // namespace matred
#endif /* __MATRED_UTILITIES_HPP */
