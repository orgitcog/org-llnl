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
   @file utilities.cpp
   @brief Utility functions.
*/

#include <assert.h>

#include "utilities.hpp"

namespace matred
{

vector<HYPRE_Int> GenerateOffsets(MPI_Comm comm, HYPRE_Int loc_size)
{
   int num_procs, my_offsets_start(0);
   MPI_Comm_size(comm, &num_procs);

   vector<HYPRE_Int> offsets(HYPRE_AssumedPartitionCheck() ? 3 : num_procs+1);
   if (HYPRE_AssumedPartitionCheck())
   {
      MPI_Scan(&loc_size, &offsets[2], 1, HYPRE_MPI_INT, MPI_SUM, comm);
      offsets[0] = offsets[2] - loc_size;
      offsets[1] = offsets[2];
      MPI_Bcast(&offsets[2], 1, HYPRE_MPI_INT, num_procs-1, comm);
   }
   else
   {
      MPI_Allgather(&loc_size, 1, HYPRE_MPI_INT, &offsets[1], 1, HYPRE_MPI_INT, comm);
      for (int j = 0; j < num_procs; j++) { offsets[j+1] += offsets[j]; }
      MPI_Comm_rank(comm, &my_offsets_start);
   }

   // Check for overflow in offsets (TODO: define our own verify)
   assert(offsets[my_offsets_start] >= 0 && offsets[my_offsets_start+1] >= 0);
   return offsets;
}

} // namespace matred
