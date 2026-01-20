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
   @file redistributor.cpp
   @brief Core functions that implement the redistribution
*/

#include <map>
#include <algorithm>
#include <numeric>

#include "utilities.hpp"
#include "redistributor.hpp"

using namespace std;

namespace matred
{

ParMatrix BuildEntityToTrueEntity(const ParMatrix& entity_trueEntity_entity)
{
   hypre_ParCSRMatrix* e_te_e = entity_trueEntity_entity;
   HYPRE_Int* e_te_e_offd_i = e_te_e->offd->i;
   HYPRE_Int owned_e_max_global_id = e_te_e->last_row_index;

   // TODO: find minimum of all shared entities if e_te_e->offd->j is not sorted
   auto SharedEntitiesMinGlobalID = [&](bool entity_is_shared, int offset)
   {
      return entity_is_shared ? e_te_e->col_map_offd[e_te_e->offd->j[offset]] : 0;
   };

   // Build a "block diagonal" select matrix to pick true entities
   const int num_entities = e_te_e->diag->num_cols;
   HYPRE_Int* select_i = HypreCTAlloc<HYPRE_Int>(num_entities+1); // Init with 0
   for (int e = 0; e < num_entities; e++)
   {
      int offset = e_te_e_offd_i[e];
      bool e_is_shared = e_te_e_offd_i[e+1] > offset;
      auto shared_e_min_global_id = SharedEntitiesMinGlobalID(e_is_shared, offset);
      bool global_id_is_smaller_shared = shared_e_min_global_id > owned_e_max_global_id;
      select_i[e+1] = select_i[e] + (!e_is_shared || global_id_is_smaller_shared);
   }

   const int num_trueEntities = select_i[num_entities];
   HYPRE_Int* select_j = HypreCTAlloc<HYPRE_Int>(num_trueEntities);
   std::iota(select_j, select_j + num_trueEntities, 0);
   double* select_data = HypreCTAlloc<double>(num_trueEntities);
   std::fill_n(select_data, num_trueEntities, 1.);

   vector<int> entity_starts(2);
   copy_n(e_te_e->row_starts, 2, entity_starts.begin());
   auto trueEntity_starts = GenerateOffsets(e_te_e->comm, num_trueEntities);
   HYPRE_Int global_num_trueEntities = trueEntity_starts.back();

   ParMatrix select(e_te_e->comm, e_te_e->global_num_rows,
                    global_num_trueEntities, move(entity_starts),
                    move(trueEntity_starts), select_i, select_j, select_data);

   return Mult(entity_trueEntity_entity, select);
}

ParMatrix EntityToProcessor(MPI_Comm comm, const vector<int>& ent_redist_procs)
{
   int num_procs, myid;
   MPI_Comm_size(comm, &num_procs);
   MPI_Comm_rank(comm, &myid);

   const unsigned int num_local_entities = ent_redist_procs.size();

   map<unsigned, unsigned> col_map_inv; // Keys of std::map are sorted
   int num_entities_remained = 0; // num_entities remained after redistribution
   bool* nonzeros_loc = new bool[num_procs]();
   bool* nonzeros = new bool[num_procs]();
   for (unsigned int i = 0; i < num_local_entities; i++)
   {
      const int destination = ent_redist_procs[i];
      nonzeros_loc[destination] = true;
      if (destination == myid) { num_entities_remained++; }
      else { col_map_inv[destination] = 1; }
   }
   int num_entities_moved = num_local_entities - num_entities_remained;

   // TODO: nonzeros is never used?
   MPI_Allreduce(nonzeros_loc, nonzeros, num_procs, MPI::BOOL, MPI_LOR, comm);
   delete[] nonzeros_loc;
   delete[] nonzeros;

   // Construct and sort the offd col map for hypre_ParCSRmatrix
   HYPRE_Int* col_map = HypreCTAlloc<HYPRE_Int>(col_map_inv.size());
   int num_procs_offd = 0;
   for (auto i = col_map_inv.begin(); i != col_map_inv.end(); i++)
   {
      col_map_inv[i->first] = num_procs_offd;
      col_map[num_procs_offd++] = i->first;
   }

   HYPRE_Int* diag_i = HypreCTAlloc<HYPRE_Int>(num_local_entities+1);
   HYPRE_Int* diag_j = HypreCTAlloc<HYPRE_Int>(num_entities_remained); // all diag_j = 0
   HYPRE_Int* offd_i = HypreCTAlloc<HYPRE_Int>(num_local_entities+1);
   HYPRE_Int* offd_j = HypreCTAlloc<HYPRE_Int>(num_entities_moved);

   num_entities_moved = 0;
   for (unsigned int i = 0; i < num_local_entities; i++)
   {
      const int destination = ent_redist_procs[i];
      if (destination == myid)
      {
         diag_i[i+1] = diag_i[i]+1;
         offd_i[i+1] = offd_i[i];
      }
      else
      {
         diag_i[i+1] = diag_i[i];
         offd_i[i+1] = offd_i[i]+1;
         offd_j[num_entities_moved++] = col_map_inv[destination];
      }
   }

   double* diag_data = HypreCTAlloc<double>(num_entities_remained);
   double* offd_data = HypreCTAlloc<double>(num_entities_moved);
   std::fill_n(diag_data, num_entities_remained, 1.0);
   std::fill_n(offd_data, num_entities_moved, 1.0);

   auto entity_starts = GenerateOffsets(comm, num_local_entities);
   auto proc_starts = GenerateOffsets(comm, 1);
   HYPRE_Int num_global_entities = entity_starts.back();
   HYPRE_Int num_global_procs = proc_starts.back();

   return ParMatrix(comm, num_global_entities, num_global_procs,
                    move(entity_starts), move(proc_starts), diag_i, diag_j, diag_data,
                    offd_i, offd_j, offd_data, num_procs_offd, col_map);
}

/// Construct "redistributed entities" to "true entities" relation table
/// based on "redistributed reference entities" to "true entities" table
/// This function expects reference entities to be of fundamental type
ParMatrix BuildRedistributedEntityToTrueEntity(const ParMatrix& redRef_trueEnt_)
{
   hypre_ParCSRMatrix* redRef_trueEnt = redRef_trueEnt_;
   HYPRE_Int* redRef_trueEnt_colmap = redRef_trueEnt->col_map_offd;

   // Each nonzero column of "rows of redRef_trueEnt owned by current processor"
   // is a redistributed entity to be owned by the current processor
   auto trueEnt_diag = FindNonZeroColumns(*(redRef_trueEnt->diag));
   auto trueEnt_offd = FindNonZeroColumns(*(redRef_trueEnt->offd));
   const int num_trueEnt_diag = trueEnt_diag.size();
   const int num_trueEnt_offd = trueEnt_offd.size();
   const int num_redEnt = num_trueEnt_diag + num_trueEnt_offd;

   vector<HYPRE_Int> trueEnt_starts(2);
   copy_n(redRef_trueEnt->col_starts, 2, trueEnt_starts.begin());

   // Find the first offd index that is greater than all diag indices
   // We will fill the permutation matrix in such a way that new local
   // distributed entities are ordered by the global indices of the
   // corresponding old distributed entities
   // (this is only needed for ParElag/smoothG)
   int order_preserve_offset = 0;
   for (; order_preserve_offset < num_trueEnt_offd; order_preserve_offset++)
   {
      if (redRef_trueEnt_colmap[order_preserve_offset] > trueEnt_starts[0])
      {
         break;
      }
   }

   // allocate memory for "redistributed entities" to "true entities" table
   HYPRE_Int* out_diag_i = HypreCTAlloc<HYPRE_Int>(num_redEnt + 1);
   HYPRE_Int* out_diag_j = HypreCTAlloc<HYPRE_Int>(num_trueEnt_diag);
   double* out_diag_data = HypreCTAlloc<double>(num_trueEnt_diag);
   HYPRE_Int* out_offd_i = HypreCTAlloc<HYPRE_Int>(num_redEnt + 1);
   HYPRE_Int* out_offd_j = HypreCTAlloc<HYPRE_Int>(num_trueEnt_offd);
   double* out_offd_data = HypreCTAlloc<double>(num_trueEnt_offd);

   // diag part of the redEnt_trueEnt table
   std::fill_n(out_diag_i, order_preserve_offset, 0);
   std::iota(out_diag_i+order_preserve_offset,
             out_diag_i+order_preserve_offset+num_trueEnt_diag+1, 0);
   std::fill_n(out_diag_i+order_preserve_offset+num_trueEnt_diag+1,
               num_trueEnt_offd-order_preserve_offset, num_trueEnt_diag);
   std::copy_n(trueEnt_diag.begin(), num_trueEnt_diag, out_diag_j);
   std::fill_n(out_diag_data, num_trueEnt_diag, 1.0);

   // offd part of the redEnt_trueEnt table
   std::iota(out_offd_i, out_offd_i+order_preserve_offset+1, 0);
   std::fill_n(out_offd_i+order_preserve_offset,
               num_trueEnt_diag, order_preserve_offset);
   std::iota(out_offd_i+num_trueEnt_diag+order_preserve_offset,
             out_offd_i+num_redEnt+1, order_preserve_offset);
   std::iota(out_offd_j, out_offd_j+num_trueEnt_offd, 0);
   std::fill_n(out_offd_data, num_trueEnt_offd, 1.0);

   // Copy the column map of redRef_trueEnt to redEnt_trueEnt
   HYPRE_Int* out_colmap = HypreCTAlloc<HYPRE_Int>(num_trueEnt_offd);
   int i = 0;
   for (auto it = trueEnt_offd.begin(); it != trueEnt_offd.end(); ++it, ++i)
   {
       out_colmap[i] = redRef_trueEnt_colmap[*it];
   }

   MPI_Comm comm = redRef_trueEnt->comm;
   auto redEnt_starts = GenerateOffsets(comm, num_redEnt);
   HYPRE_Int num_global_redEnt = redEnt_starts.back();

   // Construct redEnt_trueEnt
   return ParMatrix(comm, num_global_redEnt, redRef_trueEnt->global_num_cols,
                    move(redEnt_starts), move(trueEnt_starts), out_diag_i,
                    out_diag_j, out_diag_data, out_offd_i, out_offd_j,
                    out_offd_data, num_trueEnt_offd, out_colmap);
}

ParMatrix BuildTrueEntityRedistributionMatrix(const ParMatrix& redElem_trueEntity)
{
   auto redEntity_trueEntity = BuildRedistributedEntityToTrueEntity(redElem_trueEntity);

   auto tE_redE = Transpose(redEntity_trueEntity);
   auto redE_tE_redE = Mult(redEntity_trueEntity, tE_redE);

   auto redEntity_redTrueEntity = BuildEntityToTrueEntity(redE_tE_redE);
   auto redTE_redE = Transpose(redEntity_redTrueEntity);

   auto redTrueEntity_trueEntity = Mult(redTE_redE, redEntity_trueEntity);
   SetConstantValue(redTrueEntity_trueEntity, 1.);

   return redTrueEntity_trueEntity;
}

} // namespace matred
