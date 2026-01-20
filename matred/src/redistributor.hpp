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
   @file redistributor.hpp
   @brief Core functions that implement the redistribution
*/

#ifndef __MATRED_REDISTRIBUTOR_HPP
#define __MATRED_REDISTRIBUTOR_HPP

#include "linalg.hpp"

using namespace std;

namespace matred
{

/// Construct entities to "true entities" relation from the relation graph
/// of entities that share the same "true entities"
ParMatrix BuildEntityToTrueEntity(const ParMatrix& entity_trueEntity_entity);

/// Construct entity-to-processor relation table from a distributed vector
/// @param ent_redist_procs an array of size number of entities, procs[i]
///        indicates the processor ID where entity i will be redistributed to.
ParMatrix EntityToProcessor(MPI_Comm comm, const vector<int>& ent_redist_procs);

/// Construct "redistributed entities" to "true entities" relation table
/// based on "redistributed reference entities" to "true entities" table
/// This function expects reference entities to be of fundamental type
ParMatrix BuildRedistributedEntityToTrueEntity(const ParMatrix& redRef_trueEnt_);

/// Construct "redistributed true entities" to "true entities" relation
ParMatrix BuildTrueEntityRedistributionMatrix(const ParMatrix& redElem_trueEntity);

} // namespace matred

#endif /* __MATRED_REDISTRIBUTOR_HPP */
