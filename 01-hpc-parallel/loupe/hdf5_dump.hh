//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
//
// Written by Emilio Castillo <ecastill@bsc.es>.
// LLNL-CODE-745958. All rights reserved.
//
// This file is part of Loupe. For details, see:
// https://github.com/LLNL/loupe
// Please also read the LICENSE file for the MIT License notice.
//////////////////////////////////////////////////////////////////////////////

#ifndef __HDF5_DUMP_HH
#define __HDF5_DUMP_HH
#include <string>

class mpid_data_stats;
void hdf5_dump(mpid_data_stats* mpi_stats, const std::string& name);

#endif
