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

#include <string>
#include <set>
#include <sstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <mpi.h>
#include <hdf5.h>

#include "mpid.hh"

// We need to create datasets with the max length of all ranks
//
int max_calls_size;
int max_callsites_size;
int max_pattern_size;

std::string get_dump_file(const std::string& name) {
  std::ostringstream filename;

  //const char *dump_dir = getenv(PDUMP_DUMP_DIR);
  //if (dump_dir) {
  //  std::string dump_dir_str(dump_dir);
  //  if (!dump_dir_str.empty()) {
  //    filename << dump_dir_str;
  //    size_t back = dump_dir_str.size() - 1;
  //    if (dump_dir_str[back] != '/') {
  //      filename << "/";
  //    }
  //  }
  //}

  filename << name << ".h5";
  return filename.str();
}

void create_space(hid_t h5file, std::string name, hsize_t ndim, hsize_t* cur_dims){

  hid_t space   = H5Screate_simple(ndim, cur_dims, NULL);
  hid_t dset = H5Dcreate1(h5file, name.c_str(), H5T_NATIVE_LONG, space, H5P_DEFAULT);

  H5Sclose(space);
  H5Dclose(dset);
}

hid_t create_dump_file(const std::string& filename, mpid_data_stats* mpi_stats) {
  int nprocs;
  PMPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  hid_t access_plist = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(access_plist, MPI_COMM_WORLD, MPI_INFO_NULL);
  hid_t h5file = H5Fcreate(
    filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, access_plist);

  
  /*NEED TO CREATE ONE SPACE PER EVERY STAT THAT WE WANT TO SAVE
  
    STAT AND DIMENSIONS
    APP TIME (ms) (nr_ranks) Total spent time
    MPI TIME (ms) (nr_ranks)
  
    CALLS PROFILE (nr_ranks, nr_calls, 4) 
       35 -> [ call #_calls acc_time(us) bytes]

    CALL SITE STATS (nr_ranks, nr_symbols, 4)
       35 -> [ symbol #_calls acc_time(us) bytes]

    GLOBAL PATTERN (nr_ranks, n_comms, 2)
       35 -> [dest count bytes]

    CALL PATTERN (undef)  
  */ 
  hsize_t dims[] = {(hsize_t)nprocs};

  int calls_size = mpi_stats->n_mpi_calls();
  int callsites_size = mpi_stats->n_mpi_callsites();
  int pattern_size = mpi_stats->size_pattern();

  PMPI_Allreduce(&calls_size,     &max_calls_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  PMPI_Allreduce(&callsites_size, &max_callsites_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  PMPI_Allreduce(&pattern_size,   &max_pattern_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  hsize_t dim_calls[] =     {(hsize_t)nprocs, (hsize_t)max_calls_size,     4};
  hsize_t dim_callsites[] = {(hsize_t)nprocs, (hsize_t)max_callsites_size, 5};
  hsize_t dim_pattern[] =   {(hsize_t)nprocs, (hsize_t)max_pattern_size,   3};
  //hsize_t dim_call_pattern[] = {(hsize_t)nprocs,(hsize_t)mpi_stats->size_call_pattern(), 4};

  create_space(h5file,"app_time", 1, dims );
  create_space(h5file,"mpi_time", 1, dims );
  create_space(h5file,"calls", 3, dim_calls );
  create_space(h5file,"callsites", 3, dim_callsites );
  create_space(h5file,"pattern", 3, dim_pattern );
  //create_space(h5file,"callpattern", 3, dim_call_pattern );


  H5Pclose(access_plist);
  return h5file;
}




hid_t open_dump_file(const std::string& filename, MPI_Comm comm) {
  hid_t access_plist = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_mpio(access_plist, comm, MPI_INFO_NULL);
  hid_t file = H5Fopen(filename.c_str(), H5F_ACC_RDWR, access_plist);
  H5Pclose(access_plist);

  return file;
}


void append_row(hid_t dump_file_id, const std::string& event_name,
                const uint64_t *data, MPI_Comm comm) {
  int rank, nprocs;
  PMPI_Comm_rank(comm, &rank);
  PMPI_Comm_size(comm, &nprocs);
   
  //This is the SECTION NAME
  hid_t dataset_id = H5Dopen1(dump_file_id, event_name.c_str());
  hid_t space_id   = H5Dget_space(dataset_id);


  hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
 
  //Needs to create a mem space
  hsize_t marray[] = {1};
  hid_t mid = H5Screate_simple(1, marray, NULL);

  //Write to the space
  //Needs to select the process slab
  hsize_t start[]  = {(hsize_t)rank};
  hsize_t count[]  = {1};
  hsize_t stride[] = {1};
  //herr_t ret = H5Sselect_elements (space_id, H5S_SELECT_SET, 1, start);
  H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start, stride,
   	    count, NULL);
  H5Dwrite(dataset_id, H5T_NATIVE_LONG, mid, space_id, xfer_plist, data);

  H5Dclose(dataset_id);
  H5Sclose(space_id);
  H5Pclose(xfer_plist);
}

void append_matrix(hid_t dump_file_id, const std::string& event_name,
                uint64_t *data, MPI_Comm comm, int sx, int sy) {
  int rank, nprocs;
  PMPI_Comm_rank(comm, &rank);
  PMPI_Comm_size(comm, &nprocs);

  //This is the SECTION NAME
  hid_t dataset_id = H5Dopen1(dump_file_id, event_name.c_str());
  hid_t space_id   = H5Dget_space(dataset_id);


  hid_t xfer_plist = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
 
  //Write to the space
  //Needs to select the process slab
  hsize_t start[]  = {(hsize_t)rank,0,0};
  hsize_t count[]  = {1,(hsize_t)sx,(hsize_t)sy};
  hsize_t stride[] = {1,1,1};

  H5Sselect_hyperslab(space_id, H5S_SELECT_SET, start, stride,
   	    count, NULL);

  hid_t mid = H5Screate_simple(3, count, NULL);
  uint64_t (*datam)[sy] = (uint64_t (*)[sy])data;
  H5Dwrite(dataset_id, H5T_NATIVE_LONG, mid, space_id, xfer_plist, datam);

  H5Dclose(dataset_id);
  H5Sclose(space_id);
  H5Pclose(xfer_plist);
}


// Dump out dataviz information to potentially multipe partial
void hdf5_dump(mpid_data_stats* mpi_stats,const std::string& name) {
  
  MPI_Comm comm = MPI_COMM_WORLD;
  hid_t dump_file_id;
  std::string dump_file_name = get_dump_file(name);

  dump_file_id = create_dump_file(dump_file_name, mpi_stats);

  uint64_t app_time = mpi_stats->app_time();
  uint64_t mpi_time = mpi_stats->mpi_time();


  append_row(dump_file_id, "app_time", &app_time, comm);
  append_row(dump_file_id, "mpi_time", &mpi_time, comm);

  uint64_t* calls =     new uint64_t[max_calls_size*4];
  mpi_stats->calls_data(calls);
  append_matrix(dump_file_id, "calls", calls, comm, max_calls_size, 4);
  delete[] calls;

  uint64_t* callsites = new uint64_t[max_callsites_size *5];
  mpi_stats->callsites_data(callsites);
  append_matrix(dump_file_id, "callsites", callsites, comm, max_callsites_size, 5);
  delete[] callsites;
  uint64_t* pattern = new uint64_t[max_pattern_size*3];
  mpi_stats->pattern_data(pattern);
  append_matrix(dump_file_id, "pattern", pattern, comm,mpi_stats->size_pattern(),3);
  delete[] pattern;
  //uint64_t* callpattern = new uint64_t[max_pattern_size*4];
  //mpi_stats->call_pattern_data(callpattern);
  //append_matrix(dump_file_id, "callpattern", callpattern, comm,mpi_stats->size_call_pattern(),4);
  //delete[] callpattern;

  // Get the calls data
  H5Fclose(dump_file_id);

}
