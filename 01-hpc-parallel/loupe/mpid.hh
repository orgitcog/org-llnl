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

#ifndef __MPID_HH
#define __MPID_HH

#include <map>
#include <string>
#include <vector>
#include <sys/time.h>

class mpid_data_stats {

  public:
    mpid_data_stats(std::string p_section_name, std::string* p_call_names);

    void mpid_init();
    void mpid_finalize();

    void mpid_call_start(int p_name);
    uint64_t mpid_call_end(int p_name);

    void mpid_call_stats(int p_count, int p_datatype, uint64_t p_time, int p_name);
    void mpid_traffic_pattern(int dest, int p_count, int p_datatype, int comm, int p_name);

    void mpid_add_communicator(MPI_Comm* newcomm);

    bool is_me(int rank, MPI_Comm newcomm);
    /**
     * Functions to get the profile data as matrices for hdf5 output
     */   
    uint64_t app_time(){return m_app_time;}
    uint64_t mpi_time(){return m_mpi_acc_time;}

    uint64_t n_mpi_calls(){ return m_global_call_data.size();}
    uint64_t n_mpi_callsites(){ return m_mpi_call_data.size();}
    uint64_t size_pattern(){ return m_pattern.size();}
    uint64_t size_call_pattern();

    void calls_data(uint64_t *p_data_out);
    void callsites_data(uint64_t *p_data_out);
    void pattern_data(uint64_t *p_data_out);
    void call_pattern_data(uint64_t *p_data_out);
  private:

    mpid_data_stats(const mpid_data_stats&);  
    mpid_data_stats& operator=(const mpid_data_stats&);

    // Rank for this process in the comm world
    int m_rank;
    int m_num_ranks;
    // Map that had the rank in every comunicator
    // its mostly a translation table
    std::map<uint64_t,std::vector<int>> m_comm_rank;
    // Stores my rank in every communicator I'm in
    std::map<uint64_t, int> m_my_comm_ranks;

    struct mpi_call_t{
        uint64_t m_kbytes_sent;
        uint64_t m_total_calls;
        uint64_t m_time_spent;
        int m_name;
    };

    std::map<uint64_t, mpi_call_t> m_mpi_call_data;
    //Int are the calls identifier
    std::map<int, mpi_call_t> m_global_call_data;

    struct mpi_pattern_t{
        uint64_t m_kbytes;
        //Get the name for the per call patterns
        uint64_t m_comm;
        uint64_t m_count;
        int m_name;
    };

    std::map<int,mpi_pattern_t> m_pattern;
    //PC, <DEST NODE, KBYTES SENT>
    std::map<uint64_t, std::map<int,mpi_pattern_t>> m_call_pattern;

    struct timeval m_timer_start, m_timer_end;
    //This is for MPI functions
    struct timeval m_func_timer_start, m_func_timer_end;
    uint64_t m_app_time;
    uint64_t m_mpi_acc_time;

    std::string m_section_name;
    std::string* m_call_names;

    bool m_enabled;
};

#endif
