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

#include <mpi.h>

#include "hdf5_dump.hh"
#include "mpid.hh"
#include "util.hh"


mpid_data_stats::mpid_data_stats(std::string p_section_name, std::string* p_call_names) : m_section_name(p_section_name), m_call_names(p_call_names){
    PMPI_Comm_size(MPI_COMM_WORLD, &m_num_ranks); 
    PMPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    m_enabled = true;
}

void 
mpid_data_stats::mpid_init(){
    gettimeofday(&m_timer_start, NULL);
}

void 
mpid_data_stats::mpid_finalize(){
   
    gettimeofday(&m_timer_end, NULL);
    m_enabled = false;
    m_app_time = (m_timer_end.tv_sec - m_timer_start.tv_sec)*1e6 + (m_timer_end.tv_usec - m_timer_start.tv_usec);
    m_app_time /= 1000;
    //Create a dict file with the callsites symbols and fn names ids
    dump_symbols(m_call_names);
    //binary dump to hdf5
    hdf5_dump(this);
}

void 
mpid_data_stats::mpid_call_start(int p_name){

    gettimeofday(&m_func_timer_start, NULL);

}

uint64_t 
mpid_data_stats::mpid_call_end(int p_name){
    uint64_t elapsed;
    gettimeofday(&m_func_timer_end, NULL);
    elapsed = (m_func_timer_end.tv_sec - m_func_timer_start.tv_sec)*1e6 + (m_func_timer_end.tv_usec - m_func_timer_start.tv_usec);
    return elapsed;
}


void 
mpid_data_stats::mpid_call_stats(int p_count, int p_datatype, uint64_t p_time, int p_name){
    uint64_t pc;
    int type_size;
    if(!m_enabled) return;
    backtrace(&pc);
    PMPI_Type_size(p_datatype,&type_size);
    type_size *= p_count;
    if(m_global_call_data.find(p_name)==m_global_call_data.end()){
        m_global_call_data[p_name].m_kbytes_sent = 0;
        m_global_call_data[p_name].m_total_calls = 0;
        m_global_call_data[p_name].m_time_spent = 0;
    }
    if(m_mpi_call_data.find(pc)==m_mpi_call_data.end()){
        m_mpi_call_data[pc].m_kbytes_sent  = 0;
        m_mpi_call_data[pc].m_total_calls = 0;
        m_mpi_call_data[pc].m_time_spent  = 0;
        m_mpi_call_data[pc].m_name = p_name;
    }
    m_mpi_call_data[pc].m_kbytes_sent += type_size;
    m_mpi_call_data[pc].m_total_calls++;
    m_mpi_call_data[pc].m_time_spent +=p_time;
    m_global_call_data[p_name].m_kbytes_sent += type_size;
    m_global_call_data[p_name].m_total_calls++;
    m_global_call_data[p_name].m_time_spent +=p_time;
    m_mpi_acc_time +=p_time; 
}

void 
mpid_data_stats::mpid_traffic_pattern(int dest, int p_count, int p_datatype, int comm, int p_name){
    uint64_t pc;
    int type_size;
    if(!m_enabled) return;
    PMPI_Type_size(p_datatype, &type_size);
    uint64_t kbytes =  type_size * p_count;
    backtrace(&pc);
    //PTP COMM
    if(dest !=-1){
        if(m_pattern.find(dest) == m_pattern.end()){
            m_pattern[dest].m_kbytes = 0;
            m_pattern[dest].m_count = 0;
        }
        if(m_call_pattern[pc].find(dest) == m_call_pattern[pc].end()){
            m_call_pattern[pc][dest].m_kbytes = 0;
            m_call_pattern[pc][dest].m_count = 0;
            m_call_pattern[pc][dest].m_comm = comm;
        }
          
        m_pattern[dest].m_kbytes    += kbytes;
        m_pattern[dest].m_count++;
        m_call_pattern[pc][dest].m_kbytes += kbytes;
        m_call_pattern[pc][dest].m_count ++;
        m_call_pattern[pc][dest].m_name = p_name;

    } else if(comm!= MPI_COMM_WORLD){
        for(uint32_t i=0;i<m_comm_rank[comm].size();i++){
            int dest=m_comm_rank[comm][i];
            if(dest == m_rank) continue;
            if(m_pattern.find(dest) == m_pattern.end()){
                m_pattern[dest].m_kbytes = 0;
                m_pattern[dest].m_count = 0;
            }
            if(m_call_pattern[pc].find(dest) == m_call_pattern[pc].end()){
                m_call_pattern[pc][dest].m_kbytes = 0;
                m_call_pattern[pc][dest].m_count = 0;
                m_call_pattern[pc][dest].m_comm = comm;
            }
              
            m_pattern[dest].m_kbytes    += kbytes;
            m_pattern[dest].m_count++;
            m_call_pattern[pc][dest].m_kbytes += kbytes;
            m_call_pattern[pc][dest].m_count ++;
            m_call_pattern[pc][dest].m_name = p_name;
    
        }
    } else {
        for(int i=0;i<m_num_ranks;i++){
            int dest=i;
            if(dest == m_rank) continue;
            if(m_pattern.find(dest) == m_pattern.end()){
                m_pattern[dest].m_kbytes = 0;
                m_pattern[dest].m_count = 0;
            }
            if(m_call_pattern[pc].find(dest) == m_call_pattern[pc].end()){
                m_call_pattern[pc][dest].m_kbytes = 0;
                m_call_pattern[pc][dest].m_count = 0;
                m_call_pattern[pc][dest].m_comm = comm;
            }
              
            m_pattern[dest].m_kbytes    += kbytes;
            m_pattern[dest].m_count++;
            m_call_pattern[pc][dest].m_kbytes += kbytes;
            m_call_pattern[pc][dest].m_count ++;
            m_call_pattern[pc][dest].m_name = p_name;
    
        }

    }
}

void 
mpid_data_stats::mpid_add_communicator(MPI_Comm* newcomm){

    MPI_Group grp, world_grp;
    int grp_size;
    int *ranks;
    int *world_ranks;

    //Lets get a comm->world translation
    PMPI_Comm_group(MPI_COMM_WORLD, &world_grp);
    PMPI_Comm_group(*newcomm, &grp); 
    PMPI_Group_size(grp, &grp_size);
    ranks = new int[grp_size];
    world_ranks = new int[grp_size];
    for (int i = 0; i < grp_size; i++)
        ranks[i] = i;

    PMPI_Group_translate_ranks(grp, grp_size, ranks, world_grp, world_ranks);
    m_comm_rank[*newcomm].resize(grp_size);
    for (int i = 0; i < grp_size; i++){
        m_comm_rank[*newcomm][i] = world_ranks[i];
        if(world_ranks[i] == m_rank)
            m_my_comm_ranks[*newcomm] = i;
    }

    delete[] ranks;
    delete[] world_ranks;

}


bool 
mpid_data_stats::is_me(int rank, MPI_Comm newcomm){
   bool is_me = false;

   if(rank == MPI_COMM_WORLD){
       is_me = rank == m_rank;

   }else if(m_my_comm_ranks.find(newcomm)!=m_my_comm_ranks.end()){
       is_me = rank == m_my_comm_ranks[newcomm];
   }

   return is_me;
}

uint64_t 
mpid_data_stats::size_call_pattern(){
   uint64_t size = 0;
   for(auto it=m_call_pattern.begin();it!=m_call_pattern.end();it++)
       size += it->second.size(); 
   return size;
}

void 
mpid_data_stats::calls_data(uint64_t *p_data_out){

    int i=0;
    for(auto it=m_global_call_data.begin();it!=m_global_call_data.end();it++){ 
        //printf("%ld\n",it->first);
        p_data_out[i*4+0] = it->first;
        p_data_out[i*4+1] = it->second.m_total_calls;
        p_data_out[i*4+2] = it->second.m_time_spent;
        p_data_out[i*4+3] = it->second.m_kbytes_sent;
        //if(m_rank==3){
        //    printf("%d %ld %ld %ld %ld\n",it->first,p_data_out[i*4+0],p_data_out[i*4+1],p_data_out[i*4+2], p_data_out[i*4+3]);
        //}
        i++;
    }
}

void 
mpid_data_stats::callsites_data(uint64_t *p_data_out){

    int i=0;
    for(auto it=m_mpi_call_data.begin();it!=m_mpi_call_data.end();it++){ 
        //printf("%ld\n",it->first);
        p_data_out[i*5+0] = it->first;
        p_data_out[i*5+1] = it->second.m_name;
        p_data_out[i*5+2] = it->second.m_total_calls;
        p_data_out[i*5+3] = it->second.m_time_spent;
        p_data_out[i*5+4] = it->second.m_kbytes_sent;
        //if(m_rank==3){
        //    printf("%ld %ld %ld %ld %ld\n",p_data_out[i*4+0],p_data_out[i*4+1],p_data_out[i*4+2], p_data_out[i*4+3],p_data_out[i*4+4]);
        //}
        i++;
    }
    //printf("size is %d , *4 is %d i is %d\n",n_mpi_callsites(),n_mpi_callsites()*4,i);
    //printf("rank %d size %d\n",m_rank,m_mpi_call_data.size());
}

void 
mpid_data_stats::pattern_data(uint64_t *p_data_out){

    int i=0;
    for(auto it=m_pattern.begin();it!=m_pattern.end();it++){ 
        p_data_out[i*3+0] = it->first;
        p_data_out[i*3+1] = it->second.m_kbytes;
        p_data_out[i*3+2] = it->second.m_count;
        //if(m_rank==3){
        //    printf("%d %ld %ld\n %ld",it->first,p_data_out[i*4+0],p_data_out[i*4+1],p_data_out[i*4+2]);
        //}
        i++;
    }
}

void 
mpid_data_stats::call_pattern_data(uint64_t *p_data_out){

    int i=0;
    for(auto it=m_call_pattern.begin();it!=m_call_pattern.end();it++){ 
        for(auto it2=it->second.begin();it2!=it->second.end();it2++){ 
            p_data_out[i*4+0] = it->first; //PC
            p_data_out[i*4+1] = it2->first; //DEST
            p_data_out[i*4+2] = it2->second.m_kbytes;
            p_data_out[i*4+3] = it2->second.m_count;
            i++;
        }
    }
}
