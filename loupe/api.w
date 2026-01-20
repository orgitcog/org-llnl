#include <sys/types.h>
#include <dlfcn.h>
#include <mpi.h>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <stddef.h>

#include <sys/time.h>
#include "mpid.hh"

#include <map>
#include <vector>
#include <string>


mpid_data_stats* g_mpi_stats;

//Internal identifiers used by our profiler
{{forallfn foo}}
#define f_{{foo}} {{fn_num}}
{{endforallfn}}

std::string g_f_names[] = { {{forallfn foo}} 
                            "{{foo}}", 
                            {{endforallfn}}};

{{fn foo MPI_Init }}{
    {{callfn}}
    // Create the GLOBAL profiling data
    g_mpi_stats = new mpid_data_stats("global", g_f_names);
    g_mpi_stats->mpid_init();
}
{{endfn}}

// TODO: Eventually we need support for MPI_Thread_Multiple
{{fn foo MPI_Init_thread }}{
    {{callfn}}
    // Create the GLOBAL profiling data
    g_mpi_stats = new mpid_data_stats("global", g_f_names);
    g_mpi_stats->mpid_init();
}
{{endfn}}

{{fn foo MPI_Finalize }}{
    //We might need to do some MPI calls just before closing MPI
    g_mpi_stats->mpid_finalize();
    delete g_mpi_stats;
    {{callfn}}
}
{{endfn}}

{{fn foo MPI_Comm_split }}{
    {{callfn}}
    g_mpi_stats->mpid_add_communicator(newcomm); 
}
{{endfn}}

//TODO: hanndles for other sends
{{fn foo MPI_Send MPI_Isend MPI_Send_init }}{
    uint64_t long elapsed;
    g_mpi_stats->mpid_call_start(f_{{foo}});
    {{callfn}}
    elapsed = g_mpi_stats->mpid_call_end(f_{{foo}});

    g_mpi_stats->mpid_call_stats(count, datatype, elapsed, f_{{foo}});
    g_mpi_stats->mpid_traffic_pattern(dest, count, datatype, comm, f_{{foo}});
}
{{endfn}}

{{fn foo MPI_Recv MPI_Irecv MPI_Recv_init }}{
    uint64_t long elapsed;
    g_mpi_stats->mpid_call_start(f_{{foo}});
    {{callfn}}
    elapsed = g_mpi_stats->mpid_call_end(f_{{foo}});

    g_mpi_stats->mpid_call_stats(count, datatype, elapsed, f_{{foo}});
}
{{endfn}}

{{fn foo MPI_Waitany MPI_Wait MPI_Waitall }}{
    uint64_t long elapsed;
    g_mpi_stats->mpid_call_start(f_{{foo}});
    {{callfn}}
    elapsed = g_mpi_stats->mpid_call_end(f_{{foo}});
    g_mpi_stats->mpid_call_stats(0, MPI_UNSIGNED, elapsed, f_{{foo}});
}
{{endfn}}

{{fn foo MPI_Bcast }}{

    uint64_t long elapsed;
    g_mpi_stats->mpid_call_start(f_{{foo}});
    {{callfn}}
    elapsed = g_mpi_stats->mpid_call_end(f_{{foo}});

    g_mpi_stats->mpid_call_stats(count, datatype, elapsed, f_{{foo}});
    //Only measure traffic for the root
    //if (g_mpi_stats->is_me(root, comm)){
    //    g_mpi_stats->mpid_traffic_pattern(-1, count, datatype, comm, f_{{foo}});
    //}

}
{{endfn}}


{{fn MPI_Alltoall}}{

    uint64_t long elapsed;
    g_mpi_stats->mpid_call_start(f_{{foo}});
    {{callfn}}
    elapsed = g_mpi_stats->mpid_call_end(f_{{foo}});

    g_mpi_stats->mpid_call_stats(sendcount, sendtype, elapsed, f_{{foo}});
    //g_mpi_stats->mpid_traffic_pattern(-1, sendcount, sendtype, comm, f_{{foo}});

}
{{endfn}}


{{fn MPI_Allreduce}}{

    uint64_t long elapsed;
    g_mpi_stats->mpid_call_start(f_{{foo}});
    {{callfn}}
    elapsed = g_mpi_stats->mpid_call_end(f_{{foo}});

    g_mpi_stats->mpid_call_stats(count, datatype, elapsed, f_{{foo}});
    //g_mpi_stats->mpid_traffic_pattern(-1, count, datatype, comm, f_{{foo}});
}
{{endfn}}


