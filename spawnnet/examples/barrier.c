#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <slurm/pmi2.h>

#include "spawn.h"
#include "comm.h"

int main(int argc, char **argv)
{
    /* initialize PMI, get our rank and process group size */
    int spawned, size, rank, appnum;
    PMI2_Init(&spawned, &size, &rank, &appnum);

    /* open and endpoint and get its name */
    //spawn_net_endpoint* ep = spawn_net_open(SPAWN_NET_TYPE_IBUD);
    spawn_net_endpoint* ep = spawn_net_open(SPAWN_NET_TYPE_TCP);

    /* allocate communicator */
    lwgrp_comm comm;
    comm_create(rank, size, ep, &comm);

    /**********************
     * barrier across all processes
     **********************/
    lwgrp_barrier(comm.world);

    /**********************
     * barrier between procs on the same node
     **********************/
    lwgrp_barrier(comm.node);

    /**********************
     * barrier across all processes (two-level version),
     * procs on node signal their leader, barrier across leaders, leader signal procs on its node
     **********************/
    lwgrp_barrier(comm.node);
    int64_t rank_node = lwgrp_rank(comm.node);
    if (rank_node == 0) {
        lwgrp_barrier(comm.leaders);
    }
    lwgrp_barrier(comm.node);

    /* free communicator */
    comm_free(&comm);

    /* close our endpoint and channel */
    spawn_net_close(&ep);

    /* shut down PMI */
    PMI2_Finalize();

    return 0;
}
