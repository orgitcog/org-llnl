#include "../core/include/mpibind.h"

int main(){
    int ret;

    hwloc_topology_t topology;
//    char topo_path[128] = "/g/g90/loussert/Work/loussert-mpibind/topos/quartz_topo.xml";
    char topo_path[128] = "/g/g90/loussert/Work/loussert-mpibind/topos/lassen_login.xml";
//    char topo_path[128] = "/g/g90/loussert/Work/loussert-mpibind/topos/epyc-dual-sock.xml";

    mpibind_t *handle;
    handle = mpibind_create();


    /* Topology */
    if(access(topo_path, F_OK) == -1){     /* Check if topology file exists */
        fprintf(stderr, "##ERROR: Could not open topology file '%s'\nExit\n", topo_path);
        exit(1);
    }
    /* Allocate and initialize topology object */
    hwloc_topology_init(&topology);
    /* Retrieve topology from xml file */
    ret = hwloc_topology_set_xml(topology, topo_path);
    if(ret == -1){
        fprintf(stderr, "##ERROR: Hwloc failed to load topology file '%s'\nExit\n", topo_path);
        exit(1);
    }
    /* Set flags to retrieve PCI devices */
    hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_THISSYSTEM_ALLOWED_RESOURCES);
    hwloc_topology_set_all_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_ALL);
    /* Build Topology */
    hwloc_topology_load(topology);



    /* mpibind handle initialisation */
    mpibind_set_topology(handle, topology);
    mpibind_set_verbose(handle, 1); /* 1 for true */

    mpibind_set_local_nprocess(handle, 1);
    //mpibind_set_local_nprocess(handle, 2);
    //mpibind_set_local_nprocess(handle, 4);
    
    //mpibind_set_user_smt(handle, 1);
    mpibind_set_user_smt(handle, 4);

    //mpibind_set_user_num_thread(handle, -1);
    //mpibind_set_user_num_thread(handle, 0);
    mpibind_set_user_num_thread(handle, 4);



    /* launch mpibind */
    handle = mpibind(handle);

    /* print results */
    printf("\n\n#### Printing results from test launcher:\n");
    mpibind_print(handle);

    /* Free stuff */
    mpibind_destroy(handle);
    hwloc_topology_destroy(topology);

    return 0;
}
