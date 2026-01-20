#include "core/include/mpibind_functions.h"

#include <math.h>       /* For ceil(...) and floor(...) */

/****                         MPIBIND FUNCTIONS                          ****/
void not_implemented(){}

/* Deprecated */
//void mpibind_gather_options(){
//    char *mpibind_env;
//    char *ptr, *ptr2;
//
//    mpibind_env = getenv("MPIBIND");
//    if(mpibind_env == NULL)       /* Check if environment varibale exists */
//        return;
//
//    /* Separate the string inti individual option strings */
//    ptr2 = strtok (mpibind_env, ".");
//    while (ptr2 != NULL){
//        if(!strcmp(ptr2, "v")){
//            verbose = 1;
//        }
//        else if(!strcmp(ptr2, "dry")){
//            dryrun = 1;
//        }
//        else{
//          ptr = strstr(ptr2, "smt=");
//          if(ptr != NULL){
//              while(*ptr != '='){
//                  ptr++;
//              }
//              ptr++;
//              smt = atoi(ptr);
//          }
//        }
//        ptr2 = strtok (NULL, ".");
//    }
//    #ifdef __DEBUG
//    fprintf(stderr, "*** Getting options from the MPIBIND environment variable:\n");
//    fprintf(stderr, "\tverbose: %d\n\tdryrun: %d\n\tsmt: %d\n\n", verbose, dryrun, smt);
//    #endif /* __DEBUG */
//}
//
/* Deprecated */
//hwloc_topology_t mpibind_get_topology(){
//    int ret;
//    hwloc_topology_t topology;
//    const char *topo_path;
//
//    /* Get topology path from environment variable*/
//    topo_path = getenv("MPIBIND_TOPOLOGY_FILE");
//    if(topo_path == NULL){      /* Check if environment varibale exists */
//        fprintf(stderr, "##ERROR: Could not get topology file from environment variable 'MPIBIND_TOPOLOGY_FILE'\nExit\n");
//        exit(1);
//    }
//    if(access(topo_path, F_OK) == -1){     /* Check if topology file exists */
//        fprintf(stderr, "##ERROR: Could not open topology file '%s'\nExit\n", topo_path);
//        exit(1);
//    }
//
//    /* Allocate and initialize topology object */
//    hwloc_topology_init(&topology);
//    /* Retrieve topology from xml file */
//    ret = hwloc_topology_set_xml(topology, topo_path);
//    if(ret == -1){
//        fprintf(stderr, "##ERROR: Hwloc failed to load topology file '%s'\nExit\n", topo_path);
//        exit(1);
//    }
//    /* Set flags to retrieve PCI devices */
//    hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_THISSYSTEM_ALLOWED_RESOURCES);
//    hwloc_topology_set_all_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_ALL);
//    /* Build Topology */
//    hwloc_topology_load(topology);
//    
//    return topology;
//}

void mpibind_get_package_number(hwloc_topology_t topology){
    #ifdef __DEBUG
    fprintf(stderr, "*** Getting the number of packages\n\tPackage number: ");
    if (hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE) == HWLOC_TYPE_DEPTH_UNKNOWN) printf("unknown\n");
    else printf("%u\n", hwloc_get_nbobjs_by_depth(topology, hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE)));
    #endif /* __DEBUG */
}

void mpibind_package_list_init(hwloc_topology_t topology, hwloc_pkg_l **pkg_l, int local_nprocess){
    int i, ii, tmp_int;
    hwloc_pkg_l *tmp_pkg_l;
    static int process_loc_id_keeper = 0;
    static int pkg_nb_generator = 0;
    int numa_nb;
    hwloc_obj_t tmp_obj, tmp_obj2;


    #ifdef __DEBUG
    fprintf(stderr, "\n*** Getting the number of cores/pus/process per package\n");
    #endif /* __DEBUG */

    /* First we need to determine the number of numa node on the topology
     *     We could ask hwloc how many numa are present but some of them don't contain compute resources
     *     We need to go through each package and count the number of numa nodes there.
     */
    numa_nb = 0;
    for (i=0; i<hwloc_get_nbobjs_by_depth(topology, hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE)); i++){
        /*Get the package */
        tmp_obj = hwloc_get_obj_by_depth(topology, hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE), i);
        /* Look if package contains multiple NUMA objects */
        tmp_obj2 = hwloc_get_next_obj_inside_cpuset_by_type(topology, tmp_obj->cpuset, HWLOC_OBJ_NUMANODE, NULL);
        while(tmp_obj2){
            numa_nb ++;
            tmp_obj2 = hwloc_get_next_obj_inside_cpuset_by_type(topology, tmp_obj->cpuset, HWLOC_OBJ_NUMANODE, tmp_obj2);
        }
    }
    /* If the number of numa objects is greater than the number of processes, we will only use the first ones */
    if(numa_nb > local_nprocess) numa_nb = local_nprocess;

    tmp_int = local_nprocess; /* Used to count remaining processes to distribute */
    tmp_pkg_l = *pkg_l;       /* Used to easily insert package in the list */
    for (i=0; i<hwloc_get_nbobjs_by_depth(topology, hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE)); i++){
        /*Get the package */
        tmp_obj = hwloc_get_obj_by_depth(topology, hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE), i);
        /* Look if package contains multiple NUMA objects */
        /* The package's cpuset will alway contain one numa object 
         * If there are multiple numa object in the package, the will be put into groups 
         * Numa object don't have children objets and are not connected to the cores, caches, etc
         * Thus we will use "hwloc group objects" to put into package list if multiple numa are present in a package */
        tmp_obj2 = hwloc_get_next_obj_inside_cpuset_by_type(topology, tmp_obj->cpuset, HWLOC_OBJ_GROUP, NULL);
        if(tmp_obj2){
            while(tmp_obj2){
                /* Crete new list element */
                hwloc_pkg_l *pkg_obj; pkg_obj = NULL;
                pkg_obj = malloc(sizeof(hwloc_pkg_l));
                /* Fill object */
                pkg_obj->next       = NULL;
                pkg_obj->pkg        = tmp_obj2->parent;
                pkg_obj->index      = pkg_nb_generator;
                pkg_nb_generator++;
                pkg_obj->nb_core    = hwloc_get_nbobjs_inside_cpuset_by_type(topology, pkg_obj->pkg->cpuset, HWLOC_OBJ_CORE);
                pkg_obj->nb_pu      = hwloc_get_nbobjs_inside_cpuset_by_type(topology, pkg_obj->pkg->cpuset, HWLOC_OBJ_PU);
                pkg_obj->nb_process = (int)ceil(tmp_int / numa_nb); /* Remaining processes / remaining packages */
                tmp_int -= pkg_obj->nb_process;
                numa_nb--;
                pkg_obj->nb_worker  = 0;
                pkg_obj->gpu_l      = NULL;
                pkg_obj->cpuset_l   = NULL;
                pkg_obj->process    = NULL;
                pkg_obj->process = malloc(pkg_obj->nb_process*sizeof(mpibind_process));
                for(ii=0; ii<pkg_obj->nb_process; ii++){
                    pkg_obj->process[ii].local_id = process_loc_id_keeper;
                    process_loc_id_keeper++;
                    pkg_obj->process[ii].global_id = -1;
                    pkg_obj->process[ii].nb_thread = -1;
                }
                /* Debug prints */
                #ifdef __DEBUG
                fprintf(stderr, "\tPackage%d (os-index %d): cores:%d, pus:%d, processes:%d\n", 
                        pkg_obj->index, pkg_obj->pkg->os_index, pkg_obj->nb_core, pkg_obj->nb_pu, pkg_obj->nb_process);
                #endif /* __DEBUG */
                /* Add to the list */
                if(*pkg_l == NULL) *pkg_l = pkg_obj;
                else tmp_pkg_l->next = pkg_obj;
                tmp_pkg_l = pkg_obj;

                /* If we don't have remaining processes to distribute we stop here */
                if(tmp_int == 0){ 
                    return;
                }
                /* Get the next numa node */
                tmp_obj2 = hwloc_get_next_obj_inside_cpuset_by_type(topology, tmp_obj->cpuset, HWLOC_OBJ_GROUP, tmp_obj2);
            }
        }else{
            /* Crete new list element */
            hwloc_pkg_l *pkg_obj; pkg_obj = NULL;
            pkg_obj = malloc(sizeof(hwloc_pkg_l));
            /* Fill object */
            pkg_obj->next       = NULL;
            pkg_obj->pkg        = hwloc_get_obj_by_depth(topology, hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE), i);
            pkg_obj->index      = pkg_nb_generator;
            pkg_nb_generator++;
            pkg_obj->nb_core    = hwloc_get_nbobjs_inside_cpuset_by_type(topology, pkg_obj->pkg->cpuset, HWLOC_OBJ_CORE);
            pkg_obj->nb_pu      = hwloc_get_nbobjs_inside_cpuset_by_type(topology, pkg_obj->pkg->cpuset, HWLOC_OBJ_PU);
            pkg_obj->nb_process = (int)ceil(tmp_int / numa_nb); /* Remaining processes / remaining packages */
            tmp_int -= pkg_obj->nb_process;
            numa_nb--;
            pkg_obj->nb_worker  = 0;
            pkg_obj->gpu_l      = NULL;
            pkg_obj->cpuset_l   = NULL;
            pkg_obj->process    = NULL;
            pkg_obj->process = malloc(pkg_obj->nb_process*sizeof(mpibind_process));
            for(ii=0; ii<pkg_obj->nb_process; ii++){
                pkg_obj->process[ii].local_id = process_loc_id_keeper;
                process_loc_id_keeper++;
                pkg_obj->process[ii].global_id = -1;
                pkg_obj->process[ii].nb_thread = -1;
            }
            /* Debug prints */
            #ifdef __DEBUG
            fprintf(stderr, "\tPackage%d (os-index %d): cores:%d, pus:%d, processes:%d\n", 
                    pkg_obj->index, pkg_obj->pkg->os_index, pkg_obj->nb_core, pkg_obj->nb_pu, pkg_obj->nb_process);
            #endif /* __DEBUG */
            /* Add to the list */
            if(*pkg_l == NULL) *pkg_l = pkg_obj;
            else tmp_pkg_l->next = pkg_obj;
            tmp_pkg_l = pkg_obj;
            if(tmp_int == 0) return;
        }
    }
}

void mpibind_package_list_destroy(hwloc_pkg_l **pkg_l){
    hwloc_pkg_l *tmp_pkg_l;
    //hwloc_gpu_l *tmp_gpu_l, *tmp_gpu_l_2;
    hwloc_cpuset_l *tmp_cpuset_l;

    /* Free pkg_list (with their cpuset and gpu list) */
    tmp_pkg_l = *pkg_l;              /* Used to traverse the package list */
    while(tmp_pkg_l != NULL){
        hwloc_pkg_l *tmp_pkg_l_2;
        /* Free cpuset list */
        hwloc_cpuset_l *tmp_cpuset_l_2;
        tmp_cpuset_l = tmp_pkg_l->cpuset_l;
        while(tmp_cpuset_l != NULL){
            tmp_cpuset_l_2 = tmp_cpuset_l;
            tmp_cpuset_l   = tmp_cpuset_l->next;
            hwloc_bitmap_free(tmp_cpuset_l_2->cpuset);
            free(tmp_cpuset_l_2);
        }
        free(tmp_pkg_l->process);
        /* Free gpu list */
        //tmp_gpu_l = tmp_pkg_l->gpu_l;
        //while(tmp_gpu_l != NULL){
        //    tmp_gpu_l_2 = tmp_gpu_l;
        //    tmp_gpu_l   = tmp_gpu_l->next;
        //    free(tmp_gpu_l_2);
        //}
        /* Get next element and Free current package list element */
        tmp_pkg_l_2 = tmp_pkg_l;
        tmp_pkg_l = tmp_pkg_l->next;
        free(tmp_pkg_l_2);
    }
}

void mpibind_compute_thread_number_per_package(hwloc_topology_t topology, hwloc_pkg_l **pkg_l, int nthread){
    int i, tmp_int;
    hwloc_pkg_l *tmp_pkg_l;

    #ifdef __DEBUG
    fprintf(stderr, "\n*** Getting the number of workers\n");
    #endif /* __DEBUG */
    tmp_pkg_l = *pkg_l;          /* Used to traverse the package list */
    while(tmp_pkg_l != NULL){
        if (nthread == -1){      /* If application is not threaded */
            tmp_pkg_l->nb_worker = tmp_pkg_l->nb_process;
            for(i=0; i<tmp_pkg_l->nb_process; i++){
                tmp_pkg_l->process[i].nb_thread = 1;
            }
        }
        else if(nthread > 0){    /* Else if nb thread id defined */
            tmp_pkg_l->nb_worker = nthread * tmp_pkg_l->nb_process;
            for(i=0; i<tmp_pkg_l->nb_process; i++){
                tmp_pkg_l->process[i].nb_thread = nthread;
            }
        }
        else if(nthread == 0){
            char *tmp_char;
            /* If nthread not defined (=0) look at OMP_NUM_THREADS */
            tmp_char = getenv("OMP_NUM_THREADS");
            if(tmp_char){
                nthread = atoi(tmp_char);
                /* Check if env variable is actually an integer */
                /* if(isdigit(nthread) != 0) */ /* TODO: Not working ?! */
                tmp_pkg_l->nb_worker = nthread * tmp_pkg_l->nb_process;
                for(i=0; i<tmp_pkg_l->nb_process; i++){
                    tmp_pkg_l->process[i].nb_thread = nthread;
                }
            }
            /* If nthread still =0 -> use all cores, 1 pu per core -> nworker = nb_core */
            if (nthread == 0){ 
                tmp_pkg_l->nb_worker = tmp_pkg_l->nb_core;
                tmp_int = tmp_pkg_l->nb_core;    /* Keep track of cores left to assign */
                for(i=0; i<tmp_pkg_l->nb_process; i++){
                    tmp_pkg_l->process[i].nb_thread = (int) (tmp_int / tmp_pkg_l->nb_process - i);
                    tmp_int = tmp_int - tmp_pkg_l->process[i].nb_thread;
                    if(tmp_int % (tmp_pkg_l->nb_process - i) > 0){
                        tmp_pkg_l->process[i].nb_thread++;
                        tmp_int--;
                    }
                }
            }
        }
        #ifdef __DEBUG
        fprintf(stderr, "\tPackage%d Workers: %d\n", tmp_pkg_l->index, tmp_pkg_l->nb_worker);
        for(i=0; i<tmp_pkg_l->nb_process; i++){
            fprintf(stderr, "\t\tProcess%d Threads: %d\n", tmp_pkg_l->process[i].local_id, tmp_pkg_l->process[i].nb_thread);
        }
        #endif /* __DEBUG */
        /* next package */
        tmp_pkg_l = tmp_pkg_l->next;
    }
}

void mpibind_mappind_depth_per_package(hwloc_topology_t topology, hwloc_pkg_l **pkg_l, int smt){
    int i, tmp_int;
    hwloc_obj_t hwloc_obj;
    hwloc_pkg_l *tmp_pkg_l;

    #ifdef __DEBUG
    fprintf(stderr, "\n*** Highest topology object to map to:\n");
    #endif /* __DEBUG */
    tmp_pkg_l = *pkg_l;          /* Used to traverse the package list */
    while(tmp_pkg_l != NULL){
        if(tmp_pkg_l->nb_process == 0) break;
        else if(tmp_pkg_l->nb_worker == 1){
            tmp_pkg_l->mapping_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);
        }
        else if(smt > 0){
            tmp_pkg_l->mapping_depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
        }
        else{
            tmp_int   = hwloc_get_type_depth(topology, HWLOC_OBJ_PACKAGE);  /* Used to keep track of depth */
            while(tmp_int < hwloc_topology_get_depth(topology)){
                int nb_children; nb_children = 0;
                hwloc_obj = NULL;
                /* Get all objects at depth */
                for(i=0; i<hwloc_get_nbobjs_by_depth(topology, tmp_int); i++){
                    hwloc_obj = hwloc_get_next_obj_by_depth(topology, tmp_int, hwloc_obj);
                    /* Check if current package is parent of this node */
                    if(hwloc_obj_is_in_subtree(topology, hwloc_obj, tmp_pkg_l->pkg)) nb_children++;
                }
                /* Enough nodes at depth for workers ? */
                if(nb_children >= tmp_pkg_l->nb_worker){
                    tmp_pkg_l->mapping_depth = tmp_int;
                    break;
                }
                tmp_int++;
            }
        }

        #ifdef __DEBUG
        char *string; string = malloc(256*sizeof(char));
        hwloc_obj = hwloc_get_next_obj_by_depth(topology, tmp_pkg_l->mapping_depth, NULL);
        hwloc_obj_type_snprintf(string, 256, hwloc_obj, 0);
        fprintf(stderr, "\tPackage%d: nb_process:%d nb_worker:%d depth: %d - %s\n", 
            tmp_pkg_l->index, tmp_pkg_l->nb_process, tmp_pkg_l->nb_worker, 
            tmp_pkg_l->mapping_depth, string);
        free(string);
        #endif /* __DEBUG */

        /* next package */
        tmp_pkg_l = tmp_pkg_l->next;
    }
}

void mpibind_create_cpuset(hwloc_topology_t topology, hwloc_pkg_l **pkg_l, int smt){
    int i, ii, k, tst;
    hwloc_pkg_l *tmp_pkg_l;
    hwloc_obj_t hwloc_obj, hwloc_obj2, hwloc_obj3;
    hwloc_cpuset_l *tmp_cpuset_l;
    hwloc_bitmap_t tmp_cpuset;

    tmp_cpuset = hwloc_bitmap_alloc();

    #ifdef __DEBUG
    fprintf(stderr, "\n*** Creating cpusets:\n");
    #endif /* __DEBUG */
    tmp_pkg_l = *pkg_l;         /* Used to traverse the package list */
    while(tmp_pkg_l != NULL){
        hwloc_obj = NULL;       /* Used to traverse the objects at mapping depth */
        tmp_cpuset_l = tmp_pkg_l->cpuset_l;
        /* Find the first object at mapping depth from the current package */
        tst = 0;
        while(tst == 0){
            /* Check if current package is parent of this node */
            hwloc_obj = hwloc_get_next_obj_by_depth(topology, tmp_pkg_l->mapping_depth, hwloc_obj);
            if(hwloc_obj_is_in_subtree(topology, hwloc_obj, tmp_pkg_l->pkg)) tst = 1;
        }
        /* Note that hwloc_obj is now the first object to add to the first cpuset */
        /* Create a new cpuset for each process in the package */
        for(i=0; i<tmp_pkg_l->nb_process; i++){
            /* Create new cpuset list element */
            hwloc_cpuset_l *cpuset_elem; cpuset_elem = NULL;
            cpuset_elem = malloc(sizeof(hwloc_cpuset_l));
            /* Fill element */
            cpuset_elem->next = NULL;
            cpuset_elem->cpuset = hwloc_bitmap_alloc();
            hwloc_bitmap_zero(cpuset_elem->cpuset);
            /* Fill the cpusets*/
            /* Get the first child cpu of the proviously found object */
            if(hwloc_obj->type != HWLOC_OBJ_CORE && hwloc_obj->type != HWLOC_OBJ_PU){
            /* Tried something clever but could not make it work:
             *   here it looks at all pus and find the first one that is included in the package's cpuset */
            //    hwloc_obj2 = hwloc_get_next_child (topology, hwloc_obj, NULL);
            //    while(hwloc_obj2->type != HWLOC_OBJ_CORE || hwloc_obj2->type != HWLOC_OBJ_PU){
            //        hwloc_obj2 = hwloc_get_next_child (topology, hwloc_obj, hwloc_obj2);
            //    }
            //}
                  hwloc_obj_type_t type;
                  if(tmp_pkg_l->mapping_depth == hwloc_get_type_depth(topology, HWLOC_OBJ_PU)){
                      type = HWLOC_OBJ_PU;
                  } else type = HWLOC_OBJ_CORE;
                  hwloc_obj2 = hwloc_get_next_obj_by_depth(topology, hwloc_get_type_depth(topology, type), NULL);
                  tst = 0;
                  while(tst == 0){
                      if(hwloc_obj_is_in_subtree(topology, hwloc_obj2, tmp_pkg_l->pkg)) tst = 1;
                      else hwloc_obj2 = hwloc_get_next_obj_by_depth(topology, hwloc_get_type_depth(topology, type), hwloc_obj2); 
                  }
            }
            else hwloc_obj2 = hwloc_obj;
            /* then put process->nb_thread core in the cpuset (1 pu per core) */
            for(ii = 0; ii<tmp_pkg_l->process[i].nb_thread ; ii++){
                hwloc_bitmap_copy(tmp_cpuset, hwloc_obj->cpuset);
                hwloc_bitmap_singlify(tmp_cpuset);
                /* Add the obj's cpuset to the process' cpuset */
                hwloc_bitmap_or(cpuset_elem->cpuset, cpuset_elem->cpuset, tmp_cpuset);
                /* If smt is defined we want to use more than one hardware thread.
                 * We can add more from here by looking at all the pus from the selected cores
                 * We added the first pu before the smt loop in case there is no pu in the core */
                if(smt < 1) smt = 1; /* To simplify the allocation, if smt was not defined by user it is now set to 1 */
                /* Two cases possible:
                 *   - The mapping depth was higher than PU: the hwloc_obj found is a core, and we can allocate smt pu on it
                 *   - The mapping depth was already pu:  ???? */
                if(hwloc_obj->type == HWLOC_OBJ_CORE){
                    hwloc_obj3 = hwloc_get_next_child (topology, hwloc_obj2, NULL);
                    if(!hwloc_obj3) break;
                    for(k=0; k<smt-1; k++){
                        /* Get PU */
                        hwloc_obj3 = hwloc_get_next_child (topology, hwloc_obj2, hwloc_obj3);
                        if(!hwloc_obj3) break;
                        /* Add PU to cpuset */
                        hwloc_bitmap_copy(tmp_cpuset, hwloc_obj3->cpuset);
                        hwloc_bitmap_singlify(tmp_cpuset);
                        /* Add the obj's cpuset to the process' cpuset */
                        hwloc_bitmap_or(cpuset_elem->cpuset, cpuset_elem->cpuset, tmp_cpuset);
                    }
                }
                else{ /* Mapping depth was already pu */
                    /* TODO: I think we do nothing here */
                }
                /* Get the new object for next iteration or next process */
                hwloc_obj = hwloc_get_next_obj_by_depth(topology, tmp_pkg_l->mapping_depth, hwloc_obj);
            }
            /* Add element to the package cpuset list */
            if(tmp_pkg_l->cpuset_l == NULL){
                tmp_pkg_l->cpuset_l = cpuset_elem;
                tmp_cpuset_l = cpuset_elem;
            }else{
                tmp_cpuset_l->next = cpuset_elem;
                tmp_cpuset_l = tmp_cpuset_l->next;
            }
            #ifdef __DEBUG
            char *string;
            hwloc_bitmap_list_asprintf(&string, cpuset_elem->cpuset);
            fprintf(stderr, "\tPackage%d: Process%d: cpuset:%s\n", tmp_pkg_l->index, i, string);
            free(string);
            #endif /* __DEBUG */
        }
        tmp_cpuset_l = tmp_pkg_l->cpuset_l;
        while(tmp_cpuset_l != NULL){
            tmp_cpuset_l = tmp_cpuset_l->next;
        } 
        /* next package */
        tmp_pkg_l = tmp_pkg_l->next;
    }
    hwloc_bitmap_free(tmp_cpuset);
}

void mpibind_gpu_list_init(hwloc_topology_t topology, hwloc_gpu_l **gpu_l){
    hwloc_gpu_l *tmp_gpu_l;
    hwloc_obj_t hwloc_obj;

    tmp_gpu_l = *gpu_l;   /* Used to keep track of the end of the gpu list */
    hwloc_obj = NULL;
    /* Go through every os device and find all gpus */
    while ((hwloc_obj = hwloc_get_next_osdev(topology, hwloc_obj)) != NULL){
        /* check if object is a GPU device */
        if(hwloc_obj->attr->osdev.type == HWLOC_OBJ_OSDEV_GPU){
            /* Create new gpu_l element and appen to the list */
            hwloc_gpu_l *new; new = NULL;
            new = malloc(sizeof(hwloc_gpu_l));
            new->next = NULL;
            new->gpu = hwloc_obj;
            /* Append to the list (easy because we keep track of the last element in the list */
            if(tmp_gpu_l == NULL){
                *gpu_l = new;
                tmp_gpu_l = new;
            }
            else{
                tmp_gpu_l->next = new;
                tmp_gpu_l = tmp_gpu_l->next;
            }
        }
        /* next object */
        //hwloc_obj = hwloc_get_next_osdev(topology, hwloc_obj);
    }
    #ifdef __DEBUG
    fprintf(stderr, "\n*** GPU list creation:\n\t");
    tmp_gpu_l = *gpu_l;
    while(tmp_gpu_l != NULL){
        fprintf(stderr, "%s ", tmp_gpu_l->gpu->name);
        tmp_gpu_l = tmp_gpu_l->next;
    }
    fprintf(stderr, "\n");
    #endif /* __DEBUG */
}

void mpibind_gpu_list_destroy(hwloc_gpu_l **gpu_l){
    hwloc_gpu_l *tmp_gpu_l, *tmp_gpu_l_2;
    tmp_gpu_l = *gpu_l;
    while(tmp_gpu_l != NULL){
        tmp_gpu_l_2 = tmp_gpu_l;
        tmp_gpu_l   = tmp_gpu_l->next;
        free(tmp_gpu_l_2);
    }
}

void mpibind_assign_gpu(hwloc_topology_t topology, hwloc_pkg_l **pkg_l, hwloc_gpu_l **gpu_l){
    hwloc_pkg_l *tmp_pkg_l;
    hwloc_gpu_l *tmp_gpu_l, *tmp_gpu_l_2;
    hwloc_obj_t hwloc_obj;

    #ifdef __DEBUG
    fprintf(stderr, "\n*** GPU Assign:\n");
    #endif /* __DEBUG */
    hwloc_obj = NULL;

    /* Go through all packages and assign the right gpu (common ancestor is of type group) */
    tmp_pkg_l = *pkg_l;        /* Used to traverse the package list */
    while(tmp_pkg_l != NULL){
        /* Check each gpu */
        tmp_gpu_l = *gpu_l;
        tmp_gpu_l_2 = tmp_pkg_l->gpu_l;   /* Used to keep track of the end of the gpu list */
        while(tmp_gpu_l != NULL){
            /* Not working */
            //hwloc_obj = hwloc_get_common_ancestor_obj(topology, tmp_pkg_l->pkg, tmp_gpu_l->gpu);
            //if(hwloc_obj->type == HWLOC_OBJ_GROUP){
            /* Instead: find the first 'group obj' ancestor */
            /* Check if package is in it's subtree */
            hwloc_obj = tmp_gpu_l->gpu;
            while(hwloc_obj->type != HWLOC_OBJ_GROUP) hwloc_obj = hwloc_obj->parent;
            if(hwloc_obj_is_in_subtree(topology, tmp_pkg_l->pkg, hwloc_obj)){
                /* Create new gpu_l element for the package list */
                hwloc_gpu_l *new; new = NULL;
                new = malloc(sizeof(hwloc_gpu_l));
                new->next = NULL;
                new->gpu = tmp_gpu_l->gpu;
                /* Append to the list (easy because we keep track of the last element in the list */
                if(tmp_gpu_l_2 == NULL){
                    tmp_pkg_l->gpu_l = new;
                    tmp_gpu_l_2 = tmp_pkg_l->gpu_l;
                }
                else{
                    tmp_gpu_l_2->next = new;
                    tmp_gpu_l_2 = tmp_gpu_l_2->next;
                }
            }
            /* Next gpu */
            tmp_gpu_l = tmp_gpu_l->next;
        }
        /* If this package's gpu list is still empty at the end: assign all gpus */
        if(tmp_pkg_l->gpu_l == NULL) {
            tmp_pkg_l->gpu_l = *gpu_l;
        }
        #ifdef __DEBUG
        fprintf(stderr, "\tPackage%d GPU list:", tmp_pkg_l->index);
        tmp_gpu_l_2 = tmp_pkg_l->gpu_l;   /* Used to keep track of the end of the gpu list */
        while(tmp_gpu_l_2 != NULL){
            fprintf(stderr, " %s", tmp_gpu_l_2->gpu->name);
            tmp_gpu_l_2 = tmp_gpu_l_2->next;
        }
        fprintf(stderr, "\n");
        #endif /* __DEBUG */
        /* next package */
        tmp_pkg_l = tmp_pkg_l->next;
    }
    #ifdef __DEBUG
    fprintf(stderr, "\n");
    #endif /* __DEBUG */
}

void mpibind_format_output(mpibind_t *mh, hwloc_pkg_l *pkg_l, hwloc_gpu_l *gpu_l){
    int i, counter_th, counter_cpuset, counter_gpu;
    hwloc_pkg_l *tmp_pkg_l;
    hwloc_cpuset_l *tmp_cpuset_l;
    hwloc_gpu_l *tmp_gpu_l;
    mpibind_gpu_list *ptr;

    /* Malloc output objects */
    mh->cpuset      = malloc(mh->local_nprocess*sizeof(hwloc_bitmap_t));
    mh->gpu_l       = malloc(mh->local_nprocess*sizeof(mpibind_gpu_list));
    /* The number of thread per task is stored in pkg_l->process[x].nb_thread */
    mh->num_thread  = malloc(mh->local_nprocess*sizeof(int));

    counter_th      = 0;
    counter_cpuset  = 0;
    counter_gpu     = 0;

    tmp_pkg_l = pkg_l;          /* Used to traverse the package list */
    while(tmp_pkg_l){
        /* Set num_thread */
        for(i=0; i<tmp_pkg_l->nb_process; i++){
            mh->num_thread[counter_th] = tmp_pkg_l->process[i].nb_thread;
            counter_th++;
        }
        /* Set cpusets */
        tmp_cpuset_l = tmp_pkg_l->cpuset_l; /* Used to traverse cpuset list */
        while(tmp_cpuset_l){
            mh->cpuset[counter_cpuset] = hwloc_bitmap_alloc();
            hwloc_bitmap_copy(mh->cpuset[counter_cpuset], tmp_cpuset_l->cpuset);
            /* Next cpuset */
            tmp_cpuset_l = tmp_cpuset_l->next;
            counter_cpuset++;
        }
        /* Set Gpus */
        for(i=0; i<tmp_pkg_l->nb_process; i++){
            /* Gpus are the same for all process present on a package */
            tmp_gpu_l    = tmp_pkg_l->gpu_l;    /* Used to traverse gpu list */
            mh->gpu_l[counter_gpu] = NULL;
            ptr = NULL;
            while(tmp_gpu_l){
                /* add_gpu(tmp_gpu_l, mh->gpu_l[counter_gpu]); */
                if(ptr == NULL){
                    mh->gpu_l[counter_gpu] = malloc(sizeof(mpibind_gpu_list));
                    ptr = mh->gpu_l[counter_gpu];
                    mh->gpu_l[counter_gpu]->gpu = tmp_gpu_l->gpu;
                    mh->gpu_l[counter_gpu]->next = NULL;
                }
                else{
                    ptr->next = malloc(sizeof(mpibind_gpu_list));
                    ptr = ptr->next;
                    ptr->gpu = tmp_gpu_l->gpu;
                    ptr->next = NULL;
                }
                /* Next gpu */
                tmp_gpu_l = tmp_gpu_l->next;
            }
            counter_gpu++;
        }
        /* next package */
        tmp_pkg_l = tmp_pkg_l->next;
    }
}


/****                         PRINT FUNCTIONS                            ****/
void mpibind_print_pkg_l(hwloc_pkg_l *head){
    int ii;
    char *string;
    hwloc_cpuset_l *cpuset_l;
    hwloc_gpu_l *gpu_l;

    fprintf(stdout, "#### Mpibind binding: ####\n\n");
    while(head != NULL){
        /* Basic infos */
        fprintf(stdout, "## Package%d  nb_process:%d, nb_core:%d, nb_pu:%d, mapping_depth:%d\n",
                head->index, head->nb_process, head->nb_core, head->nb_pu, head->mapping_depth);
        /* Process specific infos */
        cpuset_l = head->cpuset_l;      /* Keep track of process's cpusets */
        gpu_l = head->gpu_l;            /* Keep track of process's gpu(s) */
        for(ii=0; ii<head->nb_process; ii++){
            /* Process infos */
            fprintf(stdout, "\tProcess%d Threads: %d\n", head->process[ii].local_id, head->process[ii].nb_thread);
            /* cpuset */
            hwloc_bitmap_list_asprintf(&string, cpuset_l->cpuset);
            fprintf(stdout, "\t\tcpuset_l: %s\n", string);
            cpuset_l = cpuset_l->next;
            free(string);
            /* GPU(s) */
            fprintf(stdout, "\t\tgpu_l:");
            while(gpu_l != NULL){
                fprintf(stdout, " %s", gpu_l->gpu->name);
                gpu_l = gpu_l->next;
            }
            fprintf(stdout, "\n");
        }
        /* Next package */
        head = head->next;
    }
}
