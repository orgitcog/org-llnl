#include "core/include/mpibind.h"
#include "core/include/mpibind_functions.h"

#include <stdlib.h>

/****************************************************************************/
/*                               FUNCTIONS                                  */
/****************************************************************************/

/* Create an instance of the mpibind handle
 **/
mpibind_t *mpibind_create (void){
    mpibind_t *handle;
    handle = malloc(sizeof(mpibind_t));

    handle->verbose             = -1;
    handle->user_smt            = -1;
    handle->user_num_thread     = -1;
    handle->omp_proc_bind       = -1;

    handle->cpuset              = NULL;
    handle->gpu_l               = NULL;
    handle->num_thread          = NULL;

    return handle;
}

/* Destroy mpibind handle and free associated memory
 **/
void mpibind_destroy (mpibind_t *mh){
    int i;
    mpibind_gpu_list *ptr;

    /* Free GPU list */
    for(i=0; i<mh->local_nprocess; i++){
        while(mh->gpu_l[i]){
            ptr = mh->gpu_l[i];
            mh->gpu_l[i] = mh->gpu_l[i]->next;
            free(ptr);
        }
    }
    free(mh->gpu_l);

    /* Free num_thread arrays */
    free(mh->num_thread);

    /* Free cpuset array */
    for(i=0; i<mh->local_nprocess; i++){
        hwloc_bitmap_free(mh->cpuset[i]);
    }
    free(mh->cpuset);


    /* free handle */
    free(mh);
}

/****************************************************************************/
/*                                SETTERS                                   */
/****************************************************************************/

int mpibind_set_topology (mpibind_t *mh, hwloc_topology_t topo){
    if (topo != NULL){
        mh->topo = topo;
        return 0;
    }
    return 1;
}

int mpibind_set_local_nprocess (mpibind_t *mh, int nprocess){
    if(nprocess > 0){
        mh->local_nprocess = nprocess;
        return 0;
    }
    else{
        mh->local_nprocess = 1;
        return 0;
    }
    return 1;
}

int mpibind_set_verbose (mpibind_t *mh, int verbose){
    if (verbose > 0){ 
        mh->verbose = 1;
        return 0;
    }
    else{ 
        mh->verbose = -1; 
        return 0;
    }
    return 1;
}

int mpibind_set_user_smt (mpibind_t *mh, int smt){
    if(smt > 0){
        mh->user_smt = smt;
        return 0;
    }
    else{
        mh->user_smt = 0;
        return 0;
    }
    return 1;
}

int mpibind_set_user_num_thread  (mpibind_t *mh, int num_thread){
    if(num_thread > 0){
        mh->user_num_thread = num_thread;
        return 0;
    }
    else if(num_thread == -1){
        mh->user_num_thread = num_thread;
        return 0;
    }
    else{
        mh->user_num_thread = 0;
        return 0;
    }
    return 1;
}

int mpibind_set_omp_proc_bind_provided (mpibind_t *mh){
    mh->omp_proc_bind = 1;
    return 0;
}

/****************************************************************************/
/*                                GETTERS                                   */
/****************************************************************************/

hwloc_topology_t mpibind_get_topology (mpibind_t *mh){
    return mh->topo;
}

int mpibind_get_local_nprocess (mpibind_t *mh){
    return mh->local_nprocess;
}

int mpibind_get_verbose (mpibind_t *mh){
    return mh->verbose;
}

int mpibind_get_user_smt (mpibind_t *mh){
    return mh->user_smt;
}

int mpibind_get_user_num_thread (mpibind_t *mh){
    return mh->user_num_thread;
}

int mpibind_get_omp_proc_bind_provided (mpibind_t *mh){
    return mh->omp_proc_bind;
}

hwloc_bitmap_t *mpibind_get_cpusets (mpibind_t *mh){
    return mh->cpuset;
}

mpibind_gpu_list **mpibind_get_gpus (mpibind_t *mh){
    return mh->gpu_l;
}

int *mpibind_get_num_thread (mpibind_t *mh){
    return mh->num_thread;
}

/****************************************************************************/
/*                                  PRINT                                   */
/****************************************************************************/

void mpibind_print(mpibind_t *mh){
    int i;
    mpibind_gpu_list *tmp_gpu;

    char string[2048], tmp[256];
    char *tmp2;

    for(i=0; i<mpibind_get_local_nprocess(mh); i++){
        sprintf(tmp, "rank %d cpu ", i);
        strcat(string, tmp);

        /* cpuset */
        hwloc_bitmap_list_asprintf(&tmp2, mh->cpuset[i]);
        strcat(string, tmp2);

        /* gpu */
        tmp_gpu = mh->gpu_l[i];
        if(tmp_gpu){
            sprintf(tmp, " gpu ");
            strcat(string, tmp);
        }
        while(tmp_gpu){
            sprintf(tmp, "%s ", tmp_gpu->gpu->name);
            strcat(string, tmp);
            tmp_gpu = tmp_gpu->next;
        }

        /* print */
        printf("%s\n", string);
        string[0] = '\0';

        free(tmp2);
    }
}

/****************************************************************************/
/*                                  MAIN                                    */
/****************************************************************************/

/* Main mpibind function */
mpibind_t *mpibind (mpibind_t *mh){
    hwloc_pkg_l *pkg_l;
    hwloc_gpu_l *gpu_l;

    pkg_l = NULL; gpu_l = NULL;

    /* Deprecated */
    ///**** Get function options from the MPIBIND environment variable ****/
    //mpibind_gather_options();
    ///**** Create and retrive topology ****/
    //topology = mpibind_get_topology();
    
    /**** Get the number of packages ****/
    mpibind_get_package_number(mpibind_get_topology(mh));

    /**** Determine the number of cores and pu per package ****/
    /**** Also determine the number of process per package ****/
    mpibind_package_list_init(mpibind_get_topology(mh), &pkg_l, mpibind_get_local_nprocess(mh));

    /**** Get number of threads with OMP_NUM_THREADS or function arguments ****/
    mpibind_compute_thread_number_per_package(mpibind_get_topology(mh), &pkg_l, mpibind_get_user_num_thread(mh));

    /**** For each package find the highest topology level such that nvertices >= nb_worker ****/
    mpibind_mappind_depth_per_package(mpibind_get_topology(mh), &pkg_l, mpibind_get_user_smt(mh));

    /**** Create process cpusets ****/
    mpibind_create_cpuset(mpibind_get_topology(mh), &pkg_l, mpibind_get_user_smt(mh));

    /*** GPU management ***/
    mpibind_gpu_list_init(mpibind_get_topology(mh), &gpu_l);
    mpibind_assign_gpu(mpibind_get_topology(mh), &pkg_l, &gpu_l);

    /**** Format the mh handle to contain the mpibind result allocation ****/
    mpibind_format_output(mh, pkg_l, gpu_l);

    /**** Print results****/
    if(mpibind_get_verbose(mh)) mpibind_print(mh);
    
    /**** Free stuff  ****/
    /* Free package list */
    mpibind_package_list_destroy(&pkg_l);
    /* Free the full gpu list */
    mpibind_gpu_list_destroy(&gpu_l);
    
    return mh;
}
