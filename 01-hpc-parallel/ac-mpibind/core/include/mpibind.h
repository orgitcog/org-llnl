#ifndef __MPIBIND_H__
#define __MPIBIND_H__

#include "hwloc.h"

/****************************************************************************/
/*                              STRUCTURES                                  */
/****************************************************************************/

struct mpibind_gpu_list_s{
    struct mpibind_gpu_list_s *next;
    hwloc_obj_t gpu;
};
typedef struct mpibind_gpu_list_s mpibind_gpu_list;

/* Opaque mpibind "handle" used to store otherwise global state.
 **  (options, hwloc_topology_t, etc.)
 **/
struct mpibind_handle{
    /* 'input' */
    hwloc_topology_t topo;    /* Activate verbose */
    int local_nprocess;       /* Number of process / mpi tasks that will be launched on the node */
    int verbose;              /* Activate verbose */
    int user_smt;             /* Multithread usage */
    int user_num_thread;      /* Number of threads per process */
    int omp_proc_bind;        /* Bool: is binding set by open mp?*/
    /* 'output' */
    hwloc_bitmap_t *cpuset;   /* Array of cpusets, one cpuset per task */
    mpibind_gpu_list **gpu_l; /* Array of gpu lists, one list per task */
    int *num_thread;          /* Array containing the number of threads per task */

};
typedef struct mpibind_handle mpibind_t;


/****************************************************************************/
/*                               FUNCTIONS                                  */
/****************************************************************************/

/* Create an instance of the mpibind handle
 **/
mpibind_t *mpibind_create (void);

/* Destroy mpibind handle and free associated memory
 **/
void mpibind_destroy (mpibind_t *mh);

///* Intiialize options from standard env vars */
//int mpibind_init_from_env (mpibind_t *mh);
//
///* Gather hwloc_topology_t in a default manner and store in handle */
//int mpibind_gather_topology (mpibind_t *mh);

/* Set (and/or get) various internal options */
int mpibind_set_topology (mpibind_t *mh, hwloc_topology_t topo);
int mpibind_set_local_nprocess (mpibind_t *mh, int nprocess);
int mpibind_set_verbose (mpibind_t *mh, int verbose);
int mpibind_set_user_smt (mpibind_t *mh, int smt);
int mpibind_set_user_num_thread (mpibind_t *mh, int num_thread);
int mpibind_set_omp_proc_bind_provided (mpibind_t *mh);

hwloc_topology_t mpibind_get_topology (mpibind_t *mh);
int mpibind_get_local_nprocess (mpibind_t *mh);
int mpibind_get_verbose (mpibind_t *mh);
int mpibind_get_user_smt (mpibind_t *mh);
int mpibind_get_user_num_thread (mpibind_t *mh);
int mpibind_get_omp_proc_bind_provided (mpibind_t *mh);
hwloc_bitmap_t *mpibind_get_cpusets (mpibind_t *mh);
mpibind_gpu_list **mpibind_get_gpus (mpibind_t *mh);
int *mpibind_get_num_thread (mpibind_t *mh);

///*  Bind process "procid" of "nprocs" assumed to have nthreads per process.
// **  If nthreads == 0, then the number of threads per process is unknown.
// **/
//int mpibind_bind (mpibind_t *mh, int nprocs, int nthreads, int procid);
//
///* Like hwloc_distribute(3), but use mpibind algorithm.
// **
// ** Variable 'set' must be an array of allocated hwloc bitmap/cpuset
// ** of at least nprocs size.
// **/
//int mpibind_distribute (mpibind_t *mh, int nprocs, int threads_per_process, hwloc_cpuset_t *set);

/* Print results in the from the mpibind handle
 **/
void mpibind_print(mpibind_t *mh);

/* Main mpibind function */
mpibind_t *mpibind (mpibind_t *mh);

#endif /* __MPIBIND_H__ */
