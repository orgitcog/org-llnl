"""
Used to do some postprocessing on the full run of stats output for codeforces data. Assumes output has already been merged.

Workflow:
    1. Run with task being 'partition' to partition the data into one file per problem. This will search through each
       merged file grabbing all stats involved with one problem at a time, and save those stats as well as their submission_id's
       right next to each other in a compressed npz file (with array names 'stats' and 'submission_ids'). This can be run 
       as multiple jobs with the --n_jobs parameter. Files will be saved in output_dir.
    2. Run with task being 'select' to make selections of id's per problem. You can change various parameters
       involving the selection methods like --size to change the number of ids per problem, --selection_method to change
       the function to use to select subsets, etc. Files will be saved in output_dir
"""
import argparse
import os
import re
import random
import pickle
import sys
import pandas as pd
import numpy as np
import multiprocessing
import numba
from bincfg import progressbar


# Number of extra bytes to add to the number of tokens to determine the max number of columns in the stats arrays
# This should be >= to the exact number, but no lower. Closer it is, the less wasted memory
NUM_EXTRA_STATS_BYTES = 100

# State to randomly shuffle problem_uids when selecting subsets for each job (that way, it is unlikely we will get
#   jobs with multiple very large number of submissions). Must be the same across all jobs
RS_STATE = 1234567


def partition_data(sub_info_path, merged_dir, output_dir, tokens_path, n_jobs=1, job_id=0):
    """Partitions already merged data by problem

    Will output data as a .npz compressed numpy file the stats in the 'stats' name, and submission_id's using the name
    'submission_ids'. The filename will be: PROBLEM_UID.npz where PROBLEM_UID is the problem uid of that problem (IE:
    'CONTEST_ID-PROBLEM_ID').

    Ensures that all output stats arrays will be the same number of columns.

    This will only keep the 'id' column of the df files along with the stats.
    
    Args:
        sub_info_path (str): path to the submission info parquet file
        merged_dir (str): directory containing all of the merged files matching the basenames: 
            'RUN_NAME-merged-NUMBER-df.parquet' and 'RUN_NAME-merged-NUMBER-stats.npz' for the submission_ids and stats
            array respectively, where RUN_NAME is the name of the run, and NUMBER is the index of that output file
        output_dir (str): directory to output files. If it doesn't exist, it will be created
        tokens_path (str): path to the dictionary of tokens that were used to generate the stats
        n_jobs (Optional[int]): the number of jobs to run as
        job_id (Optional[int]): the id of this job
    """
    # Load in the submission info
    sub_info = pd.read_parquet(sub_info_path, columns=['submission_id', 'contest_id', 'problem_id'])
    sub_info['problem_uid'] = pd.Categorical(sub_info.contest_id.astype(str) + '-' + sub_info['problem_id'].astype(str))
    sub_info = sub_info.drop(['contest_id', 'problem_id'], axis=1)

    print("There are %d total jobs, of which this is index %d" % (n_jobs, job_id))

    # Figure out which problems this job will be doing, and the submission_id's for them
    problem_uids = list(sub_info.problem_uid.unique())
    random.seed(RS_STATE)
    random.shuffle(problem_uids)
    problem_uids = np.array_split(problem_uids, n_jobs)[job_id]
    sub_ids = sub_info[sub_info.problem_uid.isin(problem_uids)]

    print("Getting %d uid's (total of %d submission_id's)" % (len(problem_uids), len(sub_ids)))

    # Figure out the maximum size of the stats arrays
    with open(tokens_path, 'rb') as f:
        num_tokens = len(pickle.load(f))

    # Create a random load order of files so that we can mitigate the problem of too many jobs trying to load the same
    #   file all at once
    df_files = [os.path.join(merged_dir, f) for f in os.listdir(merged_dir) if re.fullmatch(r'.*-merged-[0-9]+-df.(?:pq|parquet)', f)]
    random.shuffle(df_files)

    print("Found %d files to search..." % len(df_files))

    # Iterate through the files grabbing all submissions with any of the currently partitioning problem_uids
    # keep_stats is a dictionary mapping problem_uid to lists of (submission_ids: 1-d np.uint32 numpy array,
    #   stats: 2-d np.uint8 array, idx: int [keeps track of the index through the array we are at])
    # keep_col is the number of columns in the largest stats array we have found, that way we can ensure all arrays
    #   are the same size at the end
    keep_stats = {
        uid: [np.zeros([len(group)], dtype=np.uint32), np.zeros([len(group), num_tokens + NUM_EXTRA_STATS_BYTES], dtype=np.uint8), 0]
        for uid, group in sub_ids.groupby('problem_uid', observed=True)
    }
    keep_col = 0
    for file in progressbar(df_files):

        # Read in the submission_id's and reset the index. Then, select only the submission_ids we are handling, and 
        # merge with the sub_id to get the problem_uid. The index value should still correspond to the location in the np array
        chunk_df = pd.read_parquet(file, columns=['id']).reset_index(drop=True)
        chunk_df = chunk_df[chunk_df.id.isin(sub_ids.submission_id)].merge(sub_ids, how='left', left_on='id', right_on='submission_id')
        
        # Read in the numpy array, and check if the new number of columns is larger
        chunk_arr = np.load(file.replace('-df.pq', '-stats.npz').replace('-df.parquet', '-stats.npz'))['data']
        keep_col = max(chunk_arr.shape[1], keep_col)

        # Go through each problem_uid in chunk_df groups and add them to their corresponding uid in the keep_stats
        # It should only contain problem_uid's we are using
        for uid, group in chunk_df.groupby('problem_uid', observed=True):
            sub_ids_arr, stats_arr, idx = keep_stats[uid]
            end_idx = idx + len(group)
            loc = group.index.to_numpy()

            sub_ids_arr[idx:end_idx] = group['id']
            stats_arr[idx:end_idx, :chunk_arr.shape[1]] = chunk_arr[loc]
            keep_stats[uid][2] = end_idx

    # Save all of the problem_uid's we are doing
    print("Saving files...")
    for uid, (sub_ids_arr, stats_arr, idx) in keep_stats.items():
        output_path = os.path.join(output_dir, '%s.npz' % uid)
        np.savez_compressed(output_path, submission_ids=sub_ids_arr[:idx], stats=stats_arr[:idx, :keep_col])

    print("All done!")


def select_subset(input_dir, output_dir, selection_method, selection_name, size, prefilter_func, sub_info_path, n_jobs, job_id):
    """Selects subsets of data. Assumes input_dir contains files partitioned by problem_uid
    
    Can do multiple different selection methods:

        - 'random': uniform random choice over all submissions
        - 'distant': pick the most distant choices based on stats (manhattan distance between stats vectors)
    
    Will save output as a 1-d np.uint32 numpy array of submission_id's to keep. Outputs as a '.npy' file to the path:
    output_dir/selection_name-job_id.npy
    
    Args:
        input_dir (str): directory containing all of the partitioned files. Names should be: PROBLEM_UID.npz, where
            PROBLEM_UID is the problem_uid for that problem, and each file is a compressed numpy zip file with
            sub-arrays 'stats' (containing the 2-d np.uint8 stats array), and 'submission_id' (containing the 1-d
            np.uint32 submission_id array)
        output_dir (str): the directory to output file to. Will output as '.npy' file
        selection_method (str): the method to use to select subsets. Can be 'random' or 'distant' (see above for 
            explanations)
        selection_name (str): string name for this selection
        size (int): the number of examples to get per problem. If the problem has < size number of examples, then all
            will be kept
        prefilter_func (Optional[Callable]): a function to use to prefilter the data at each problem before selecting a 
            subset. Should take as input an interable of submission_id's and the string path to the codeforces submission
            info, and return an interable of submission_id's to keep, or None to keep all submission_id's. If 
            `prefilter_func` itself is None, then all submission_id's will be kept.
        sub_info_path (Optional[str]): the path to the submission info parquet file, or None if not using
        n_jobs (Optional[int]): the number of jobs to run as
        job_id (Optional[int]): the id of this job
    """
    # Make sure values are good
    selection_method = selection_method.lower()
    if selection_method in ['random', 'uniform']:
        selection_method = 'random'
    elif selection_method in ['distant', 'max_distance', 'furthest', 'apart']:
        selection_method = 'distant'
    else:
        raise ValueError("Unknown selection_method: %s" % repr(selection_method))
    
    # Make sure the output folder exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Determine which files we will be filtering
    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if re.fullmatch(r'[0-9]+-.+[.]npz', f)]
    files = np.array_split(files, n_jobs)[job_id]

    print("This job will be reading %d files, using selection_method=%s, prefilter_func=%s" % 
        (len(files), selection_method, prefilter_func.__name__))

    # Go through each problem filtering it down
    all_ids = []
    for fidx, f in enumerate(files):
        print("Reading file %d / %d: %s" % (fidx + 1, len(files), f))
        
        # Read in the arrays
        loaded = np.load(f)
        sub_ids = loaded['submission_ids']

        print("Contains %d submissions to filter" % len(sub_ids))

        # Prefilter if needed
        keep_locs = None
        if prefilter_func is not None:
            keep_ids = prefilter_func(sub_ids, sub_info_path)
            if keep_ids is not None:
                keep_locs = np.isin(sub_ids, keep_ids)
                sub_ids = sub_ids[keep_locs]
        
        # Select a subset of the ids to keep
        if len(sub_ids) < size:
            all_ids.append(sub_ids)

        elif selection_method == 'random':
            np.random.seed(RS_STATE)
            all_ids.append(np.random.choice(sub_ids, size=[size], replace=False))

        elif selection_method == 'distant':

            # Need to make sure we prefilter if using
            stats = loaded['stats']
            if keep_locs is not None:
                stats = stats[keep_locs]

            all_ids.append(_distant_select(sub_ids, stats, size))
        
        else:
            raise NotImplementedError
    
    print("Finished filtering! Concatenating values...")
    
    # Save the output to the decided path
    output_arr = np.concatenate(all_ids, axis=0, dtype=np.int32)
    output_path = os.path.join(output_dir, '%s-%d.npy' % (selection_name, job_id))
    np.save(output_path, output_arr)

    print("Saved file to:", output_path)

    # Return it if using
    return output_arr


_DF_CACHE = None
def prefilter_ok_verdict(sub_ids, sub_info_path):
    """Prefilters submissions to only those with an 'OK' verdict"""
    global _DF_CACHE
    if _DF_CACHE is None:
        print("Loading submission_info...")
        _DF_CACHE = pd.read_parquet(sub_info_path, columns=['submission_id', 'verdict'])
        print("Submission_info loaded!")
    return _DF_CACHE[_DF_CACHE.submission_id.isin(sub_ids) & (_DF_CACHE.verdict == 'OK')].submission_id.to_numpy()


@numba.njit()
def _distant_select(sub_ids, stats, size):
    """Selects submissions that maximize the minimum distance between them
    
    Assumes np.random has already been seeded if you care about reproducibility

    Assumes size < len(sub_ids)
    """

    # Keep track of all distances/kept points
    min_distances = np.full(stats.shape[0], 2**62)
    last_measured = np.zeros(stats.shape[0], dtype=np.int32)  # Keep track of the last point measured against for each point
    points = np.empty((size, stats.shape[1]))
    selected_ids = np.full((size,), -1, dtype=np.int32)
 
    # Pick the first point randomly
    rand_first = np.random.randint(0, len(stats))
    points[0] = stats[rand_first]
    selected_ids[0] = sub_ids[rand_first]
 
    # Compute the distances from all points to original point first
    min_distances = np.sum(np.abs(stats - points[0]), axis=1)
    arg_furthest = np.argmax(min_distances)
    points[1] = stats[arg_furthest]
    selected_ids[1] = sub_ids[arg_furthest]
   
    # Loop through all new points
    for point_idx in range(2, size):
 
        # Go through all datapoints measuring the distances needed, keeping track of the current max min_distance
        curr_max = 0
        curr_idx = 0
        for compare_idx in range(len(stats)):
 
            # We only need to measure all values if this point's current min_distance is > curr_max
            if min_distances[compare_idx] > curr_max:
               
                # Iterate through all the rest of the points we need to measure against, checking that our current min
                #   distance is > curr_max
                for measure_idx in range(last_measured[compare_idx], point_idx):
                    new_distance = np.sum(np.abs(stats[compare_idx] - points[measure_idx]))
 
                    # Update this point's new min_distance since we've already computed it
                    if new_distance < min_distances[compare_idx]:
                        min_distances[compare_idx] = new_distance
                   
                    # Check to make sure this new_distance is still > curr_max to continue checking distances
                    if new_distance <= curr_max:
                        break
               
                # Update the index we have measured for this point
                last_measured[compare_idx] = measure_idx
           
            # Otherwise, we can continue
            else:
                continue
 
            # Now that we've checked more measurements, we need to check if we should still update our current max min_distance/point
            if min_distances[compare_idx] > curr_max:
                curr_max = min_distances[compare_idx]
                curr_idx = compare_idx
       
        # Now we have the idx of the point with the current largest distance, add it to points
        points[point_idx] = stats[curr_idx]
        selected_ids[point_idx] = sub_ids[curr_idx]
   
    return selected_ids


def merge_subsets(input_dir, delete=False):
    """Merges together processed subsets

    Merges all unique subset names into their own file in the same input directory.

    Assumes all files that need to be merged follow the format NAME-IDX.npy, where NAME is the string name of that
    subset, and IDX is the integer index. Also assumes each file is a 1-D iterable of submission_id's
    
    Args:
        input_dir (str): string path to the input directory
        delete (bool): if True, then the original files will be deleted once merged
    """
    print("Merging subsets...")

    files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if re.fullmatch(r'.*-[0-9]+[.]npy', f)]

    # Keep track of read files in a dictionary mapping subset name to list of numpy arrays
    output_arrs = {}
    for f in files:
        name = os.path.basename(f).rpartition('-')[0]
        output_arrs.setdefault(name, []).append(np.load(f))
    
    print("Found %d subset names to merge: %s" % (len(output_arrs), list(output_arrs.keys())))
    
    # Concatenate all the arrays and write them to files
    for k, arr in output_arrs.items():
        output_arrs[k] = np.concatenate(arr)
        np.save(os.path.join(os.path.dirname(files[0]), '%s-merged.npy' % k), output_arrs[k])
    
    # Delete the files if needed
    if delete:
        print("Deleting files...")
        for f in files:
            os.remove(f)
    
    print("All done!")


def _mp_call(task, sub_info_path, input_dir, output_dir, tokens_path, s_method, s_name, size, prefilter_func, n_jobs, 
    job_id, threads, thread_idx):
    """Split up like this for multiprocessing call"""
    # If we are using threads, update the n_jobs and job_id
    if threads > 1:
        n_jobs, job_id = n_jobs * threads, job_id * threads + thread_idx

    # Call the appropriate task
    if task in ['partition', 'partitioned', 'parts', 'part']:
        if sub_info_path is None:
            raise ValueError("Must pass --submission_info_path for --task=partition!")
        elif tokens_path is None:
            raise ValueError("Must pass --tokens_path for --task=partition!")
        partition_data(sub_info_path, input_dir, output_dir, tokens_path, n_jobs, job_id)
    elif task in ['select', 'selection', 'subset', 'subsets']:
        if sub_info_path is None and prefilter_func is not None:
            raise ValueError("Must pass --submission_info_path when using a prefilter_func!")
        select_subset(input_dir, output_dir, s_method, s_name, size, prefilter_func, sub_info_path, n_jobs, job_id)
    else:
        raise ValueError("Unknown task: %s" % repr(task))
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', action='store', help='The task to complete. Currently can be: \'partition\', \'select\', '
        ' or \'merge\' used to partition the already merged data, select subsets of submission_ids, or merge selected subsets '
        'into a single file.')
    parser.add_argument('input_dir', action='store', help='The directory containing input files')
    parser.add_argument('--output_dir', action='store', default=None, help='The directory to output files to')
    parser.add_argument('--submission_info_path', action='store', default=None, help='The path to the submission info parquet file')
    parser.add_argument('--tokens_path', action='store', default=None, help='The path to the tokens pickle file')
    parser.add_argument('--n_jobs', action='store', type=int, default=1, help='The number of jobs to use')
    parser.add_argument('--job_id', action='store', type=int, default=None, help='The id of this currently running job. '
        'If n_jobs > 1, then this id determines what job this instance is. Will be ignored if n_jobs == 1, and will '
        'default to checking the \'SLURM_ARRAY_TASK_ID\' environment variable if it is not passed.')
    parser.add_argument('--s_method', action='store', default='random', help='The selection_method to use when --task=\'select\'. '
        'Can be either \'random\' to select values uniformly randomly, or \'distant\' to select cfg\'s that are most '
        'distant from one another (largest minimum manhattan distance between stats vectors)')
    parser.add_argument('--s_name', action='store', default=None, help='The selection_name to use when --task=\'select\'. '
        'This is just a string name to use to identify this current selection run.')
    parser.add_argument('--s_size', action='store', type=int, default=100, help='The size to use when --task=\'select\'. '
        'This is the number of submission_id\'s to use per problem')
    parser.add_argument('--s_prefilter_func', action='store', default=None, help='The prefilter_func to use when --task=\'select\'. '
        'This is the name of a function to use to prefilter submission_id\'s before selection. See the documentation for '
        'more info. Can be the string name of any function that exists in the current global namespace, or None')
    parser.add_argument('--delete', action='store_true', default=False, help='If using task=\'merged\' and this arg is '
        'passed, then all of the original files will be deleted once they have been merged')
    parser.add_argument('--threads', action='store', type=int, default=1, help='Number of threads to use')
    
    args = parser.parse_args()

    task = args.task.lower()

    sub_info_path = args.submission_info_path
    if sub_info_path is not None:
        if not os.path.exists(sub_info_path):
            raise FileNotFoundError("Could not find submission info parquet file: %s" % sub_info_path)
        elif not os.path.isfile(sub_info_path):
            raise ValueError("Given submission info path is not a file: %s" % sub_info_path)

    tokens_path = args.tokens_path
    if tokens_path is not None:
        if not os.path.exists(tokens_path):
            raise FileNotFoundError("Could not find tokens pickle file: %s" % tokens_path)
        elif not os.path.isfile(tokens_path):
            raise ValueError("Given tokens path not a file: %s" % tokens_path)

    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        raise FileNotFoundError("Could not find directory: %s" % input_dir)
    elif not os.path.isdir(input_dir):
        raise ValueError("Given input_dir is not a directory: %s" % input_dir)

    # Check for merging subsets
    if task in ['merge', 'merged', 'merging']:
        merge_subsets(input_dir, args.delete)
        sys.exit(0)
    
    output_dir = args.output_dir
    if output_dir is None:
        raise ValueError("Must pass output_dir for subset/partition tasks!")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif not os.path.isdir(output_dir):
        raise ValueError("Given output_dir is not a directory: %s" % output_dir)
    
    n_jobs = args.n_jobs
    if n_jobs < 1:
        raise ValueError("n_jobs must be >= 1: %d" % n_jobs)
    
    job_id = args.job_id
    if n_jobs > 1:
        if job_id is None:
            try:
                job_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
            except KeyError:
                raise ValueError("Could not get job_id. Either pass it as a command line arg, or set it to the environment"
                    " variable SLURM_ARRAY_TASK_ID")
        if job_id < 0 or job_id >= n_jobs:
            raise ValueError("job_id must be in range [0, n_jobs) = [0, %d), got: %d" % (n_jobs, job_id))
    else:
        job_id = 0
    
    threads = args.threads
    if threads <= 0:
        raise ValueError("threads must be >= 1: %d" % threads)
    
    s_method = args.s_method.lower()
    if s_method not in ['random', 'distant']:
        raise ValueError("Unknown s_method: %s" % s_method)
    
    s_name = ('%s-select' % s_method) if args.s_name is None else args.s_name

    size = args.s_size
    if size <= 0:
        raise ValueError("'size' must be >= 1, got: %d" % size)
    
    try:
        prefilter_func = None if args.s_prefilter_func in [None, 'None', 'none'] else globals()[args.s_prefilter_func]
    except KeyError:
        raise ValueError("Unknown prefilter_func: %s" % repr(args.s_prefilter_func)) from None
    
    if threads == 1:
        _mp_call(task, sub_info_path, input_dir, output_dir, tokens_path, s_method, s_name, size, prefilter_func, n_jobs,
            job_id, threads, 0)
    else:
        print("Using %d threads!" % threads)
        with multiprocessing.Pool(threads) as pool:
            pool.starmap(_mp_call, [[task, sub_info_path, input_dir, output_dir, tokens_path, s_method, s_name, size, 
                prefilter_func, n_jobs, job_id, threads, i] for i in range(threads)])
