"""
Used to merge the CAP outputs together
"""

import pandas as pd
import os
import pickle
import re
import argparse
from bincfg import progressbar


def merge_outputs(output_path, chunksize):
    """Combines output parquet files"""
    files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith(('.parquet', '.pq')) and re.fullmatch(r'merged-[0-9]+.parquet', f) is None]

    curr_dfs = []
    merge_idx = 0
    for f in progressbar(files):
        try:
            curr_dfs.append(pd.read_parquet(f))
        except Exception as e:
            print("Failed on file, ignoring: %s" % repr(f))
            continue

        if len(curr_dfs) >= chunksize:
            num_in_chunk = len(curr_dfs)
            curr_dfs = pd.concat(curr_dfs)

            print("Current dataframe chunk consists of %d files and takes up %.4f GB of memory" 
                % (num_in_chunk, curr_dfs.memory_usage(deep=True).sum() / 2 ** 30))
            curr_dfs.to_parquet(os.path.join(output_path, "merged-%d.parquet" % merge_idx), index=False, row_group_size=200)

            curr_dfs = []
            merge_idx += 1


def merge_logs(cap_path, log_name, delete):
    """
    Merges all the logs together. Assumes all logs in the directory that start with a number are the correct logs to merge.
    Merges only the debug files.
    """
    log_path = os.path.join(cap_path, 'logs')
    debug_files = [os.path.join(log_path, f) for f in os.listdir(log_path) if f.endswith('_debug.log')]

    if log_name[0] in "0123456789":
        raise ValueError("Log name cannot start with a number")
    if not log_name.endswith('.log'):
        log_name = log_name + '.log'
    
    full_lines = []
    for f in progressbar(debug_files):
        with open(f, 'r') as _f:
            full_lines += _f.readlines()
    
    with open(os.path.join(log_path, log_name), 'w') as f:
        f.writelines(full_lines)
    
    if delete:
        for f in [os.path.join(log_path, f) for f in os.listdir(log_path) if f.endswith('_debug.log') or f.endswith('_error.log')]:
            os.remove(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine cap outputs')
    parser.add_argument('path', action='store', help='The path to the cap output')
    parser.add_argument('chunksize', action='store', type=float, help='The number of files to merge into a single chunk')
    parser.add_argument('--task', action='store', default='output', help='The merging task to do. Can be \'logs\' or \'output\'. Defaults to \'output\'')
    

    args = parser.parse_args()

    chunksize = args.chunksize if args.chunksize > 0 else 1000000000000000

    if args.task.lower() in ['logs', 'log']:
        print("Running task:", args.task.lower())
        merge_logs(args.path, 'logs-merged.log')
    elif args.task.lower() in ['outputs', 'output']:
        print("Running task:", args.task.lower())
        merge_outputs(args.path, chunksize)
    else:
        raise ValueError("Unknown merging task: %s" % repr(args.task))
