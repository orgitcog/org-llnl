"""
Compile. Analyze. Prepare.
"""

import sys
import os

# Add this directory to path in case we call it from other places
sys.path.append(os.path.dirname(__file__))

import argparse
import re
import pandas as pd
import numpy as np
import multiprocessing
import time
import shutil
import traceback
import importlib
import copy
import atomicwrites  # To make sure we have it here
from pprint import pformat
from parsing.container_info import load_compiler_info, load_analyzer_info, get_analyzer_methods, get_container_platform
from compiler_selection.compiler_selector import CompilerSelector
from process_data.data_handler import CAPDataHandler
from process_data.process_cap import cap_folder, cap_single_file
from process_data.clean_source import clean_source
from utils.logs import init_logging, ExceptionLogger, MPLogger
from utils.misc import get_language_family
from utils.auto_detect import auto_detect_cap_task, auto_detect_file_task, auto_detect_folder_task, get_known_tabular_type, \
    get_known_compiled_type, get_known_project_type, get_known_source_type
from bincfg.utils import arg_array_split, hash_obj, progressbar


# The logger that will be used for this multiprocessing thread
LOGGER = None

# Number of seconds to wait on each process before starting to load in the data. This way, not all processes are
#   (hopefully) attempting to load in the data at once, which may take up a lot of memory
PROCESS_INIT_SLEEP_TIME_SECONDS = 15

# The file extensions and their languages
FILE_EXTENSIONS_TO_LANGUAGE = {
    'c': 'c',
    'cpp': 'c++',
    'java': 'java',
    'cs': 'c#',
}
LANGUAGE_FILE_EXTENSIONS = {v: k for k, v in FILE_EXTENSIONS_TO_LANGUAGE.items()}

# File extensions for compiled binaries. Make sure to include the '.' if using!
BINARY_FILE_EXTENSIONS = {
    'elf': '',
    'macho': '',
    'pe': '',
    'java': '.class',
    'c#': '.dll',
}

# Default values for execution info
DEFAULT_EXEC_INFO_VALS = {'postprocessing': [], 'drop_columns': [], 'normalizers': [], 'analyzers': [], 
                          'compile_methods': [], 'container_platform': None, 'fail_on_error': False, 'await_load': False}


def _main_misc(paths, exec_info, n_jobs, task_id, threads, task, progress=False):
    """Compiles, Analyzes, and Prepares miscellaneous data

    Handles single folders/files, as well as a directory containing some number of folders/files

    See the README.md for how to set up the data. Assumes everything is all set up before this, most inputs are
    sanitized, etc.

    Args:
        paths (Dict[str, str]): dictionary of paths to use
        exec_info (Dict[str, Any]): dictionary of execution info
        n_jobs (int): the total number of jobs running
        task_id (int): the id of this task
        threads (int): the number of threads per job
        task (str): the task to complete. Should be a 'source-', 'binary-', 'tabular-', 'project-', 'misc', or 'misc-recursive'
        progress (bool): if True, will show a progressbar
    """
    exec_info = exec_info.copy()

    LOGGER.info("Running CAP-%s with paths: %s\nAnd exec_info: %s" % (task, pformat(paths), pformat(exec_info)))

    # Find all the folders/files to CAP
    found = {'source': [], 'binary': [], 'tabular': [], 'project': [], 'unknown': []}
    
    # Add in known files/folders
    for k in ['source', 'binary', 'tabular', 'project']:
        if task.startswith(k + '-'):
            found[k].append((task.split('-')[1], paths['input']))
            break
    
    # Find miscellaneous folders/files
    else:
        if task not in ['misc', 'misc-recursive']:
            raise ValueError("_main_misc got an unknown task: %s" % repr(task))
        recursive = task == 'misc-recursive'

        if not os.path.exists(paths['input']) or not os.path.isdir(paths['input']):
            raise ValueError("CAP-misc input_path does not exist, or is not a directory: %s" % repr(paths['input']))
        
        LOGGER.info("Recursively searching files/folders in path: %s" % repr(paths['input']))
        _rec_search_misc_folder(found, paths['input'], recursive, progress=progress)
    
    if len(found['unknown']) > 0:
        LOGGER.info("Found %d unknown files/folders, these will be ignored:\n%s" % (len(found['unknown']), pformat(found['unknown'])))
    del found['unknown']

    #LOGGER.debug("CAP-ing the following files/folders:\n%s" % pformat(found))

    # Load in the compile/analyzer methods, make sure they are [None] if the lists are empty
    compile_methods, analyzer_methods = _load_cm_anal_methods(paths, exec_info)
    compile_methods = [None] if len(compile_methods) == 0 else compile_methods
    analyzer_methods = [None] if len(analyzer_methods) == 0 else analyzer_methods

    # Figure out how we are splitting all of the different CAP's and compile methods
    caps = [(k, v, cm, a) for k, l in found.items() for cm in compile_methods for a in analyzer_methods for v in l]

    # Use a smaller number of files if in debug mode
    if DEBUG_NUM_FILES is not None:
        LOGGER.info("RUNNING IN DEBUG MODE, ONLY USING %d FOLDERS/FILES!" % DEBUG_NUM_FILES)
        caps = caps[:DEBUG_NUM_FILES]

    # Remove the tabular datasets, and add on the elements of tabular datasets one at a time. Done this way so we can
    #   do the _DEBUG_NUM_FILES based on the number of files/folders, not individual elements within them
    caps = [t for t in caps if t[0] not in ['tabular']]
    caps += [(k, v, cm, a, t, e) for k in ['tabular'] for l in found[k] for cm in compile_methods for a in analyzer_methods 
             for v in l for t, df in _load_tabular(v, columns=['id'], tab_type=v[0]) for e in df['id']]
    
    # Now that we have every available item to CAP, we can sort and split based on tasks. We'll just assume they all take
    #   roughly the same amount of time for now. Also print some debug info
    LOGGER.info("Found %d total items to CAP: %s" % (len(caps), {k: "%d files" % len(v) for k, v in found.items()}))
    caps = np.array_split(np.array(sorted(caps), dtype=object), n_jobs)[task_id]
    LOGGER.info("This is task number %d/%d and will be CAP-ing %d total items" % (task_id + 1, n_jobs, len(caps)))

    # Create the data handler
    LOGGER.info("Ininitializing DataHandler...")
    data_handler = CAPDataHandler(paths, exec_info)

    # Tell other threads that we are done loading data
    _data_loaded()

    # Keep track of the number of items we CAP
    curr_item_idx, num_failed = 0, 0

    # Start with all of the tabular datas
    this_tabs = {}
    for t in caps:
        if t[0] in ['tabular']:
            this_tabs.setdefault(t[0], (t[1], t[4], []))[1].append(t[5])
    
    # If we found some tabular data, run that
    if len(this_tabs) > 0:
        new_curr_items, new_num_failed = _tabular_misc_cap(this_tabs, caps, data_handler, paths, progress)
        curr_item_idx, num_failed = curr_item_idx + new_curr_items, num_failed + new_num_failed
    
    # Now, run through all capping that is not tabular
    for _, (task, fp), cm, am in progressbar([t for t in caps if t[0] not in ['tabular']], progress=progress):
        
        failed = True
        with ExceptionLogger(LOGGER, handle=True, message="Error with file at path: %s, task: %s" % (fp, repr(task))):
            LOGGER.debug("Beginning CAP for file path: %s" % repr(fp))
            curr_item_idx += 1

            # The id of this can just be the hash of its filepath for now, make it fit in signed int64
            f_id = hash_obj(fp, return_int=True) % (2 ** 63 - 1)

            if t[0].startswith('source'):
                language_family = get_language_family(task)

                # Read in, clean, and write out the source
                with open(fp, 'r') as f:
                    source = f.read()
                source = clean_source(source, language_family)
                source_filepath = os.path.join(paths['temp'], '%d.%s' % (task_id, LANGUAGE_FILE_EXTENSIONS[language_family]))
                with open(source_filepath, 'w') as f:
                    f.write(source_filepath)

                # Get the compile method being used
                rng = np.random.default_rng(seed=hash_obj([task, cm, am, exec_info['exec_uid']], return_int=True))
                cm = cm(language_family, rng)

                # The metadata will only include a few things
                metadata = {'id': f_id, 'language': task, 'language_family': language_family, 'source': source, 'filepath': fp}

                # CAP the data and handle it
                data_dict = cap_single_file(paths, source_filepath, f_id, [cm], [am], metadata=metadata,
                                            container_platform=exec_info['container_platform'], fail_on_error=exec_info['fail_on_error'])
                data_handler.add_data(data_dict, empty_temp=True)

            elif t[0].startswith('binary'):
                # Move the binary to its temp position
                binary_filepath = os.path.join(paths['temp'], '%d%s' % (task_id, BINARY_FILE_EXTENSIONS[task]))
                shutil.copy(fp, binary_filepath)

                # The metadata will only include a few things. The binary should be automatically kept
                metadata = {'id': f_id, 'filepath': fp, 'binary_type': task}

                # CAP the data and handle it
                data_dict = cap_single_file(paths, binary_filepath, f_id, [cm], [am], metadata=metadata,
                                            container_platform=exec_info['container_platform'], fail_on_error=exec_info['fail_on_error'])
                data_handler.add_data(data_dict, empty_temp=True)

            elif t[0].startswith('project'):
                # Copy the folder to its temp position
                project_dir = os.path.join(paths['temp'], '%d-project' % f_id)
                if os.path.exists(project_dir):
                    shutil.rmtree(project_dir)
                shutil.copytree(fp, project_dir, symlinks=True)
                
                # The metadata will only include a few things
                metadata = {'id': f_id, 'folder_path': fp, 'project_type': task}

                # CAP the data and handle it
                cap_folder(paths, project_dir, f_id, task, [cm], [am], data_handler, LOGGER, exec_info, metadata=metadata)

                # Remove the temp copy of this project
                shutil.rmtree(project_dir)
            
            else:
                raise NotImplementedError("Unknown task type: %s" % repr(task))
            
            failed = False
        
        if failed:
            num_failed += 1
    
    # Compilation finished, save data
    data_handler.save()

    LOGGER.info("Failed to CAP %d items" % num_failed)


def _tabular_misc_cap(this_tabs, caps, data_handler, paths, progress):
    """Performs the tabular CAP-ing for the main_misc task
    
    Returns:
        Tuple[Int, Int]: tuple of (num_items, num_failed)
    """
    LOGGER.info("Beginning CAP for tabular files...")
    curr_item_idx, num_failed = 0, 0

    pb = progressbar(sum(len(ids) for ((_, fp), t, ids) in this_tabs.values()))

    for tab_type, ((_, fp), t, ids) in this_tabs.items():
        LOGGER.info("Loading in data for tabular file: %s, cap_type: %s, %d ids to CAP" % (fp, t, len(ids)))

        # Load in all of the datapoints we will be CAP-ing within this tabular file
        with ExceptionLogger(LOGGER, handle=True, message="Error with tabular file: %s" % fp):
            failed_file = True
            curr_tab = _load_tabular(fp, tab_type=tab_type)
            curr_tab = curr_tab[curr_tab.id.isin(ids)]

            LOGGER.info("Found %d/%d ids to CAP within tabular file" % (len(curr_tab), len(ids)))

            # CAP each individual datapoint/compile/analysis tuple that we have
            # Make a mapping for quick lookup
            tid_lookup = {tid: (cm, am, t, tid) for _, _, cm, am, t, tid in caps if t[1] == fp}
            for t_idx in range(len(curr_tab)):

                # No matter what, increment the item counter
                curr_item_idx += 1
                next(pb)

                with ExceptionLogger(LOGGER, handle=True, message="Error with t_tid: %s" % curr_tab.iloc[t_idx]['id']):
                    failed_tid = True
                
                    LOGGER.debug("Starting CAP item number %d/%d for this task_id (%d/%d for this tabular file)" 
                                % (curr_item_idx, len(caps), t_idx, len(curr_tab)))

                    # Get the item to CAP and info about it
                    t_item = curr_tab.iloc[t_idx]
                    t_cm, t_am, t_t, t_tid = tid_lookup[t_item['id']]
                    
                    # Figure out whether we are compiling/analyzing or just analyzing
                    if t_t == 'compile':
                        language = t_item['programming_language']
                        language_family = get_language_family(language)
                        data_to_write, write_type = clean_source(t_item['source'], language_family), 'w'
                        data_filepath = os.path.join(paths['temp'], '%d.%s' % (task_id, LANGUAGE_FILE_EXTENSIONS[language_family]))

                        # Get the compile method being used
                        rng = np.random.default_rng(seed=hash_obj([language, t_tid, t_cm, t_am, exec_info['exec_uid']], return_int=True))
                        t_cm = t_cm(language_family, rng)

                        if not isinstance(data_to_write, str):
                            raise TypeError("Data to write for cap_type 'compile' should be of type 'str', not %s" 
                                            % repr(type(data_to_write).__name__))
                        LOGGER.debug("Compiling and Analyzing - ID: %s, Language: %s, Language family: %s" 
                                    % (t_tid, language, language_family))
                        
                    elif t_t == 'analyze':
                        binary_type = get_known_compiled_type(t_item['binary'])
                        data_to_write, write_type = t_item['binary'], 'wb'
                        data_filepath = os.path.join(paths['temp'], '%d%s' % (task_id, BINARY_FILE_EXTENSIONS[binary_type]))

                        if not isinstance(data_to_write, bytes):
                            raise TypeError("Data to write for cap_type 'analyze' should be of type 'bytes', not %s" 
                                            % repr(type(data_to_write).__name__))
                        LOGGER.debug("Analyzing only - ID: %s, binary type: %s" % (t_tid, binary_type))
                        
                    else:
                        raise NotImplementedError("Tabular cap_type: %s" % repr(t_t))

                    # Write the data to file. NOTE: things like Java files will be renamed automatically when compiling
                    with open(data_filepath, write_type) as f:
                        f.write(data_to_write)
                    
                    # Get the metadata for this source code
                    metadata = t_item.to_dict()

                    # Run the single-file compilation process
                    data_dict = cap_single_file(paths, data_filepath, t_tid, [t_cm], [t_am], metadata=metadata,
                                                container_platform=exec_info['container_platform'], fail_on_error=exec_info['fail_on_error'])
                    data_handler.add_data(data_dict, empty_temp=True)

                    failed_tid = False
                
                # If we failed on this t_tid, then increment the failed counter
                if failed_tid:
                    num_failed += 1
                
            failed_file = False
        
        # If we failed loading the file, increment the counters
        if failed_file:
            num_failed += len(ids)
            curr_item_idx += len(ids)
            [next(pb) for _ in range(len(ids))]
    
    # Finish the progressbar
    [x for x in pb]
    
    return curr_item_idx, num_failed


def _load_tabular(tabular_path, columns=None, tab_type=None):
    """Loads in tabular data from the given path
    
    Error-tolerant. Will return an empty dataframe if something failed, printing error to logger

    Returns:
        Tuple[str, pd.DataFrame]: tuple of (cap_type: str, data: pd.DataFrame) where cap_type is either 'compile' for
            data that should be compiled/analyzed for 'analyze' for data that should only be analyzed. If an error
            occurs, then the tuple ('None', pd.DataFrame({'id': []})) will be returned
    """
    try:
        if tab_type in ['tabular', None]:
            tab_type = get_known_tabular_type(tabular_path)
            if tab_type is None:
                raise ValueError("Given `tabular_path` was not a known tabular file type")

        tab_type = tab_type.split('-')[1]
        if tab_type == 'parquet':
            ret = pd.read_parquet(tabular_path, columns=columns)
        elif tab_type == 'csv':
            ret = pd.read_csv(tabular_path, columns=columns)
        else:
            raise NotImplementedError("Unknown tabular file type: %s" % repr(tab_type))
        
        if all(k in ret for k in ['source', 'programming_language']):
            if 'binary' in ret.columns:
                del ret['binary']
            return 'compile', ret
        elif 'binary' in ret:
            return 'analyze', ret
        else:
            raise ValueError("Could not find necessary columns in tabular data, either %s or %s columns required"
                             % (['source', 'programming_language'], ['binary']))
        
    except Exception as e:
        LOGGER.error("Failed to load tabular data from path: %s\nFor reason: %s" % (tabular_path, str(e)))
        return 'None', pd.DataFrame({'id': []})


def _rec_search_misc_folder(found, input_path, recursive, progress=True):
    """Recursively (or non-recursively) checks files/folders in the given directory and stores them into passed found dict"""
    rec_files = [os.path.join(input_path, f) for f in os.listdir(input_path)]
    if DEBUG_NUM_FILES is not None:
        rec_files = rec_files[:DEBUG_NUM_FILES]
    
    for f in progressbar(rec_files, progress=progress):
        task = auto_detect_cap_task(f)
        if task is None:
            found['unknown'].append(f)
        elif task in ['misc', 'misc-recursive']:
            if recursive:
                _rec_search_misc_folder(found, f, recursive, progress=False)
            else:
                found['unknown'].append(f)
        else:
            found[task.split('-')[0]].append((task.split('-')[1], f))
    

def _main_partitioned(paths, exec_info, n_jobs, task_id, threads, progress=False):
    """Compiles, Analyzes, and Prepares partitioned data

    See the README.md for how to set up the data. Assumes everything is all set up before this, most inputs are
    sanitized, etc.

    Args:
        paths (Dict[str, str]): dictionary of paths to use
        exec_info (Dict[str, Any]): dictionary of execution info
        n_jobs (int): the total number of jobs running
        task_id (int): the id of this task
        threads (int): the number of threads per job
        progress (bool): whether or not to show a progressbar
    """
    exec_info = exec_info.copy()

    LOGGER.info("Running CAP-partitioned with paths: %s\nAnd exec_info: %s" % (pformat(paths), pformat(exec_info)))
    for k in ['partitioned_info', 'partitioned']:
        if k not in paths or paths[k] is None:
            raise ValueError("Did not pass a %s path when CAP-ing competition-like data" % repr(k))

    # Load in the partitioned_info
    part_info = pd.read_parquet(paths['partitioned_info'])
    
    LOGGER.info("Loaded partitioned info with a max of %d possible source codes to compile" % len(part_info))

    # Use a smaller number of files if in debug mode
    if DEBUG_NUM_FILES is not None:
        LOGGER.info("RUNNING IN DEBUG MODE, ONLY USING %d FILES!" % DEBUG_NUM_FILES)
        part_info = part_info.iloc[:DEBUG_NUM_FILES]

    # Get all of the ids's that we will be processing in this job
    if n_jobs > 1:
        part_info = part_info.iloc[slice(*arg_array_split(len(part_info), n_jobs, return_index=task_id))]
    
    # Set the part_info index to the id that way we can more easily index values in it.
    # Also add in a 'compiled' boolean column so we can print out which ones weren't compiled right at the end
    part_info.set_index('id', inplace=True)
    part_info['compiled'] = False

    unique_langs = part_info['programming_language'].unique()
    LOGGER.info("Found %d 'unique' languages: %s" % (len(unique_langs), list(unique_langs)))
    LOGGER.info("This is task number %d/%d and will be compiling %d total source codes" % (task_id + 1, n_jobs, len(part_info)))

    # Load in the compile/analyzer methods
    compile_methods, analyzer_methods = _load_cm_anal_methods(paths, exec_info)

    # Check for other, necessary directories
    partitioned_dir = paths['partitioned']

    # Prepare the data handler
    LOGGER.info("Ininitializing DataHandler...")
    data_handler = CAPDataHandler(paths, exec_info)

    # Now that all the data has been loaded, we can make the file to tell other threads to load
    _data_loaded()

    # Iterate through all the partition files, loading them in if necessary and getting the source codes
    curr_part_idx = 0
    for f in [os.path.join(partitioned_dir, f) for f in os.listdir(partitioned_dir) if re.fullmatch(r'[0-9]+-[0-9]+.(?:parquet|pq)', f) is not None]:
        
        # Figure out if there are any source codes in the range of this file, returning if not
        start_id, end_id = map(int, os.path.basename(f).split('.')[0].split('-'))
        within_range = part_info[(part_info.index >= start_id) & (part_info.index < end_id)]

        if len(within_range) == 0:
            continue

        LOGGER.info("Found %d source codes to compile within the range [%d, %d)" % (len(within_range), start_id, end_id))

        with ExceptionLogger(LOGGER, handle=True, message="Error with chunk %s:" % f):

            # Get all the source codes in this chunk that this job will compile
            chunk = pd.read_parquet(f, columns=['id', 'source']).set_index('id')
            chunk = chunk[chunk.index.isin(within_range.index)]

            LOGGER.info("Starting file: '%s'. Contains %d source codes that this task will compile." % (f, len(chunk)))

            # Compile each source code
            for i, part_id in enumerate(chunk.index):

                added = False
                source_filepath = None
                with ExceptionLogger(LOGGER, handle=True, message="Error building binary for partition id %d:" % part_id):

                    # Get the language, and clean the source code
                    language = within_range.loc[part_id]['programming_language']
                    language_family = get_language_family(language)
                    source_code = clean_source(chunk.loc[part_id]['source'], language_family)

                    curr_part_idx += 1

                    LOGGER.debug("Starting source code number %d / %d for this chunk (%d / %d total for this task). ID: %d, Language: %s, Language family: %s"
                        % (i + 1, len(chunk), curr_part_idx, len(part_info), part_id, language, language_family))

                    # Write the source code to file. NOTE: things like Java files will be renamed automatically when compiling
                    source_filepath = os.path.join(paths['temp'], '%d.%s' % (task_id, LANGUAGE_FILE_EXTENSIONS[language_family]))
                    with open(source_filepath, 'w') as f:
                        f.write(source_code)
                    
                    # Get the metadata for this source code
                    metadata = part_info.loc[part_id][[c for c in part_info.columns if c not in ['compiled']]].to_dict()

                    # Select the compile methods for this submission
                    rng = np.random.default_rng(seed=10000 + abs(hash(part_id)) + abs(hash(exec_info['execution_uid'])))
                    cms = [cm(language_family, rng) for cm in compile_methods]
                    
                    # Run the single-file compilation process
                    data_dict = cap_single_file(paths, source_filepath, part_id, cms, analyzer_methods, metadata=metadata,
                                                container_platform=exec_info['container_platform'], fail_on_error=exec_info['fail_on_error'])
                    added = True

                    data_handler.add_data(data_dict, empty_temp=True)

                    if all(e is None for e in data_dict['error']):
                        part_info.loc[part_id, 'compiled'] = True
                
                # Delete the raw data file if it exists
                if source_filepath is not None and os.path.exists(source_filepath):
                    os.remove(source_filepath)
                
                # Add count of failed to data for correct timing. Assume it failed on the last compile_method
                if not added:
                    data_handler.num_failed += len(compile_methods)

    # Compilation finished, save data
    data_handler.save()

    # Make a note about uncompiled datas
    uncompiled = part_info[~part_info.compiled]
    if len(uncompiled) > 0:
        LOGGER.info("Compilation failed for %d ids's: %s" % (len(uncompiled), list(uncompiled.index)))
    else:
        LOGGER.info("Complation fully successful!")


def _load_cm_anal_methods(paths, exec_info):
    """Loads in and returns (ompile_methods, analyzer_methods), and prints logging info"""
    # Load in the YAML infos
    compiler_info, analyzer_info = load_compiler_info(paths['container_info']), load_analyzer_info(paths['container_info'])

    # Load in the compile methods and analyzer methods
    compile_methods = [CompilerSelector(cm, compiler_info) for cm in exec_info['compile_methods']]
    analyzer_methods = get_analyzer_methods(exec_info['analyzers'], analyzer_info)

    LOGGER.info("Loaded compiler info. Using %d compile methods: \n%s" % (len(compile_methods), \
        '\n\n'.join([pformat(cm) for cm in exec_info['compile_methods']])))
    LOGGER.info("Using %d compiler info generators: \n\t- %s" % (len(compile_methods), '\n\t- '.join([str(cm) for cm in compile_methods])))
    LOGGER.info("Using %d analyzer methods: \n%s" % (len(analyzer_methods), '\n\n'.join([pformat(am) for am in analyzer_methods])))
    return compile_methods, analyzer_methods


def init_cap_logging(exec_uid, log_path, task_id, max_task_id, task, file_name=None, with_stdout=False):
    """This exists so functions in log_parsing can set up the logger as well. Returns the logger"""
    global LOGGER
    if LOGGER is None:
        LOGGER = MPLogger(task_id)
        init_logging(log_path, task_id, exec_uid=str(exec_uid) + '-' + task, file_name=file_name, with_stdout=with_stdout, max_task_id=max_task_id)
        import bincfg
        bincfg.utils.misc_utils.set_logger(LOGGER)
    return LOGGER


def _mp_call(paths, exec_info, n_jobs, task_id, task, threads, progress):
    """Multiprocessing call helper function"""
    global _AWAIT_LOAD_FILE_PATH
    init_cap_logging(exec_info['execution_uid'], paths['logs'], task_id, n_jobs, task)

    new_tempdir = None

    with ExceptionLogger(LOGGER):
        task = task.lower()

        # Attempt to detect which task we are doing if needed
        searches = {
            'auto': (['auto', 'automatic'], auto_detect_cap_task, paths),
            'file': (['file'], auto_detect_file_task, paths['input']),
            'folder': (['folder'], auto_detect_folder_task, paths['input']),
            'project': (['project'], get_known_project_type, paths['input']),
            'binary': (['binary', 'compiled', 'bin'], get_known_compiled_type, paths['input']),
            'tabular': (['tabular', 'parquet', 'pq', 'csv'], get_known_tabular_type, paths['input']),
            'source': (['source', 'code', 'source_code'], get_known_source_type, paths['input']),
        }
        for k, (vs, m, strs) in searches.items():
            if task in vs:
                task = m(strs)
                if task is None:
                    raise ValueError("Could not auto-detect CAP %s task from path(s):\n%s" % (repr(k), pformat(strs)))
        
        # Await previous processes on this machine to start loading data
        node_idx, thread_idx = divmod(task_id, threads)
        if exec_info['await_load']:
            _AWAIT_LOAD_FILE_PATH = os.path.join(paths['temp'], "__await_load_%d-%d.file" % (node_idx, thread_idx))
            LOGGER.info("Awaiting previous threads to load data")
            if thread_idx != 0:
                LOGGER.info("Node idx: %d, thread_idx: %d - Waiting until previous threads have finished loading in data..." % (node_idx, thread_idx))
                prev_path = os.path.join(paths['temp'], "__await_load_%d-%d.file" % (node_idx, thread_idx - 1))
                for i in range(5 * 60):
                    if os.path.exists(prev_path):
                        LOGGER.info("Previous thread loaded after %d seconds of waiting!" % i)
                        os.remove(prev_path)
                        break
                    time.sleep(1)
                else:
                    LOGGER.info("Previous threads never finished load, loading now...")
            else:
                LOGGER.info("This is thread #0, loading data now...")

        # Move the temp directory one down so we can do multiple compiles of the same thing
        new_tempdir = os.path.join(paths['temp'], 'temp-%s-%d' % (exec_info['execution_uid'], task_id))
        for td in [new_tempdir]:
            if os.path.exists(td):
                shutil.rmtree(td)
            os.makedirs(td)
        paths['temp'] = new_tempdir

        exec_info.update({'task': task, 'task_id': task_id, 'n_jobs': n_jobs})
        
        # Pulling out the expected file information
        if task in ['partitioned', 'part', 'partition']:
            _main_partitioned(paths, exec_info, n_jobs, task_id, threads, progress=progress)
        elif task.startswith(('source-', 'binary-', 'tabular-', 'project-')) or task in ['misc', 'misc-recursive']:
            _main_misc(paths, exec_info, n_jobs, task_id, threads, task, progress=progress)
        else:
            raise ValueError("Unknown CAP task: %s" % repr(task))

    # Remove the new_tempdir and temp_homedir
    for rm_dir in [new_tempdir]:
        if rm_dir is not None and os.path.exists(rm_dir):
            shutil.rmtree(rm_dir)


_AWAIT_LOAD_FILE_PATH = None
def _data_loaded():
    """Makes a temp file telling other processes on this machine that this thread has finished loading data"""
    if _AWAIT_LOAD_FILE_PATH is not None:
        with open(_AWAIT_LOAD_FILE_PATH, 'w') as f:
            f.write('')


def cap_main(paths, exec_info, task, n_jobs=1, threads=1, task_id=0, hpc_copy_containers=False, specific_tasks=None, progress=True):
    """The main entrypoint for a CAP process

    See the README.md for info on how to set everything up, arguments, etc.

    Paths that for sure exist before calling various _main's: 'atomic_data', 'containers', 'logs', 'output', 'input',
        'temp', 'container_info'. Ones that are optional: 'partitioned_info', 'partitioned'
    
    Args:
        paths (Dict[str, str]): dictionary of paths to use. Available keys: 'default', 'atomic_data', 'containers',
            'logs', 'output', 'input', 'temp', 'partitioned_info', 'partitioned', 'container_info'

            Some paths may have various substrings which will be replaced. These are:

                - "[default_dir]": replaces with the default directory path
                - "[input_path]": replaces with the input data directory path
                - "[main]" or "[cap]": replaces with the path to the directory containing this file
            
            These paths must be present to use and the default directory path cannot use any of them

            NOTE: not all of these have to be present, just enough so that we can build a path for every needed one
        
        exec_info (Dict[str, Any]): dictionary of execution info. Must contain the keys:

            - 'execution_uid' (str): unique string identifier for this execution

            Can contain the optional keys:

            - 'postprocessing' (Optional[Union[str, List[str]]], default=[]): string or list of strings for the 
              postprocessings to apply to analyzer outputs, or None to not apply any. Available strings:

              * 'cfg': build a CFG() object, one for each of the normalizers in exec_info['normalizers']
              * 'memcfg': build a MemCFG() object, one for each of the normalizers in exec_info['normalizers']
              * 'stats': build a CFG() object and get the graph statistics with cfg.get_compressed_stats(), one for
                each of the normalizers in exec_info['normalizers']
            
              NOTE: these will be stored as pickled bytes() objects

            - 'drop_columns' (Optional[Union[str, List[str]]], default=[]): by default, all of the data generated is kept.
              This if not None, can be a string or list of strings of the column or group of columns to drop. You may also
              pass any columns that would appear in the metadata, and those will be dropped. Metadata columns to 
              drop can be passed either as their original name, or with the prefix 'meta_' as they would appear in
              the output data. Any columns that do not correspond to data being kept will raise an error, unless they 
              start with the prefix 'meta_', in which case it is assume that that column is a possible metadata column 
              which may or may not exist. Available non-metadata columns to drop: 
                
              'analyzer', 'binaries', 'analyzer_output', 'metadata', 'error', 'compile_stdout', 'compile_stderr', 
              'analyzer_stdout', 'analyzer_stderr', 'compile_time', 'analysis_time', 'language_family', 'compiler_family',
              'compiler', 'compiler_version', 'architecture', 'flags'

              There are also a couple of special strings that will drop groups of columns including:

              * 'metadata': drop any metadata that was passed in metadata dictionaries
              * 'compile_info': drop all of the compilation info
              * 'timings': drop all of the timing information ('compile_time', 'analysis_time')
              * 'stdio': drop all of the stdio information ('compile_stderr', 'analyzer_stdout', etc.)
              * 'stdout': drop all of the stdout information ('compile_stdout', 'analyzer_stdout')
              * 'stderr': drop all of the stderr information ('compile_stderr', 'analyzer_stderr')

              See the README.md for what all of these columns are.

            - 'normalizers' (Optional[Union[str, Normalizer, Iterable[Union[None, str, Normalizer]]]], default=[]): normalizers 
              to use when building postprocessing CFG's. Will build those CFG's once for each of the normalizers here. Can be:

              * None: will not normalize at all (use raw input) for CFG's, will use 'unnormalized' normalization
                (default BaseNormalizer()) for MemCFG's
              * str: string name of a normalizer to use
              * Normalizer: Normalizer-like object to use
              * Iterable of any of the above: will do one of each normalization for each datapoint
            
            - 'analyzers' (Optional[Union[str, List[str]]], default=[]): string or list of strings for the analyzers to
               use. Can be empty if you wish to not analyze files, but only compile
            - 'compile_methods' (Optional[List[Dict[str, Any]]], default=[]): list of compile methods to use. See the 
              readme for this
            - 'container_platform' (Optional[str]): the container platform to use, or None to detect one by default
            - 'fail_on_error' (Optional[bool]): By default, most errors will be captured and saved silently into the
              output data during the CAP process. If this is True, then any error while CAP-ing a file/folder will
              instead be raised, an error will be printed to the log files, and that data will not be stored in the
              output files. This will not stop the entire CAP process, however, as files and folders will continue
              to be CAP-ed. This just makes the errors visible in the logs and doesn't save them along with the output data
            - 'await_load' (Optional[bool]): If True, then each thread within an execution will wait to begin loading 
              its data until the previous thread has completed the data loading process to save memory during the intial 
              loading/splitting phase

        task (str): the task being run. Can be:

            - "auto": automatically determine what type of task is being run based on the input directories
            - "partitioned": CAP-ing partitioned data
            - "source": CAP-ing a single source file. Language will be automatically detected
            - "source-[language]": CAP-ing a single source file, with the '[language]' being the language family used
            - "project": CAP-ing a project (a directory of files that produce one or more binaries as a part of the
              same project). Project built type will be automatically detected
            - "project-[build_type]": CAP-ing a project (a directory of files that produce one or more binaries as a 
              part of the same project). '[build_type]' determines the project build type
            - "binary": CAP-ing a single precompiled binary. Binary type will be automatically detected
            - "binary-[binary_type]": CAP-ing a single precompiled binary. '[binary_type]' determines the binary type
            - "tabular": CAP-ing a single file containing tabular data (EG: csv, parquet, etc.). File type will be
              automatically detected
            - "tabular-[file_type]": CAP-ing a single file containing tabular data (EG: csv, parquet, etc.). '[file_type]'
              determines the type of tabular data in the file
            - "file": CAP-ing a single file. Type of file will be auto-detected
            - "folder": CAP-ing a folder. Type of folder will be auto-detected
            - "misc": CAP-ing a bunch of files/folders within a directory. Files and folders types will be automatically
              detected and CAP-ed
            - "misc-recursive": same as "misc", but will recursively check subfolders for other files/projects to CAP
        
        n_jobs (int): the total number of jobs being run
        threads (int): the number of threads to use per task
        task_id (int): the id of this current task
        hpc_copy_containers (bool): if True, will assume we are running on HPC and copy all of the containers over
            to "[temp_dir]/containers" for faster loading
        specific_tasks (Optional[List[int]]): if passed, then only these specific tasks will be run
        progress (bool): whether or not to show a progressbar. Forced to False when using multiprocessing. Defaults to
            True otherwise.
    """
    paths = paths.copy()
    exec_info = exec_info.copy()

    # Make sure exec_info contains all the needed values
    for k in ['execution_uid']:
        if k not in exec_info:
            raise ValueError("Did not pass needed key %s in exec_info" % repr(k))
    if not isinstance(exec_info['execution_uid'], str):
        raise TypeError("The execution_uid must be str, not %s" % repr(type(exec_info['execution_uid']).__name__))
    
    # Replace optional exec_info values with their defaults
    for k, v in DEFAULT_EXEC_INFO_VALS.items():
        if k not in exec_info or exec_info[k] is None:
            exec_info[k] = copy.deepcopy(v)  # Make a copy of the value just in case cap_main() is called multiple times
    
    # Auto-detect the container platform if it doesn't exist
    if exec_info['container_platform'] is None:
        exec_info['container_platform'] = get_container_platform(None)
    
    default_path = paths['default'] if 'default' in paths else None
    for k in ['atomic_data', 'containers', 'logs', 'output', 'input', 'temp']:  # Directories defaulting to within default_dir
        if k not in paths or paths[k] is None:
            if default_path is None: raise ValueError("Did not pass needed path %s" % repr(k))
            paths[k] = os.path.join(default_path, k)
    
    # Other special location directories
    if 'container_info' not in paths or paths['container_info'] is None:
        paths['container_info'] = '[main]/container_info.yaml'
    if 'partitioned' not in paths or paths['partitioned'] is None:
        paths['partitioned'] = '[input_path]/partitioned'
    if 'partitioned_info' not in paths or paths['partitioned_info'] is None:
        paths['partitioned_info'] = '[input_path]/[exec_uid].parquet'
    
    # Replace various strings
    if "[input_path]" in paths['input']:
        raise ValueError("Cannot use '[input_path]' replacement inside of input dir string")
    for k in [k for k in paths if k not in ['default']]:
        # This is bad code lol
        paths[k] = paths[k].replace("[input_path]", paths['input']).replace("[input_path]", paths['input'])\
            .replace("[default_dir]", paths['default']).replace("[main]", os.path.dirname(__file__))\
            .replace('[exec_uid]', exec_info['execution_uid']).replace("[cap]", os.path.dirname(__file__))\
            .replace("[CAP]", os.path.dirname(__file__))

    # Make sure the necessary dirs/paths exist
    for k in ['containers']:
        if not os.path.exists(paths[k]) or not os.path.isdir(paths[k]):
            raise ValueError("Path %s does not exist, or is not a directory: %s" % (repr(k), repr(paths[k])))
    for k in ['container_info']:
        if not os.path.exists(paths[k]) or not os.path.isfile(paths[k]):
            raise ValueError("Path %s does not exist or is not a file: %s" % (repr(k), repr(paths[k])))
    for k in ['input']:
        if not os.path.exists(paths[k]):
            raise ValueError("Path %s does not exist: %s" % (repr(k), repr(paths[k])))
    
    # If the temp dir already exists, delete any of the __await_load files
    if os.path.exists(paths['temp']):
        for f in [os.path.join(paths['temp'], f) for f in os.listdir(paths['temp']) if f.startswith('__await_load')]:
            os.remove(f)
    
    # Make the output, logs, temp, and atomic_data dirs if they don't already exist
    for k in ['atomic_data', 'logs', 'output', 'temp']:
        if not os.path.exists(paths[k]):
            os.makedirs(paths[k])
        elif not os.path.isdir(paths[k]):
            raise ValueError("Path %s exists but is not a directory" % repr(paths[k]))

    # Copy files over if using HPC
    if hpc_copy_containers:
        new_containers_dir = os.path.join(paths['temp'], 'containers')
        shutil.copytree(paths['containers'], new_containers_dir)
        paths['containers'] = new_containers_dir
    
    # Check for a 'specific_tasks'
    if specific_tasks is not None:
        if n_jobs is None or n_jobs <= 0:
            raise ValueError("When using `specific_tasks`, `n_jobs` must be passed and be an integer > 0. Got: %s" % n_jobs)
        if task_id is not None:
            raise ValueError("When using `specific_tasks`, `task_id` must not be passed. Got: %s" % task_id)

        if len(specific_tasks) <= 1:
            _mp_call(paths, exec_info, n_jobs * threads, specific_tasks[0], task, threads, progress)
        
        else:
            with multiprocessing.Pool(len(specific_tasks)) as pool:
                pool.starmap(_mp_call, [[paths, exec_info, n_jobs * threads, t, task, threads, False] for t in specific_tasks])
    
    # Otherwise, do this the normal way
    else:
        if threads <= 1:
            _mp_call(paths, exec_info, n_jobs, task_id, task, threads, progress)
        
        else:
            with multiprocessing.Pool(threads) as pool:
                pool.starmap(_mp_call, [[paths, exec_info, n_jobs * threads, task_id * threads + i, task, threads, False] for i in range(threads)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compile. Analyze. Prepare.')
    parser.add_argument('--atomic_data_dir', type=str, default=None, action='store', help='The path to a directory for'
        ' atomic data')
    parser.add_argument('--containers_dir', type=str, default=None, action='store', help='The path to a directory for'
        ' containers')
    parser.add_argument('--logs_dir', type=str, default=None, action='store', help='The path to a directory for'
        ' log files')
    parser.add_argument('--output_dir', type=str, default=None, action='store', help='The path to a directory for'
        ' output data')
    parser.add_argument('--input_path', type=str, default=None, action='store', help='The path to a directory/file for'
        ' the input data')
    parser.add_argument('--partitioned_info_path', type=str, default=None, action='store', help='The path to a parquet file'
        ' containing the metadata and id keys for a "partitioned" CAP process')
    parser.add_argument('--partitioned_dir', type=str, default=None, action='store', help='The path to a directory'
        ' containing partitioned data for a "partitioned" CAP process')
    parser.add_argument('--container_info_path', type=str, default=None, action='store', help='The path to a YAML file'
        ' containing the container_info. Defaults to "container_info.yaml" file assumed to be right next to this file.')
    parser.add_argument('--temp_dir', type=str, default=None, action='store', help='The path to a directory for'
        ' temporary files')
    parser.add_argument('--execution_info_path', type=str, default=None, action='store', help='The path to a python file'
        ' that should be imported to get the execution information')
    parser.add_argument('--default_dir', type=str, default=None, action='store', help='The path to a directory for'
        ' any unpassed default directories. Any missing directories will use default names and be subdirectories of this one')
    parser.add_argument('-n', '--n_jobs', type=int, default=None, action='store', help='The number of jobs running. If '
        'this argument is not passed, will first check the os enironment variable SLURM_ARRAY_TASK_COUNT. If that '
        'does\'t exist, then will assume n_jobs=1')
    parser.add_argument('-t', '--task_id', type=int, action='store', default=None,
                        help='The task_id for this process. Should be in the range [0, num_jobs - 1] and unique for each process.'
                        ' If this argument is not passed, then the SLURM_ARRAY_TASK_ID environment variable will be used.')
    parser.add_argument('--threads', type=int, default=1, action='store', help='Number of threads to use for this task.')
    parser.add_argument('--task', type=str, default='auto', action='store', help='Which task to run. Can be: "auto", '
                        '"partitioned", "source", "source-[language]", "project", "project-[build_type]", "binary", '
                        ' "binary-[binary_type]", "tabular", "tabular-[file_type]", "file", "folder", "misc", or "misc-recursive". See '
                        'README.md for info')
    parser.add_argument('--container_platform', type=str, default=None, action='store', help='The container platform to use')
    parser.add_argument('--fail_on_error', action='store_true', help='By default, most errors will be captured and saved '
        'silently into the output data during the CAP process. If this is True, then any error while CAP-ing a file/folder '
        'will instead be raised, an error will be printed to the log files, and that data will not be stored in the '
        'output files. This will not stop the entire CAP process, however, as files and folders will continue to be CAP-ed. '
        'This just makes the errors visible in the logs and doesn\'t save them along with the output data')
    parser.add_argument('--await_load', action='store_true', help='If passed, then each thread within an execution will '
                        'wait to begin loading its data until the previous thread has completed the data loading process '
                        'to save memory during the intial loading/splitting phase')
    parser.add_argument('--hpc_copy_containers', action='store_true', help='If this flag is passed, then it is assumed '
        'that we are running on HPC systems, and we should copy container files from the given `containers_dir` into a '
        'temporary place on this node\'s in-memory filesystem for faster loading of containers. The \'containers\' path '
        'will be automatically updated to be "[temp_path]/containers" with all containers for the original "--containers_dir" '
        'directory being copied into that path')
    parser.add_argument('--specific_tasks', type=str, default=None, action='store', help='Specific task_id\'s you wish '
        'to run. Should be a comma separated list of integer task_id\'s. It is assumed that if there are multiple tasks, '
        'then they should be run in parallel. If this is passed, then you must pass most values directly. EG: `n_jobs` '
        'must be passed, `task_id` must not be passed, `threads` should be the same value that was used during full '
        'execution and will not specify the number of threads to use to run these specific tasks (it is only used for '
        'proper logging, `task` must be passed and cannot be \'all\', and `task_id_offset` must not be passed or must '
        'be set to the default value of 0.')

    args = parser.parse_args()

    if args.n_jobs is None:
        try:
            n_jobs = int(os.environ['SLURM_ARRAY_TASK_COUNT'])
        except KeyError:
            n_jobs = 1
    else:
        n_jobs = args.n_jobs
    
    if n_jobs == 1:
        task_id = 0
    elif args.task_id is not None:
        task_id = args.task_id
    else:
        try:
            task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])
        except KeyError:
            raise ValueError("Could not get task_id. Either pass it as a command line arg, or set it to the environment"
                " variable SLURM_ARRAY_TASK_ID")
    
    # Need to set the n_jobs to None if using specific tasks and args.n_jobs is None, that way an error will be raised
    if args.specific_tasks is not None:
        specific_tasks = [int(s.strip()) for s in args.specific_tasks.split(',') if len(s.strip()) > 0]
        n_jobs = args.n_jobs if args.n_jobs is None else n_jobs
    else:
        specific_tasks = None
    
    _paths = {
        'atomic_data': args.atomic_data_dir,
        'containers': args.containers_dir,
        'logs': args.logs_dir,
        'output': args.output_dir,
        'input': args.input_path,
        'partitioned_info': args.partitioned_info_path,
        'partitioned': args.partitioned_dir,
        'container_info': args.container_info_path,
        'temp': args.temp_dir,
        'default': args.default_dir,
    }
    
    # Find the execution information python file to import
    exec_info_path = os.path.join(os.path.dirname(__file__), 'execution_info.py')
    if args.execution_info_path is not None:
        exec_info_path = args.execution_info_path
    
    try:
        sys.path.append(os.path.dirname(exec_info_path))
        ei_base = os.path.basename(exec_info_path)
        ei = importlib.import_module(ei_base.rpartition('.')[0] if '.' in ei_base else ei_base)

        DEBUG_NUM_FILES = ei.DEBUG_NUM_FILES if hasattr(ei, 'DEBUG_NUM_FILES') else None

        exec_info = {k: getattr(ei, k.upper()) for k in DEFAULT_EXEC_INFO_VALS if hasattr(ei, k.upper())}
        exec_info['execution_uid'] = ei.EXECUTION_UID
    except Exception as e:
        raise ImportError("Could not get execution_info from python file at path %s for reason:\n%s: %s\nTraceback:\n%s"
                          % (repr(exec_info_path), type(e).__name__, e, traceback.format_exc()))
    
    # Add in the exec_info that is passed on the command line
    exec_info['container_platform'] = args.container_platform if args.container_platform is not None \
        else exec_info['container_platform'] if 'container_platform' in exec_info else None
    exec_info['fail_on_error'] = True if args.fail_on_error else exec_info['fail_on_error'] if 'fail_on_error' in exec_info else False
    exec_info['await_load'] = True if args.await_load else exec_info['await_load'] if 'await_load' in exec_info else False

    cap_main(_paths, exec_info, args.task, n_jobs, args.threads, task_id, hpc_copy_containers=args.hpc_copy_containers, 
             specific_tasks=specific_tasks)
