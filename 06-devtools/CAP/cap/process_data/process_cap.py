"""
Methods to compile and analyze source code
"""

import numpy as np
import stat
import os
import subprocess
import json
import shutil
import traceback
from timeit import default_timer
from pprint import pformat
from utils.logs import MPLogger, ExceptionLogger
from utils.misc import CriticalError, get_language_family
from utils.auto_detect import _KNOWN_PROJECT_FILES, auto_detect_file_task
from parsing.container_info import get_container_platform
from process_data.compile import compile_single_file
from process_data.analyze import run_analysis_cmd
from process_data.clean_source import clean_source
from bincfg import hash_obj


# Include files and corresponding flags for dlls
DLL_IMPORT_FLAGS = {
    'math.h': '-lm'
}

# Files that will be included on the command line because some submissions don't have them
INCLUDE_FILES = {
    'c++': ['climits', 'cstring', 'math.h', 'ctype.h', 'cstdio', 'vector', 'string', 'numeric', 'tuple'],
    'c': [],
    'java': [],
}

# Script to build cmake folders
FOLDER_BUILD_SCRIPT_CMAKE = """#!/bin/bash

compile_flags=$@
disable_encryption=On

rm -rf $CAP_BUILD_DIR

cmake -DDISABLE_ENCRYPTION=$disable_encryption -D CMAKE_CXX_FLAGS="$compile_flags" -B $CAP_BUILD_DIR .
cmake --build $CAP_BUILD_DIR
"""

# The logger to use
LOGGER = MPLogger()

# The maximum number of characters to keep in a stdout/stderr string
MAX_STD_STR_LEN = 10_000

# The different filenames that can exist for CAP project tasks
_CAP_PROJECT_FILENAMES = {k: v for k, v in _KNOWN_PROJECT_FILES}


def cap_single_file(paths, cap_file_path, cap_id, compile_methods, analyzer_methods, metadata=None, container_platform=None, 
                    fail_on_error=False):
    """Compiles, analyzes, and prepares a single file

    This can compile and analyze a single datapoint. It can take either a source file or precompiled binary, and will
    compile/analyze as needed depending on parameters passed

    Args:
        paths (Dict[str, str]): the dictionary of file paths
        cap_file_path (str): the path to the file to compile, analyze, and prepare
        cap_id (Union[int, str]): the id of this cap datapoint
        compile_methods (Optional[List[Dict]]): a list of compiler infos, IE: you called compile_method() one or more 
            times, and these are the results. Otherwise None if no compilation should be performed and the `cap_file_path`
            should be treated as a path to a precompiled binary
        analyzer_methods (Optional[List[Dict]]): a list of analyzer info dictionaries, or None to not perform any analysis
        metadata (Optional[Dict]): dictionary of metadata which will be shallow copied and returned as the metadata
        container_platform (Optional[str]): the container platform to use, or None to auto-detect it
        fail_on_error (bool): if False, then any errors raised will be caught and returned within the 'errors' list, but
            CAP-ing will attempt to continue. Otherwise errors will be immediately raised and the CAP-ing(s) of this
            file will all fail
    
    Returns:
        Dict[str, List[Any]]: dictionary of lists of output data. Contains the keys/values:

            - 'id' (List[Union[int, str]]): `cap_id` parameter, the id of this datapoint
            - 'analyzer' (List[Optional[str]]): the string name of the analyzer used, or None if no analysis was performed or
              an error occurred before or during analysis
            - 'compile_info' (List[Optional[Dict]]): the compile information used to compile, or None if no compilation was performed
              or an error occurred before or during compilation
            - 'binary_file_paths' (List[Optional[List[str]]]): list of all output binaries associated with this cap file, with 
              the 'main' binary being the first in the list. For many compilations/binaries, this will only have one element,
              but for some binaries (IE: Java with multiple classfiles), there may be multiple 'binaries' after a compilation.
              In this case, the 'main' binary will always be first in the list. This will only be None if an error occurred.
            - 'analyzer_output_path' (List[Optional[str]]): the string path to the analyzer output file, or None if no analysis 
              was performed or an error occurred before or during analysis
            - 'metadata' (List[Optional[Dict]]): `metadata` parameter shallow copied, or None if no metadata was passed
            - 'error' (List[Optional[str]]): string error message for any error occurr during CAP. Will be None if no error
              occurred. Specifically, this is an error from within python, not an error from compile/analysis stderr
            - 'compile_stdout' (List[Optional[str]]): string output from stdout during compilation process, or None if no 
              compilation was performed or an error occurred before or during compilation
            - 'compile_stderr' (List[Optional[str]]): string output from stderr during compilation process, or None if no 
              compilation was performed or an error occurred before or during compilation
            - 'analyzer_stdout' (List[Optional[str]]): string output from stdout during analyzer process, or None if no 
              analysis was performed or an error occurred before or during analysis
            - 'analyzer_stderr' (List[Optional[str]]): string output from stderr during analyzer process, or None if no 
              analysis was performed or an error occurred before or during analysis
            - 'compile_time' (List[Optional[float]]): time in seconds required to compile, or None if no compilation was 
              performed or an error occurred before or during compilation
            - 'analysis_time' (List[Optional[float]]): time in seconds required to analyze, or None if no analysis was 
              performed or an error occurred before or during analysis
    """
    container_platform = get_container_platform(container_platform)
    
    analyzer_methods = [analyzer_methods] if isinstance(analyzer_methods, dict) or analyzer_methods is None else analyzer_methods
    compile_methods = [compile_methods] if isinstance(compile_methods, dict) or compile_methods is None else compile_methods

    num_ret = len(analyzer_methods) * len(compile_methods)
    ret = {k: [None] * num_ret for k in ['analyzer', 'binary_file_paths', 'analyzer_output_path', 'error', 'compile_stdout',
                             'compile_stderr', 'analyzer_stdout', 'analyzer_stderr', 'compile_time', 'analysis_time',
                             'compile_info']}
    ret.update({'id': [cap_id] * num_ret, 'metadata': [metadata] * num_ret})

    def _set_ret_cm_info(key, cmi, val):
        start, end = cmi * len(analyzer_methods), (cmi + 1) * len(analyzer_methods)
        ret[key][start:end] = [val] * (end - start)

    cm = am = None

    # Start with compile methods as each analyzer can use the already compiled output, so we don't redo compiles
    for cm_idx, cm in enumerate(compile_methods):
        
        try:
            # Figure out the new filenames we will use to move files so they don't clash
            nbfp_str = os.path.join(paths['temp'], '%%s-%s-%d.compiled' % (cap_id, cm_idx))
            nafp_str = os.path.join(paths['temp'], '%%s-%s-%d-%%d.analyzed' % (cap_id, cm_idx))
            
            # Perform compilation if using
            if cm is not None:
                init_compile_time = default_timer()
                bin_fps, (cso, cse) = compile_single_file(paths, cm, cap_file_path, container_platform)
                _set_ret_cm_info('compile_info', cm_idx, cm)
                _set_ret_cm_info('compile_time', cm_idx, default_timer() - init_compile_time)
                _set_ret_cm_info('compile_stdout', cm_idx, cso)
                _set_ret_cm_info('compile_stderr', cm_idx, cse)
                if isinstance(bin_fps, Exception):
                    raise bin_fps

            # Otherwise if cm is None, assume cap_file_path is a path to a binary
            else:
                bin_fps = [cap_file_path]

            # Copy binary file paths and add them to return list. We want to copy them to a unique filename so they don't
            #   get overwritted by subsequent compilation methods, but the original files are left there in case they
            #   are needed by the analyzers (EG: with java files)
            new_bin_fps = [nbfp_str % os.path.basename(bfp) for bfp in bin_fps]
            for old_binfp, new_binfp in zip(bin_fps, new_bin_fps):
                shutil.copy(old_binfp, new_binfp)

            _set_ret_cm_info('binary_file_paths', cm_idx, new_bin_fps)

        except Exception as e:
            if isinstance(e, CriticalError):
                raise
            err_idx = cm_idx * len(analyzer_methods)
            error = CAPError(e, am, cm, 'file-binary', ret['compile_stdout'][err_idx], ret['compile_stderr'][err_idx])
            if fail_on_error:
                raise error
            _set_ret_cm_info('error', cm_idx, str(error))
            continue

        for am_idx, am in enumerate(analyzer_methods):
            try:
                ret_idx = cm_idx * len(analyzer_methods) + am_idx

                # Run analysis if using. Analyze, move file, and add values to lists
                if am is not None:
                    ret['analyzer'][ret_idx] = am['name']
                    init_analysis_time = default_timer()

                    # We use the original bin_fps here for analysis, just this once
                    analyzed_fp, (aso, ase) = run_analysis_cmd(paths, bin_fps, am, container_platform)
                    ret['analysis_time'][ret_idx] = default_timer() - init_analysis_time
                    ret['analyzer_stdout'][ret_idx] = aso
                    ret['analyzer_stderr'][ret_idx] = ase

                    new_analyzed_fp = nafp_str % (cap_id, am_idx)
                    os.rename(analyzed_fp, new_analyzed_fp)
                    ret['analyzer_output_path'][ret_idx] = new_analyzed_fp
                
            except Exception as e:
                if isinstance(e, CriticalError):
                    raise
                error = CAPError(e, am, cm, 'file-analyze', ret['analyzer_stdout'][ret_idx], ret['analyzer_stderr'][ret_idx])
                if fail_on_error:
                    raise error
                ret['error'][ret_idx] = str(error)
            
    return ret


def cap_folder(paths, folder_path, cap_id, task, compile_methods, analyzer_methods, data_handler, logger, exec_info, metadata=None):
    """Compiles, analyzes, and prepares a project folder

    This can compile multiple different project file types including:

        1. 'cap': folders that contain a 'CAP.json' file describing what to do. This file should be a JSON file containing
           a dictionary with the following keys/values:
          
          * 'only_cap': a string or list of string filenames relative to this directory for the file/files that should be
            CAP-ed in this project. This is useful for times when there needs to be multiple files in the same directory
            as the main CAP file while compiling/analyzing (EG: header files when compiling, shared libraries when analyzing
            with rose, etc.). Files should only be either source code or binary files, not tabular files or project directories.
            Files should be auto-detectable
      
          NOTE: dictionary keys will be searched for in the above order. If there are multiple conflicting keys, the first
          one found in the order is what will be used
        
        NOTE: files will be searched for in the above order. If there are multiple conflicting files, the first one found
        is what will be used
    
    NOTE: this assumes that the given folder_path has already been copied, if you so wish

    Args:
        paths (Dict[str, str]): the dictionary of file paths
        folder_path (str): the path to the file to compile, analyze, and prepare
        cap_id (Union[int, str]): the id of this cap datapoint
        task (str): the type of project being CAP-ed
        compile_methods (Optional[List[Dict]]): a list of compiler infos, IE: you called compile_method() one or more 
            times, and these are the results. Otherwise None if no compilation should be performed and the `cap_file_path`
            should be treated as a path to a precompiled binary
        analyzers (Optional[List[Dict]]): a list of analyzer info dictionaries, or None to not perform any analysis
        data_handler (DataHandler): the DataHandler that should be used to handle data as it is created
        logger (Logger): the logger to use when outputting logging info
        exec_info (Dict[str, Any]): dictionary of execution info
        metadata (Optional[Dict]): dictionary of metadata which will be shallow copied and returned as the metadata
    """
    # Move the temp directory to be the folder path, make a copy so we don't overwrite the original
    paths = paths.copy()
    paths['temp'] = folder_path

    if task == 'cap':
        _cap_folder_CAP(folder_path, cap_id, logger, metadata, exec_info, compile_methods, analyzer_methods, paths, data_handler)

    else:
        raise NotImplementedError("Unknown project task: %s" % repr(task))


def _cap_folder_CAP(folder_path, cap_id, logger, metadata, exec_info, compile_methods, analyzer_methods, paths, data_handler):
    """Does the CAP type for cap folders"""
    # Find and read in the JSON for this task
    existing = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f in _CAP_PROJECT_FILENAMES['cap']]
    if len(existing) == 0:
        raise ValueError("Could not find the CAP.txt file")
    
    with open(existing[0], 'r') as f:
        cap_project_info = json.loads(f.read())
    if not isinstance(cap_project_info, dict):
        raise TypeError("CAP project JSON object must be a dict, not %s" % repr(type(cap_project_info).__name__))
    
    # If we are doing an 'only_cap' task, we only CAP those given files
    if 'only_cap' in cap_project_info:
        if not isinstance(cap_project_info['only_cap'], (list, tuple, str)):
            raise TypeError("The 'only_cap' key value should be a str or iterable of str, not %s" % repr(type(cap_project_info['only_cap'].__name__)))
        oc = sorted([cap_project_info['only_cap']] if isinstance(cap_project_info['only_cap'], str) else list(cap_project_info['only_cap']))

        logger.info("Performing a 'cap' project type CAP-ing with the 'only_cap' key, %d files in 'only_cap' list" % len(oc))

        # We will CAP in-place here, with a different cap_id for each
        for cap_fp in oc:
            with ExceptionLogger(logger, handle=True, message="Error with 'only_cap' file: %s" % repr(cap_fp)):
                cap_fp_id = hash_obj([cap_id, cap_fp], return_int=True) % (2 ** 63 - 1)
                cap_fp = os.path.join(folder_path, cap_fp)

                # Create a copy of the metadata for this file
                _md = metadata.copy()
                _md.update({'filename': os.path.basename(cap_fp)})

                # Determine what type of file this is
                file_type = auto_detect_file_task(cap_fp)
                if file_type is None or not file_type.startswith(('source', 'binary')):
                    raise ValueError("Could not determine file type, or file type was invalid: %s" % repr(file_type))
                
                # If this is a source file type, get the language and clean the source file
                if file_type.startswith("source"):
                    language = file_type.split('-')[1]
                    language_family = get_language_family(language)
                    _md.update({'language': language, 'language_family': language_family})

                    with open(cap_fp, 'r') as f:
                        source = clean_source(f.read(), language_family)
                    with open(cap_fp, 'w') as f:
                        f.write(source)
                    
                    # Make the compile methods we will be using
                    rng = np.random.default_rng(seed=hash_obj([language, exec_info['exec_uid']], return_int=True))
                    cms = [cm(language_family, rng) for cm in compile_methods]
                
                # Otherwise if this is a binary, add in that metadata and set compiler methods to [None]'s
                elif file_type.startswith("binary"):
                    metadata = {'binary_type': file_type.split('-')[1]}
                    cms = [None for _ in compile_methods]
                
                else:
                    raise NotImplementedError("Unknown file type for CAP 'only_cap' task: %s" % repr(file_type))

                data_dict = cap_single_file(paths, cap_fp, cap_fp_id, cms, analyzer_methods, metadata=_md, 
                                            container_platform=exec_info['container_platform'], fail_on_error=exec_info['fail_on_error'])
                data_handler.add_data(data_dict)

    else:
        raise NotImplementedError("CAP.txt JSON dict did not contain a known control key: %s" % cap_project_info)


class CAPError(Exception):
    """Error during the CAP-ing of a file/project"""
    def __init__(self, other_error, am, cm, cap_type, stdout, stderr):
        super().__init__("Failed during CAP-%s process using analyzer:\n%s\nAnd compile method:\n%s\n\n:\n%s: %s\nTraceback: %s\n\nStdout:\n%s\n\nStderr:\n%s"
                         % (repr(cap_type), pformat(am), pformat(cm), type(other_error).__name__, str(other_error), traceback.format_exc(), stdout, stderr))





"""
def cap_folder(paths, folder_path, task_id, compile_methods, analyzers, container_platform=None, seed_start=None):
    "" "
    CAP a folder

    Args:
        paths (Dict[str, str]): the dictionary of file paths. See main.main_codeforces() for more info
        folder_path (str): the path to the folder to compile, analyze, and prepare
        task_id (int): the task_id of this process
        compile_methods (List[CompilerSelector]): the list of compilation methods to use
        analyzers (Union[str, List[str]]): the analyzer(s) to use
        container_platform (Optional[str]): the container platform to use, or None to auto-detect it
        compile_container (Optiona[str]): either string for container to use for compilation, or None to use the default.
            EG: string filepath for singularity, tag name for docker
        seed_start (Optional[int]): optional int to seed this compilation RNG

    Returns:
        Tuple[List[str], List[str], List[Dict[str, str]], List[str], List[str]]: 5-tuple of
            (compile_ids, analyzers, compile_info, binary_file_paths, analyzer_output_paths)
    "" "
    analyzers = _get_analyzers(analyzers)
    container_platform = _get_container_platform(container_platform)

    files = os.listdir(folder_path)
    cap_build_temp = '__cap_build_%d__' % task_id
    build_dir = os.path.join(folder_path, cap_build_temp)
    build_scripts = [f for f in ['build_cap', 'build_cap.sh'] if f in files and os.path.isfile(f)]

    compile_ids, ret_analyzers, compile_info, analyzer_output_paths, binary_file_paths = [], [], [], [], []

    self_made_script = False

    # Generate the build script if needed
    if len(build_scripts) > 0:
        build_script = os.path.join(folder_path, build_scripts[0])
    else:
        build_script = os.path.join(folder_path, 'build_cap_%d.sh' % task_id)

        # This is a cmake folder
        if 'CMakeLists.txt' in files:
            with open(build_script, 'w') as f:
                LOGGER.debug("Writing CMake script to file")
                f.write(FOLDER_BUILD_SCRIPT_CMAKE)
                self_made_script = True
        
        # Can't compile this folder
        else:
            raise FileNotFoundError("Could not find an appropriate file to compile this folder: %s" % folder_path)
    
    # Make the build directory
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)
    
    # Make sure the build script is executable
    os.chmod(build_script, os.stat(build_script).st_mode | stat.S_IEXEC)
    
    # Compile using each of the compile methods, and each of the analyzers
    for a_i, a in enumerate(analyzers):
        for cm_i, cm in enumerate(compile_methods):
            # Get the random state if needed
            rand = None
            if seed_start is not None:
                cm_hash = (int(hash(cm) + seed_start) ** 7 * 81299903 * (cm_i + 17) ** 31) % 4292969783
                rand = np.random.default_rng(cm_hash)

            # Get compile info. Assume c++ always for now
            ci = cm('c++', rand=rand)

            # Check to make sure there isn't double quotes in the flags string, and remove the std from flags if it's there
            if any('"' in f for f in ci['flags']):
                raise ValueError("Folder flags string contains quotes!")
            flags = [f for f in flags if not f.startswith("--std=")]
            
            # Compile the folder
            bash_command = "CAP_BUILD_DIR=%s CXX=%s /mounted/%s %s" % \
                (os.path.join('/mounted', cap_build_temp), ci['binary_name'], os.path.basename(build_script), flags)
            command = COMPILE_COMMANDS['folder-' + container_platform].format(folder_path=folder_path, 
                container=_get_container(paths['containers'], ci['container'], container_platform), folder_build_command=bash_command)

            # Execute the command, catching any errors
            LOGGER.debug("Executing command: '%s'" % command)
            proc = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
            _, cmd_out = proc.communicate()
            cmd_out = None if cmd_out is None else cmd_out.decode("utf-8")

            # If there was an error, raise an error
            if cmd_out is not None and _is_true_compile_error(cmd_out):
                raise SubprocessCallError('Command failed with stderr:\n' + cmd_out)
            
            # Now, all binaries should be in build_dir/bin
            bin_dir = os.path.join(build_dir, 'bin')

            if not os.path.exists(bin_dir) or len(os.listdir(bin_dir)) == 0:
                raise ValueError("Building folder resulted in no binaries! They should be located at: %s" % bin_dir)
            
            # Go through each binary decompiling and analyzing with rose
            for bin_fp in [os.path.join(bin_dir, f) for f in os.listdir(bin_dir)]:
                compile_id = '-'.join([os.path.basename(folder_path), os.path.basename(bin_fp), d, str(d_i), str(cm_i)])
                compile_ids.append(compile_id)
                ret_analyzers.append(d)
                compile_info.append({k: ci[k] for k in _CINFO_KEYS})

                # Move the binary to a temporary location
                temp_bin_fp = os.path.join(paths['temp'], "%d-folder-%s.compiled" % (task_id, compile_id))
                shutil.move(bin_fp, temp_bin_fp)
                binary_file_paths.append(temp_bin_fp)

                analyzer_output_paths.append(run_analysis_cmd(paths, temp_bin_fp, a, container_platform, '-'.join(['misc', str(a_i), str(cm_i)])))
        
    # Delete the build_dir directory, and any self-made scripts
    shutil.rmtree(build_dir)
    if self_made_script:
        os.remove(build_script)

    return compile_ids, ret_analyzers, compile_info, binary_file_paths, analyzer_output_paths
"""