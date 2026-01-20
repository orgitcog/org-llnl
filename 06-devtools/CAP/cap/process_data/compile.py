"""
Functions to compile source code

Each function should have signature - 

Args:
    paths (Dict[str, str]): the dictionary of file paths. Assumes that we are the only process working in the paths['temp'] dir
    ci (CompilerSelector): the compiler selector to use
    file_path (str): the path to the file to compile. This file may be copied during the compilation process.
    container_platform (Optional[str]): the container platform to use

Returns:
    Tuple[Union[str, List[str], Exception], Tuple[str, str]]: 2-tuple of (output_paths, std_out_and_err)
    
        - output_paths: string or list of strings for all compiled output paths. Often this will have only one element, 
          but for some languages (EG: java with multiple classfiles for a single source code) there may be multiple. Will
          return the main 'analyzable' file as the first index (EG: the file that you would pass through Rose for java 
          multiple classfiles). If there was an error, then the error can be returned here and will be immediately raised
          after saving the std_err and std_out for the error message
        - std_out_and_err: tuple of (stdout: str, stderr: str) decoded communication from compilation subprocess
"""

import os
import shutil
import subprocess
from utils.logs import MPLogger
from utils.misc import get_java_classname, MAX_STD_STR_LEN, get_symlink_binds
from parsing.container_info import get_container_platform, CONTAINER_PLATFORMS, MOUNTING_DIR, get_container_path


# Include files and corresponding flags for dlls. Currently holds: math.h,
DLL_IMPORT_FLAGS = ['-lm']

# Files that will be included on the command line because some codeforces submissions don't have them in the code
CPP_INCLUDE_FILES = ['climits', 'cstring', 'math.h', 'ctype.h', 'cstdio', 'vector', 'string', 'numeric', 'tuple']

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


def compile_java_file(paths, ci, file_path, container_platform):
    """Runs a single compilation for a single Java file"""
    with open(file_path, 'r') as f:
        source = f.read()
    
    classname = get_java_classname(source)
    binary_fp = os.path.join(paths['temp'], '%s.java' % classname)

    # Copy the java file to the correct name if needed. Java files must have same name as main class
    if file_path != binary_fp:
        shutil.copy(file_path, binary_fp)
    
    input_path = '/mounted/%s' % os.path.basename('%s.java' % classname)

    compiler_flags = _escape_string(' '.join(ci['flags']))

    # "compiler_path /mounted/raw_file_path compiler_flags"
    std_oe = _exec_compile_subprocess(paths, ' '.join([ci['binary_name'], input_path, compiler_flags]), ci, container_platform)

    # Look for all the classfiles matching this classname in case there were multiple
    main_fp = os.path.join(paths['temp'], '%s.class' % classname)
    all_classfiles = [os.path.join(paths['temp'], f) for f in os.listdir(paths['temp']) \
                        if (f == ('%s.class' % classname)) or (f.startswith(classname + "$") and f.endswith('.class'))]
    if main_fp not in all_classfiles:
        ret = ValueError("Could not find expected main java classfile after compilation: %s" % repr(main_fp))
    else:
        ret = [main_fp] + sorted([p for p in all_classfiles if p != main_fp])
    return ret, std_oe


def compile_dotnet_file(paths, ci, file_path, container_platform):
    """Runs a single compilation for a single dotnet file"""
    binary_fp_name = os.path.basename(file_path).rpartition('.')[0]
    output_selector = "/mounted/%s.dll" % binary_fp_name
    output_path = os.path.join(paths['temp'], '%s.dll' % binary_fp_name)
    input_path = '/mounted/%s' % os.path.basename(file_path)

    compiler_flags = _escape_string(' '.join(ci['flags']))

    command = ' '.join([ci['binary_name'], input_path, output_selector, compiler_flags])
    std_oe = _exec_compile_subprocess(paths, command, ci, container_platform)
    
    return output_path, std_oe


def compile_c_cpp_file(paths, ci, file_path, container_platform):
    """Runs a single compilation for a single C/C++ file"""
    binary_fp_name = os.path.basename(file_path).rpartition('.')[0]
    output_selector = "-o /mounted/%s" % binary_fp_name
    output_path = os.path.join(paths['temp'], binary_fp_name)
    input_path = '/mounted/%s' % os.path.basename(file_path)

    compiler_flags = _escape_string(' '.join(ci['flags']))

    # Extra dll's which sometimes needed to be manually included. It looks like they do not affect output binary when
    #   not used, even at -O0 optimization
    dll_str = ' '.join(DLL_IMPORT_FLAGS)

    # Extra files to include by default in case they are needed. So far, I have only found the need to apply this to
    #   c++ codeforces submissions, but after some testing, it looks like any included files that are not used do
    #   not affect the output code, even at -O0 optimizations
    if ci['language'] == 'c++':
        include_str = ''.join([(' -include %s' % incl) for incl in CPP_INCLUDE_FILES])

    # "compiler_path output_selector /mounted/raw_file_path compiler_flags includes dlls"
    command = ' '.join([ci['binary_name'], output_selector, input_path, compiler_flags, include_str, dll_str])
    std_oe = _exec_compile_subprocess(paths, command, ci, container_platform)
    
    return output_path, std_oe


def compile_single_file(paths, ci, file_path, container_platform):
    """Compiles a single file, selecting the appropriate compilation function based on the given language_family
    
    Args:
        paths (Dict[str, str]): the dictionary of file paths
        ci (Dict): the compile information to use
        file_path (str): the path to the file to compile. Expected to reside within the paths['temp'] directory. This 
            file may be copied during the compilation process.
        container_platform (Optional[str]): the container platform to use

    Returns:
        Tuple[Union[List[str], Exception], Tuple[str, str]]: 2-tuple of (output_paths, std_out_and_err)
        
            - output_paths: string or list of strings for all compiled output paths. Often this will have only one element, 
              but for some languages (EG: java with multiple classfiles for a single source code) there may be multiple. Will
              return the main 'analyzable' file as the first index (EG: the file that you would pass through Rose for java 
              multiple classfiles). If there was an error, then the error can be returned here and will be immediately raised
              after saving the std_err and std_out for the error message
            - std_out_and_err: tuple of (stdout: str, stderr: str) decoded communication from compilation subprocess
    """
    if ci['language'] in ['c', 'c++']:
        ret, (stdout, stderr) = compile_c_cpp_file(paths, ci, file_path, container_platform)
    elif ci['language'] in ['java']:
        ret, (stdout, stderr) = compile_java_file(paths, ci, file_path, container_platform)
    elif ci['language'] in ['c#']:
        ret, (stdout, stderr) = compile_dotnet_file(paths, ci, file_path, container_platform)
    else:
        raise NotImplementedError("Unknown language_family: %s" % repr(ci['language']))
    
    # Make sure return values from compile function are either string, list, or tuple
    if not isinstance(ret, (str, list, tuple, Exception)):
        raise TypeError("compile_single_file() got an unexpected return type: %s" % repr(type(ret).__name__))
    if isinstance(stdout, bytes):
        stdout = stdout.decode('utf-8')
    if isinstance(stderr, bytes):
        stderr = stderr.decode('utf-8')
    return ([ret] if isinstance(ret, str) else ret if isinstance(ret, Exception) else list(ret)), \
        (stdout[:MAX_STD_STR_LEN], stderr[:MAX_STD_STR_LEN])


def _exec_compile_subprocess(paths, compile_command, container_info, container_platform):
    """Set err_check_func to check for sure something is an error depending on output message, or None to assume all errors
    
    Args:
        paths (Dict[str, str]): dictionary of paths
        compile_command (str): command to execute within the container to compile
        container_info (Dict): dictionary mapping container platforms to containers to use
    """
    container_platform = get_container_platform(container_platform)

    container_dict = CONTAINER_PLATFORMS[container_platform]
    bind_cmd = container_dict['bind_command'].format(host_path=paths['temp'], container_path=MOUNTING_DIR)
    extra_args = get_symlink_binds(container_dict['bind_command'], paths['temp'])
    command = container_dict['execution_command'].format(dir_bind=bind_cmd, extra_args=extra_args, command=compile_command,
                                                         container=get_container_path(paths, container_info, container_platform))
    
    # Execute the command, catching any errors
    LOGGER.debug("Executing compile command: '%s'" % command)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    return stdout, stderr


def _escape_string(string):
    """Escapes both quotes and backslashes so they don't interfere with bash command"""
    return string.replace('\\', '\\\\').replace('"', '\\"')
