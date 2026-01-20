"""
Functions for doing various parsing of logfiles
"""

import re
import os
import subprocess
import shutil
import pandas as pd
from bincfg import progressbar


def parse_logs(log_info, data_type='codeforces', include=None, cap_path=None, filter_func=None, filter_loc=None, progress=False):
    """Parses error information out of logs

    Assumes extra data (if using for include) is in the normal locations relative to cap_path
    
    Args:
        log_info (Union[str, Iterable[str]]): the log information. Can be a string path to a logfile or an iterable of lines
        data_type (str): the type of data used to generate these logs. Available strings: 'codeforces'
        include (Union[None, str, Iterable[str]]): extra information to include. Can be

            * None - don't include any extra information (fast)
            * str - include this one set of available extra information. Available strings are:

                - 'metadata'/'meta': source metadata. Codeforces submission metadata if data_type=='codeforces'
                - 'source'/'sub'/'submission': the raw source code
                - 'bin'/'binary': the compiled binary. NOTE: this will enforce that both 'source' and 'metadata' are included
                - 'all': include all of the above data
            
            * Iterable[str] - include all of the data in this list
        
        cap_path (Optional[str]): the path to the cap directory, if including extra information
        filter_func (Optional[Callable[DataFrame, DataFrame]]): if not None, then a filtering function that takes in a
            pandas dataframe and returns a filtered pandas dataframe, removing any unneccessary/duplicate rows as you
            deem fit. Will be called before getting source or binaries.
        filter_loc (Optional[str]): if not None, then the location at which to apply the filter_func. By default, the
            function is applied after everything but the source code/binary part. Other places can be: 'premeta'
        progress (bool): if True, show progressbar/information
    """
    # Read in the logfile lines
    if isinstance(log_info, str):
        with open(log_info, 'r') as f:
            log_info = f.readlines()
    else:
        log_info = list(log_info)
    
    if data_type != 'codeforces':
        raise NotImplementedError("Data_type: %s" % repr(data_type))
    
    # Get the data to include
    include = [] if include is None else [include] if isinstance(include, str) else list(include)
    temp = set()
    for s in include:
        added, lower = False, s.lower()
        if lower in ['all', 'meta', 'metadata']:
            temp.add('metadata'); added=True
        if lower in ['all', 'source', 'sub', 'subs', 'submission', 'submissions']:
            temp.add('source'); added=True
        if lower in ['all', 'bin', 'binary', 'binaries']:
            temp.add('binary'); added=True
        if not added:
            raise ValueError("Unknown data include string: %s" % repr(s))
    include = temp

    if 'binary' in include:
        include.update(['metadata', 'source'])
    if len(include) > 0 and cap_path is None:
        raise ValueError("Must pass a cap_path if including extra data")

    # Get each submission execution as a list of string lines
    if progress: print("Parsing logfile into individual error executions...")
    err_sub_executions = []
    curr_execution = []
    started = False
    for line in progressbar(log_info, progress=progress):
        if 'Starting submission number' in line:
            if len(curr_execution) > 0 and any('Error: ' in l for l in curr_execution):
                err_sub_executions.append(curr_execution)
            curr_execution = []
            started = True
        elif not started:
            continue
        curr_execution.append(line)
    if len(curr_execution) > 0 and any('Error: ' in l for l in curr_execution):
        err_sub_executions.append(curr_execution)

    # Get the error information from each error execution lines list
    if progress: print("Parsing error information from individual executions...")
    ret = {'id': [], 'exception': [], 'traceback': [], 'rose_stderr': [], 'rose_stderr_errinfo': [], 'container': [], 'compile_command': []}
    for exec_list in progressbar(err_sub_executions, progress=progress):
        ret['id'].append(re.findall(r'ID: ([^,]*),', exec_list[0])[0])
        ret['container'].append((re.findall(r'/tmp/containers/([^.]*).sif', exec_list[1]) + [None])[0])
        ret['compile_command'].append((re.findall(r'/bin/bash -c "([^"]*)"', exec_list[1]) + [None])[0])

        joined_lines = ''.join(exec_list)
        if 'Rose printed warning with stderr:' in joined_lines:
            ret['rose_stderr'].append(joined_lines.split('Rose printed warning with stderr:\n')[1].split('\nTASK-')[0])
            ret['rose_stderr_errinfo'].append(_get_rose_error_info(ret['rose_stderr'][-1]))
        else:
            ret['rose_stderr'].append(None)
            ret['rose_stderr_errinfo'].append(None)
        ret['exception'].append(re.findall(r'\n([a-zA-Z0-9_.]*Error): ', joined_lines)[-1])
        ret['traceback'].append('Traceback ' + ''.join(joined_lines.split("\nTraceback ")[1:]))
    
    ret = pd.DataFrame(ret)

    # Do some dataset cleanup depending on the data_type
    if data_type == 'codeforces':
        ret['id'] = ret['id'].astype(int)

    # Check if there is a filter function and call it if so
    if filter_func is not None and filter_loc == 'premeta':
        if progress: print("Filtering into subset...")
        ret = filter_func(ret)
        filter_func = None
    
    # Get the metadata if using
    if 'metadata' in include:
        if data_type == 'codeforces':
            df_file = os.path.join(cap_path, 'raw_data', 'cf_submission_info.parquet')
            if progress: print("Loading metadata from file: %s" % df_file)
            ret = ret.merge(pd.read_parquet(df_file), how='left', left_on='id', right_on='submission_id', validate='many_to_one').drop(['submission_id'], axis=1)

            ret['source_link'] = ['https://codeforces.com/problemset/submission/%d/%d' % (ret.iloc[i].contest_id, ret.iloc[i].id) for i in range(len(ret))]
            ret['file_ext'] = [_get_file_ext(ret.iloc[i].programming_language) for i in range(len(ret))]
            ret['compiled_file_ext'] = ['' if ret.iloc[i].file_ext in ['cpp', 'c'] else '.class' if ret.iloc[i].file_ext in ['java'] else 'UNKNOWN_FILEEXT' for i in range(len(ret))]
        else:
            raise NotImplementedError("Metadata: %s" % data_type)
    
    # Check if there is a filter function and call it if so
    if filter_func is not None:
        if progress: print("Filtering into subset...")
        ret = filter_func(ret)
    
    # Get the source code if using
    if 'source' in include:
        if data_type == 'codeforces':
            part_dir = os.path.join(cap_path, 'raw_data', 'cf_partitioned')
            if progress: print("Loading source files from directory: %s" % part_dir)

            subs = []
            for f in progressbar([f for f in os.listdir(part_dir) if f.endswith('.parquet')], progress=progress):
                start, end = map(int, f.split('.')[0].split('-'))
                using_ids = ret[(ret.id >= start) & (ret.id < end)].id
                if len(using_ids) > 0:
                    chunk = pd.read_parquet(os.path.join(part_dir, f))
                    subs.append(chunk[chunk.submission_id.isin(using_ids)])
            
            ret = ret.merge(pd.concat(subs, axis=0), how='left', left_on='id', right_on='submission_id', validate='many_to_one').drop(['submission_id'], axis=1)
            ret['source'] = ret['submission']
        else:
            raise NotImplementedError("Source: %s" % data_type)
    
    # Compile into a binary if using
    if 'binary' in include:
        if 'source' not in include or 'metadata' not in include:
            raise ValueError("Must have both ")
        
        if progress: print("Compiling binaries...")
        ret['binary'] = [_compile_source(ret.iloc[i], data_type, cap_path) for i in progressbar(range(len(ret)), progress=True)]
    
    if progress: print("Done!")

    return ret.reset_index(drop=True)


def _get_rose_error_info(rose_err_str):
    """Returns all of the error information in a rose error string"""
    error_lines = []
    for l in rose_err_str.split('\n'):
        splits = re.split(r'Rose\[(FATAL)\]: ', l)
        if len(splits) > 1:
            error_lines.append('Rose[%s]: %s' % (splits[1], ''.join(splits[2:])))
    return '\n'.join(error_lines)


def _get_file_ext(language):
    if 'C++' in language or 'Clang++' in language:
        return 'cpp'
    elif re.fullmatch('.*C[0-9][0-9]?', language) or language in ['C', 'GNU C']:
        return 'c'
    elif 'Java' in language and 'JavaScript' not in language:
        return 'java'
    else:
        raise ValueError("Unknown programming language for file extenstion: %s" % repr(language))


def _compile_source(df_row, data_type, cap_path):
    """Compiles a source file, expects the source and metadata to be present in df_row
    
    Args:
        df_row: a row of a dataframe
        data_type (str): the string datatype
    """
    if data_type not in ['codeforces']:
        raise NotImplementedError("Compilation: %s" % data_type)
    
    if df_row.compile_command is None or df_row.container is None:
        return None
    
    temp_dir = os.path.abspath('./.temp_compile_dir')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    orig_compile_name = re.findall(r'/mounted/(.*)[.]%s' % df_row.file_ext, df_row.compile_command)
    if len(orig_compile_name) == 0:
        raise ValueError("Could not find original compile name (file_ext: %s) in compile command: %s" % (repr(df_row.file_ext), repr(df_row.compile_command)))
    orig_compile_name = orig_compile_name[0]
    
    sourcepath = os.path.join(temp_dir, orig_compile_name + '.' + df_row.file_ext)
    with open(sourcepath, 'w') as f:
        f.write(df_row['source'])
    
    container_path = os.path.join(cap_path, 'containers', '%s.sif' % df_row.container)
    command = 'singularity exec --bind %s:/mounted %s /bin/bash -c "%s"' % (temp_dir, container_path, df_row.compile_command)
    output_path = os.path.join(temp_dir, orig_compile_name + df_row.compiled_file_ext)
    
    proc = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
    _, cmd_out = proc.communicate()
    cmd_out = None if cmd_out is None else cmd_out.decode("utf-8")

    if cmd_out is not None and cmd_out != '':
        ret = "Error during compilation: %s" % cmd_out
    else:
        with open(output_path, 'rb') as f:
            ret = f.read()
    
    shutil.rmtree(temp_dir)
    return ret


def build_rose_error_dir(error_df, dirpath='./rose_errors'):
    """Saves this dataframe and all error files for rose team to the given directory path
    
    Args:
        error_df (DataFrame): a pandas dataframe of the alread-parsed error information from logfiles (including
            source code and binaries)
        dirpath (str): the path to a directory to store files in
    """
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    error_df.to_parquet(os.path.join(dirpath, 'rose_errors.parquet'))

    for i in range(len(error_df)):
        r = error_df.iloc[i]
        with open(os.path.join(dirpath, '%s.%s' % (r.id, r.file_ext)), 'w') as f:
            f.write(r.source)
        with open(os.path.join(dirpath, '%s%s' % (r.id, r.compiled_file_ext)), 'wb') as f:
            f.write(r.binary)
        

"""
Random other code:
\"""
Methods to help parse logs from cap process.
\"""
import re
import pandas as pd
import os
import subprocess
import multiprocessing
from utils.misc import get_language_family
from parsing.compiler_info import load_compiler_info
from compile.compile import run_single_file_compile_cmd, get_submission_dlls, run_analysis_cmd
from utils.misc import ensure_dirs
from utils.logs import ExceptionLogger, init_logging
from bincfg import progressbar


ROSE_OTHER_WARNINGS = ['data-flow max iterations reached', 'no dispatch ability for']
LANGUAGE_FILE_EXTENSIONS = {
    'c++': 'cpp',
    'c': 'c',
    'java': 'java'
}


def parse_debug(lines, cap_path, num_threads=1, get_extra_info=False):
    ret = pd.DataFrame({'error_method': [], 'error_type': [], 'traceback': [], 'compile_info': [], 'stderr_output': [], 
        'rose_no_dispatch': [], 'submission_id': [], 'line_idx': []})
    line_idx = 0

    # Keep track of backwards things
    last_error_start = None
    last_start = None
    last_compile_info = None

    line_lists = []

    while line_idx < len(lines):
        line = lines[line_idx]
        if len(line.strip().replace('\n', '')) == 0:
            line_idx += 1
            continue

        # Check for a new start
        if 'Starting submission number' in line:
            # Finish things up if needed
            if last_error_start is not None:
                line_lists.append((lines[last_error_start:line_idx], lines[last_start], \
                    lines[last_compile_info] if last_compile_info is not None else None, last_start))

            last_start = line_idx
            last_error_start = None
            last_compile_info = None
        
        # Check for compilation info
        elif 'Executing command:' in line and '-w' in line:
            last_compile_info = line_idx
        
        # Check for an error start
        elif 'ERROR' in line and line.split(" ")[1] == 'ERROR':
            last_error_start = line_idx
        
        line_idx += 1
    
    # Parse the errors
    if num_threads > 1:
        with multiprocessing.Pool(num_threads) as pool:
            errs = pool.starmap(_parse_error, line_lists, chunksize=max(1, int(len(line_lists) / 1000)))
    else:
        errs = [_parse_error(*l) for l in line_lists]
    
    # Convert into dataframe
    ret = {}
    for e in errs:
        for k, v in e.items():
            if k not in ret:
                ret[k] = []
            ret[k].append(v)
    lengths = [len(v) for k, v in ret.items()]
    if any(l != lengths[0] for l in lengths):
        raise ValueError("Got differently lengthed lists!")
    ret = pd.DataFrame(ret)
    
    if get_extra_info:
        extra_info = get_sub_info(*list(ret.submission_id), cap_path=cap_path).sort_values('submission_id').reset_index(drop=True)
        ret = ret.sort_values('submission_id').reset_index(drop=True)
        for k in [c for c in extra_info.columns if c not in ['submission_id']]:
            ret[k] = extra_info[k]
    
    ret = ret.astype({'submission_id': int, 'line_idx': int})
    return ret.drop_duplicates(subset=['error_method', 'traceback', 'stderr_output'])


def _parse_error(lines, start_line, compile_info_line, line_idx):
    ret = {'compile_info': None, 'stderr_output': None, 'rose_no_dispatch': None}

    # Join the lines and split to find the traceback/error_type
    joined_lines = ''.join(lines)
    print(joined_lines)
    split_err = re.split(r'\n((?:[a-zA-Z]+[.])*[A-Z][a-z]*(?:[A-Z][a-z]*)*Error)(?:: |\n|$)', joined_lines, maxsplit=1)

    # If there is no traceback, don't worry about the traceback (thanks for coming to my ted talk)
    if len(split_err) > 1:
        try:
            ret['traceback'] = '\n'.join(split_err[0].split('\n')[1:]) + '\n' + split_err[1] + ": " + split_err[2].split('\n')[0]
            ret['error_type'] = split_err[1]
        except Exception as e:
            raise ValueError("Error on line: %d, \n%s\nReason:\n%s" % (line_idx, lines, e))
    else:
        ret['error_type'] = 'warning'
        ret['traceback'] = ''

    ret['line_idx'] = line_idx

    # Get the compile info and submission id of this error
    ret['submission_id'] = int(start_line.split('ID: ')[1].split(',')[0])
    if compile_info_line is not None:
        ret['compile_info'] = ' '.join(compile_info_line.split('"')[1].split(' ')[:-4])

    # Parse out special error types (rose and compilation)
    if ret['error_type'] in ['SubprocessCallError', 'cap.SubprocessCallError', 'warning', 'FileNotFoundError']:
        
        # If this is a ROSE error (Specifically one that was catastrophic)
        if 'Rose printed warning with stderr:' in joined_lines:
            ret['error_method'] = 'rose'
            ret['stderr_output'] = joined_lines.split("Rose printed warning with stderr:\n")[1]
            ret['rose_no_dispatch'] = ','.join(set([l.split(' ')[0][1:-1] for l in ret['stderr_output'].split('no dispatch ability for ')[1:]]))
            ret['rose_no_dispatch'] = ret['rose_no_dispatch'] if ret['rose_no_dispatch'] != '' else None

        # Otherwise this should be a compiler error
        elif 'SubprocessCallError: Command failed with stderr:' in joined_lines:
            ret['error_method'] = 'compile'
            ret['stderr_output'] = joined_lines.split("SubprocessCallError: Command failed with stderr:\n")[1]

        # A FileNotFoundError can occur too, who knows what's going on anymore...
        elif 'FileNotFoundError: Could not find rose file:' in joined_lines:
            ret['error_method'] = 'compile'
            ret['stderr_output'] = joined_lines.split("Rose printed warning with stderr:\n")[1]
        
        else:
            raise ValueError("Couldn't parse SubprocessCallError at line index: %d" % line_idx)

    # Otherwise this is a boring, plain error
    else:
        ret['error_method'] = 'other'
    
    return ret


def get_rose_no_dispatch(errors, curr_i=None, print_errs=True):
    \"""
    Gets/prints info on ROSE no dispatch errors, and returns a dataframe of only these unique rose errors
    \"""
    curr_i = set() if curr_i is None else curr_i
    rose_errs = errors[errors.error_method == 'rose']
    keep_subs = []

    for i in range(len(rose_errs)):
        row = rose_errs.iloc[i]

        if row.rose_no_dispatch is None:
            continue

        new_i = [i for i in row.rose_no_dispatch.split(',') if i not in curr_i and i != 'unknown']

        if len(new_i) == 0:
            continue

        curr_i = curr_i.union(new_i)
        keep_subs.append(row['submission_id'])

        ci_split = row['compile_info'].split(' ')
        if print_errs:
            print(', '.join(new_i), ', '.join(['Compiler: %s' % ci_split[0], 'Flags: "%s"' % ' '.join(ci_split[1:]), 
                'Submission_id: %d' % row['submission_id'], 'Contest_id: %d' % row['contest_id'],
                'Codeforces link: %s' % row['link']]))
    
    return rose_errs[rose_errs.submission_id.isin(keep_subs)]


def print_other_rose_error_info(errs):
    \"""
    Prints the info for other rose errors
    \"""
    for i in range(len(errs)):
        row = errs.iloc[i]
        compiler, flags = get_cfc(row['compile_info'])
        print(', '.join(['Compiler: %s' % compiler, 'Flags: "%s"' % flags, 
                    'Submission_id: %d' % row['submission_id'], 'Contest_id: %d' % row['contest_id'],
                    'Codeforces link: %s' % row['link']]))


def get_rose_other_errors(errs):
    \"""
    Returns a dataframe of all errors that are rose errors, but not no_dispatch errors
    \"""
    rose_errs = errs[errs.error_method == 'rose']
    warnings = set()
    keep_subs = []

    for i in range(len(rose_errs)):
        row = rose_errs.iloc[i]

        # Go through each split on '[WARN  getting all of the different warnings that are not 'no dispatch'
        for warning in row.stderr_output.split('[WARN')[1:]:
            warning = ' '.join(warning.replace(']:', '').strip().split('\n')[0].split(' ')[:4])
            if warning not in ROSE_OTHER_WARNINGS and warning not in warnings:
                keep_subs.append(row.submission_id)
                warnings.add(warning)
    
    return rose_errs[rose_errs.submission_id.isin(keep_subs)], warnings


def get_sub_info(*submission_ids, sub_info=None, cap_path=None):
    \"""
    :return: Dataframe with columns ['submission_id', 'contest_id', 'problem_id', 'submission']
   \"""
    raw_data_path = os.path.join(cap_path, 'raw_data')
    sub_info_path = os.path.join(raw_data_path, 'cf_submission_info.parquet')
    df = pd.read_parquet(sub_info_path, columns=['submission_id', 'contest_id', 'problem_id', 'programming_language']) \
        if sub_info is None else sub_info
    df = df[df.submission_id.isin(submission_ids)].sort_values('submission_id')

    # Load in all of the submissions
    subs = []
    partitioned_path = os.path.join(raw_data_path, 'cf_partitioned')
    print("Reading files...")
    for file in progressbar([os.path.join(partitioned_path, f) for f in os.listdir(partitioned_path) \
        if os.path.join(partitioned_path, f) != sub_info_path and os.path.isfile(os.path.join(partitioned_path, f))]):
            chunk = pd.read_parquet(file, columns=['submission_id'])
            if len(chunk[chunk.submission_id.isin(submission_ids)]) > 0:
                chunk = pd.read_parquet(file)
                subs.append(chunk[chunk.submission_id.isin(submission_ids)])
    subs = pd.concat(subs, axis=0).sort_values('submission_id')

    df['submission'] = list(subs.submission)
    df['link'] = ['https://codeforces.com/problemset/submission/%d/%d' % 
        (df.iloc[i].contest_id, df.iloc[i].submission_id) for i in range(len(df))]
    df['language_family'] = [get_language_family(df.iloc[i].programming_language) for i in range(len(df))]
    return df


def generate_error_files(errors, cap_path):
    \"""
    Writes each submission that causes an error to a file (with its submission_id as the name and LANGUAGE_FILE_EXTENSION
        for its language family as the file extension). Then, compiles that file according to how it should have been
        compiled based on its flags in errors and outputs the compiled binary to a file with the submission_id as its
        name and no file extension.
   \"""
    temp_path = os.path.join(cap_path, '__rose_build_temp__')
    ensure_dirs(temp_path)
    ensure_dirs(os.path.join(cap_path, 'temp'))
    
    # Set up logging
    LOGGER = init_logging(cap_path, 0, file_name='_rose_err', with_stdout=True)

    compile_info = load_compiler_info('./compiler_info.yaml')
    
    for i in range(len(errors)):
        submission_id, submission, ci, lang, rose_log = errors.iloc[i][['submission_id', 'submission', 'compile_info', \
            'language_family', 'stderr_output']]
        
        with ExceptionLogger(LOGGER, handle=True, message="Error building rose error file: %d" % submission_id):
            ext = LANGUAGE_FILE_EXTENSIONS[lang]

            # Write raw file
            raw_file_path = os.path.join(temp_path, '%d.%s' % (submission_id, ext))
            with open(raw_file_path, 'w') as f:
                f.write(submission)
            
            # Write the rose log file
            with open(os.path.join(temp_path, '%d.log' % submission_id), 'w') as f:
                f.write(rose_log)
            
            # Compile it
            compiler, flags, container = get_cfc(ci, compile_info)
            ci = {
                'flags': re.sub(r'-o [^ ]*.compiled [^ ]*.(c |cpp )', '', flags),
                'binary_name': compiler,
                'container': container,
            }
            run_single_file_compile_cmd(ci, cap_path, 0, raw_file_path, get_submission_dlls(submission))

            # Move the compiled file, as well as the temporary submission file (in case it needed to be altered) over
            #   to the temp directory
            command = "mv %s %s & mv %s %s" % \
                (os.path.join(cap_path, 'temp', '0.compiled'), os.path.join(temp_path, '%d' % submission_id),
                 os.path.join(cap_path, 'temp', '0.%s' % ext), os.path.join(temp_path, '%d.%s' % (submission_id, ext)))

            LOGGER.debug("Executing command: '%s'" % command)
            proc = subprocess.Popen(command, shell=True, stderr=subprocess.PIPE)
            _, cmd_out = proc.communicate()
            cmd_out = None if cmd_out is None else cmd_out.decode("utf-8")
            if cmd_out is not None and cmd_out != '':
                LOGGER.info("File move command failed with response: %s" % cmd_out)


def get_cfc(ci, compile_info=None):
    \"""
    Returns a tuple of (compiler [IE: the binary name], flags, container) for a given single compilation info string
    \"""
    compiler, _, flags = ci.partition(' ')
    if compile_info is not None:
        return compiler, flags, get_container(compile_info, compiler)
    return compiler, flags


def get_container(compile_info, compiler):
    \"""
    Returns the container name for a given compiler binary_name
    \"""
    container = None
    for family, sd in compile_info.items():
        if container is not None: break
        for surname, vd in sd.items():
            if container is not None: break
            if surname in ['supported_languages']: continue
            for version, ad in vd.items():
                if container is not None: break
                for arch, arch_info in ad.items():
                    if compiler == arch_info['binary_name']:
                        container = arch_info['container']
                        break
    return container


def recompile_errors(errors, cap_path, error_methods=['compile']):
    \"""
    Recompiles all 'compile' errors in errors and returns a pandas dataframe of remaining errors
    \"""
    temp_dir = os.path.join(cap_path, '__compile_err_build_temp__')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    errors = errors[errors.error_method.isin(error_methods)]
    compile_info = load_compiler_info('./compiler_info.yaml')

    # Set up logging
    LOGGER = init_logging(cap_path, 0, file_name='_compile_err', with_stdout=True)

    # Go through each error
    for i in range(len(errors)):
        row = errors.iloc[i]

        # Get the information
        compiler, flags, container = get_cfc(row.compile_info, compile_info)
        flags = re.sub(r'-o [^ ]*.compiled [^ ]*.(c |cpp )', '', flags)
        file_ext = 'cpp' if row.language_family == 'c++' else 'c'
        if '--std=' not in flags:
            flags = ('--std=%s ' % ('gnu++11' if '++' in compiler else 'gnu11')) + flags
        cap_file_path = os.path.join(temp_dir, "0.%s" % file_ext)
        ci = {'binary_name': compiler, 'container': container, 'flags': flags, 'language': row.language_family}

        # Write submission to file
        with open(cap_file_path, 'w') as f:
            f.write(row.submission)

        LOGGER.debug("Starting submission number %d. ID: %d," % (i, row.submission_id))
        with ExceptionLogger(LOGGER, handle=True, message="Error building binary for submission_id %d:" % row.submission_id):
            compiled_file_path = run_single_file_compile_cmd(ci, cap_path, 0, cap_file_path, get_submission_dlls(row.submission))
            rose_file = run_analysis_cmd(cap_path, 0, compiled_file_path)
            print("Reading rose file: %s" % rose_file)
            #rose_cfg = RoseCFG(rose_file, file_type='gv', metadata={})
            #mem_cfg = MemCFG(rose_cfg, norm_method='safe')
    
    print("Re-Compilation complete!")


def get_known_ndi_from_str(s):
    \"""
    Returns a set of all of the known rose no dispatch instructions from a string. The string should just be a full
        copy-paste of the table on the confluence page
   \"""
    vals = set()
    for l in s.split('\n'):
        vals = vals.union(l.split('Compiler:')[0].strip().replace(' ', '').split(','))
    return vals

"""