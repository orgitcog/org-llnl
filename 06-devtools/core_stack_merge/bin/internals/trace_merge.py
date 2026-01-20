import sys, os
from subprocess import *
import string
import merge
import getopt
import time

addr2line_exe = '/usr/bin/addr2line'
addr2line_map = {}

def get_traces(file_path, exe_path):
    global addr2line_map, addr2line_exe
    f = open(file_path, 'r')
    lines = f.readlines()
    current = []
    traces = []
    in_stack = False
    for line in lines:
        if line.find('+++STACK') != -1 or line.find('Function Call Chain') != -1:
            current = []
            continue
        if line.find("Frame Address") == 0:
            in_stack = True
        if line.find('---STACK') != -1 or line.find('End of stack') != -1:
            current.reverse()
            traces.append(current)
            current = []
            in_stack = False
            continue
        line = line.strip(' ')
        if line.find('0x') == 0 or in_stack == True:
            addr = line.split(' ')[-1]
            #print addr
            if addr in addr2line_map:
                #print 'reuse'
                current.append(addr2line_map[addr])
            else:
                #print addr
                output = Popen([addr2line_exe, '-e' , exe_path, '--demangle', '-s', '-f', addr], stdout = PIPE).communicate()[0]
                out_lines = output.split('\n')
                line_info = '%s@%s' %(out_lines[0], out_lines[1])
                line_info = line_info.replace('<', '\<').replace('>', '\>')
                current.append(line_info)
                addr2line_map[addr] = line_info

    # we need to reverse the order of the stack trace (oldest call first)
    return traces 

def get_high_rank(cores):
    # determine the highest ranked task for graphlib initialization
    high = 0
    for file in cores:
        if file.find('core.') != -1:
            rank = file[file.find('core.')+5:]
            rank = int(rank)
            if rank > high:
                high = rank
    return high

def print_usage():
    sys.stderr.write('\nThis tool will merge the stack traces from the user specified lightweight core files and output 2 .dot files, one with just function names, the other with function names + line number information\n')
    sys.stderr.write('\nusage:\n\tpython trace_merge.py -x <exe_path> -c <corefile>*\n\tpython tracer_merge.py -x <exe_path> -c <core_files_dir>\n')
    sys.stderr.write('\nnote: linux has a limit for the number of command line arguments.  For a large number of core files, specify -c <core_files_dir> instead of individual core files.\n')
    sys.stderr.write('\nexaples:\n\tpython trace_merge.py -x a.out -c core.0 core.1\n\tpython trace_merge.py -x a.out -c core.*\n\tpython trace_merge.py -x a.out -c ./\n\tpython trace_merge.py -x a.out -c core_dir\n')
    sys.exit(-1)

def parse_args():
    # first parse the core file arguments
    try:
        core_delim_index = sys.argv.index('-c')
    except:
        print_usage()
    core_args = sys.argv[sys.argv.index('-c') + 1:]
    sys.argv = sys.argv[0:sys.argv.index('-c')]

    # parse the args
    high = -1
    if len(core_args) == 1:
        if os.path.isdir(core_args[0]):
            cores = []
            for file in os.listdir(core_args[0]):
                if file.find('core.') != -1:
                    cores.append(core_args[0] + '/' + file)
        else:
            cores = [core_args[0]]
    else: 
        cores = core_args[0:]
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hx:f:t:H:", ["help", "exe=", "fileprefix=", "type=", "high="])
    except getopt.GetoptError, err:
        sys.stderr.write('\n%s\n' %str(err))
        print_usage()
    except:
        print_usage()
    exe_path = 'NULL'
    type = 'dot'
    file_prefix = 'NULL'
    
    for option, arg in opts:
        if option in ("-x", "--exe"):
            exe_path = arg
        elif option in ("-h", "--help"):
            print_usage()
        elif option in ("-f", "--fileprefix"):
            file_prefix = arg
        elif option in ("-t", "--type"):
            type = arg
            if not type in ['dot', 'raw']:
                sys.stderr.write('\nunknown file type "%s". Type may be either "dot" or "raw"\n' %(type))
                print_usage()
        elif option in ("-H", "--high"):
            high = int(arg)
        else:
            sys.stderr.write('unknown option %s\n' %(option))
            print_usage()

    if exe_path == 'NULL':
        sys.stderr.write('\nExecutable path not specified\n')
        print_usage()
    return exe_path, type, file_prefix, high, cores

def trace_merge_main(exe_path, cores, function_only_filename, line_number_filename, format = 'dot', high = -1, verbose = False):
    # initialize graphlib and generate the graph objects
    ret = merge.Init_Graphlib(high)
    if ret != 0:
        sys.stderr.write('Failed to initialize graphlib\n')
        sys.exit(1)
    function_only_handle = merge.New_Graph();
    if function_only_handle == -1:
        sys.stderr.write('Failed to create new graph\n')
        sys.exit(1)
    line_number_handle = merge.New_Graph();
    if line_number_handle == -1:
        sys.stderr.write('Failed to create new graph\n')
        sys.exit(1)

    # parse and merge the traces
    length = len(cores)
    j = -1
    for file in cores:
        j += 1
        if verbose:
            sys.stdout.write('\b\b\b\b%03u%%' %(j / (length / 100.0)))
            sys.stdout.flush()
        if string.find(file, 'core.') != -1:
            rank = file[string.find(file, 'core.')+5:]
            rank = int(rank)
            traces = get_traces(file, exe_path)
            for trace in traces:
                # create the trace for function name + line number info
                ret = merge.Add_Trace(line_number_handle, rank, trace)
                if ret != 0:
                    sys.stderr.write('Failed to add trace\n')
                    sys.exit(1)
    
                # create the trace for function names only
                function_only_trace = []
                for frame in trace:
                    function_only_trace.append(frame[:frame.find('@')])
                ret = merge.Add_Trace(function_only_handle, rank, function_only_trace)
                if ret != 0:
                    sys.stderr.write('Failed to add trace\n')
                    sys.exit(1)
    if verbose:
        sys.stdout.write('... done!\n')
    
    # now get a merge with only the function name
    if verbose:
        sys.stdout.write('outputing to file "%s" ...' %(function_only_filename))
    if format == 'dot':        
        merge.Output_Graph(function_only_handle, function_only_filename)        
    elif format == 'raw':
        merge.Serialize_Graph(function_only_handle, function_only_filename)
    if verbose:
        sys.stdout.write('done!  This file contains the merged stack traces with only function names.\n')

    # merge with line number info
    if verbose:
        sys.stdout.write('outputing to file "%s" ...' %(line_number_filename))
    if format == 'dot':        
        merge.Output_Graph(line_number_handle, line_number_filename)        
    elif format == 'raw':
        merge.Serialize_Graph(line_number_handle, line_number_filename)
    if verbose:
        sys.stdout.write('done!  This file contains the merged stack traces with function names and line number information.\n')

    if verbose:
        sys.stdout.write('View the outputted .dot files with `STATview`\n')
    return 0

if __name__ == '__main__':
    exe_path, type, file_prefix, high, cores = parse_args()
    if high == -1:
        high = get_high_rank(cores)

    #sys.stdout.write('merging %d core files\n' %len(cores))
    #sys.stdout.flush()

    if file_prefix == 'NULL':
        # make sure we generate unique file names
        function_only_filename = ''
        line_number_filename = ''
        for i in range(8192):
            if i == 0:
                function_only_filename = os.path.basename(exe_path) + '.' + type
                line_number_filename = os.path.basename(exe_path) + '_line.' + type
            else:
                function_only_filename = os.path.basename(exe_path) + '.' + str(i) + '.' + type
                line_number_filename = os.path.basename(exe_path) + '_line.' + str(i) + '.' + type
            if not(os.path.exists(function_only_filename) or os.path.exists(line_number_filename)):
                break
    else:
        function_only_filename = file_prefix + '.' + type
        line_number_filename = file_prefix + '_line' + '.' + type

    limit = 8192
    if len(cores) > limit:
        verbose = True
        sub_processes = {}
        chunks = len(cores) / limit
        if len(cores) % limit != 0:
            chunks += 1
        sys.stdout.write('spawning sub processes: 000%')
        sys.stdout.flush()
        for i in range(chunks):
            sys.stdout.write('\b\b\b\b%03u%%' %((i + 1) / (chunks / 100.0)))
            sys.stdout.flush()
            tmp_file_prefix = 'tmp.%d' %(i)
            if i == chunks - 1:
                cores_subset = cores[i * limit:]
            else:
                cores_subset = cores[i * limit:(i + 1) * limit]
            command = ['python', sys.argv[0], '-x', exe_path, '-f', tmp_file_prefix, '-t', 'raw', '-H', str(high), '-c'] + cores_subset
            sub_processes[i] = [tmp_file_prefix, Popen(command)]
        sys.stdout.write('\n')
        sys.stdout.flush()

        ret = merge.Init_Graphlib(high)
        if ret != 0:
            sys.stderr.write('Failed to initialize graphlib\n')
            sys.exit(1)

        total = len(sub_processes.keys())
        remain = total
        sys.stdout.write('synching sub processes: 000%')
        sys.stdout.flush()
        while remain > 0:
            time.sleep(.1)
            for i in range(remain):
                key = sub_processes.keys()[i]
                sub_process = sub_processes[key][1]
                ret = sub_process.poll() 
                if ret != None:
                    tmp_file_prefix = sub_processes[key][0]
                    function_only_file_path = '%s.raw' %(tmp_file_prefix)
                    line_number_file_path = '%s_line.raw' %(tmp_file_prefix)
                    if ret != 0:
                        sys.stderr.write('sub process %d of %d returned with error %d. Continuing without this subset\n' %(key, total, ret))
                        del sub_processes[key]
                        remain -= 1
                        os.remove(function_only_file_path)
                        os.remove(line_number_file_path)
                        break
                    sys.stdout.write('\b\b\b\b%03u%%' %((1 + total - remain) / (total / 100.0)))
                    sys.stdout.flush()
                    current_function_only_handle = merge.Deserialize_Graph(function_only_file_path)
                    if current_function_only_handle == -1:
                        sys.stderr.write('failed to deserialize file %s from sub process %d of %d. Continuing without this subset\n' %(function_only_file_path, key, total))
                        del sub_processes[key]
                        remain -= 1
                        os.remove(function_only_file_path)
                        os.remove(line_number_file_path)
                        break
                    os.remove(function_only_file_path)
                    current_line_number_handle = merge.Deserialize_Graph(line_number_file_path)
                    if current_line_number_handle == -1:
                        sys.stderr.write('failed to deserialize file %s from sub process %d of %d. Continuing without this subset\n' %(line_number_file_path, key, total))
                        del sub_processes[key]
                        remain -= 1
                        os.remove(line_number_file_path)
                        break
                    os.remove(line_number_file_path)
                    if (remain == total):
                        function_only_handle = current_function_only_handle
                        line_number_handle = current_line_number_handle
                    else:
                        ret = merge.Merge_Traces(function_only_handle, current_function_only_handle)
                        if ret == -1:   
                            sys.stderr.write('failed to merge handle %d into handle %d from sub process %d of %d. Continuing without this subset\n' %(function_only_handle, current_function_only_handle, key, total))
                        ret = merge.Merge_Traces(line_number_handle, current_line_number_handle)
                        if ret == -1:   
                            sys.stderr.write('failed to merge handle %d into handle %d from sub process %d of %d. Continuing without this subset\n' %(line_number_handle, current_line_number_handle, key, total))
                    del sub_processes[key]
                    remain -= 1
                    break
        sys.stdout.write('\n')
        sys.stdout.flush()

        # now get a merge with only the function name
        if verbose:
            sys.stdout.write('outputing to file "%s" ...' %(function_only_filename))
        if type == 'dot':        
            merge.Output_Graph(function_only_handle, function_only_filename)        
        elif type == 'raw':
            merge.Serialize_Graph(function_only_handle, function_only_filename)
        if verbose:
            sys.stdout.write('done!  This file contains the merged stack traces with only function names.\n')
    
        # merge with line number info
        if verbose:
            sys.stdout.write('outputing to file "%s" ...' %(line_number_filename))
        if type == 'dot':        
            merge.Output_Graph(line_number_handle, line_number_filename)        
        elif type == 'raw':
            merge.Serialize_Graph(line_number_handle, line_number_filename)
        if verbose:
            sys.stdout.write('done!  This file contains the merged stack traces with function names and line number information.\n')
    
        if verbose:
            sys.stdout.write('View the outputted .dot files with `STATview`\n')
        sys.exit(0)

    verbose = False
    if type == 'dot':
        verbose = True
    trace_merge_main(exe_path, cores, function_only_filename, line_number_filename, type, high, verbose)
