# Compile. Analyze. Prepare.

CAP (Compile. Analyze. Prepare.) is a python tool to help automate much of the tedious compile -> analyze -> preprocess
pipeline required when performing large-scale binary analysis.

The /cap directory contains the main launching script used to start a CAP process.

There are many ways to run it: single threaded, multi-threaded, multi-node HPC. This can compile competition-like data
(EG: codeforces, aizu), miscellaneous folders/projects, and can simply analyze precompiled binaries. It is made to be
(semi-) easily extendable to new compilers, languages, compiler flags, analyzers, and preprocessing steps. Information
on how to extend this code to add new features can be found in the 'Extend Me' section.

This project is well integrated with the CFG-handling tool BinCFG: https://github.com/LLNL/BinCFG


## Setup
This section will describe how to set up your data, directories, and configuration files to run CAP.

### Directories And Paths

There are a number of directories/paths needed to run CAP. To make things easier, you can pass a single directory with the
'--default_dir' flag to specify the default directory, within which it is assumed we will find most of the needed
directories which are not otherwise explicitly passed on the command line (IE: any directories passed will override
their expected location in the 'default_dir'). 

The directories/paths needed are -

    * '--atomic_data_dir' (default: "[default_dir]/atomic_data", automatically created): 
      directory to output atomically-updated data shared across multiple threads/processes/nodes
      (EG: normalized assembly tokens). It is expected that, if data is meant to be shared, then
      all processes that are running concurrently have access to this same folder within the 
      same filesystem. See the 'OUTPUT_DATA' section for more info on what will appear here.

    * '--containers_dir' (default: "[default_dir]/containers", should already exist): directory 
      containing all file-based containers that are needed to execute compilations and analysis. 
      This is needed for container programs like singularity, but would not be needed if using 
      something like docker. See the 'Containers' subsection for info on what containers should 
      reside here and how they should be built

    * '--logs_dir' (default: "[default_dir]/logs", automatically created): directory for logs

    * '--output_dir' (default: "[default_dir]/output", automatically created): directory for 
      output data. See 'OUTPUT_DATA' section for more info on what outputs are generated
    
    * '--input_path' (default: "[default_dir]/input", should be passed and already exist): 
      file/directory containing the raw data that should be CAP-ed. It must exist before 
      execution. Defaults to assuming the path at "[default_dir]/input" is a file/directory 
      containing data to CAP 
    
    * '--temp_dir' (default: "[default_dir]/temp", automatically created if nonexistent): 
      directory within which subdirectories will be created to house temporary scratch files 
      created during the CAP process. While I make every effort to ensure all temporary files 
      that are created are eventually deleted, this is not 100% ensured, especially if execution
      is interrupted due to exceptions, signals, etc. It is recommended to override the default
      directory for this temp_dir and set it to a node's temporary scratch directory when using
      HPC (these are often emptied between jobs on HPC)
    
    * '--partitioned_info_path' (default: "[input_path]/[exec_uid].parquet", should exist before 
      execution if using): path to a file containing the info for a 'partitioned' data CAP 
      process. Only used if performing the 'partitioned' CAP process. Defaults to a file within
      the input_path directory whose name is the execution uid of the current CAP process. See 
      the 'Input Data' subsection for more info.
    
    * '--partitioned_dir' (default: "[input_path]/partitioned", should exist before execution 
      if using): path to a folder containing all of the partitioned data for a 'partitioned' 
      data CAP process. Only used if performing the 'partitioned' CAP process. See the 
      'Input Data' subsection for more info on how partitioned files should be structured

    * '--container_info_path' (default: "[cap]/container_info.yaml", should already exist): path 
      to a YAML file containing information and configurations about containers used in the CAP 
      process. If not passed, then this will use the default "container_info.yaml" file shipped 
      along with this code expected to be right next to the 'CAP.py' file. See the 
      'Configuration Files' and 'Containers' subsections for info about using the default 
      containers shipped with this code, and the 'Extend Me' section for how this file is 
      structured and how to extend it to accomodate new containers.
    
    * '--execution_info_path' (default: "[cap]/execution_info.py", must exist before execution):
      path to a python file containing information about this CAP execution such as the 
      execution_uid, the data to keep, the normalizers/analyzers to use, etc. This file will be 
      imported, then all of the needed global constants will be imported. See the 
      'Configuration Files' section for more info on what global constants should be present

Within any of these paths, you may insert various strings that will be replaced when resolving the final paths:

    * "[default_dir]": replaces with the full path to the default directory, without a '/' at 
      the end. Cannot exist within the '--default_dir' directory path
    * "[input_path]": replaces with the full path to the input directory, without a '/' at the 
      end. Cannot exist within the '--input_path' or '--default_dir' path
    * "[main]" or "[cap]": replaces with the full path to the directory containing the "CAP.py" 
      file, without a '/' at the end
    * "[exec_uid]": replaces with the execution uid being used for this CAP execution

NOTE: if you do not pass a '--default_dir' value, then it is expected that every one of these file paths which depend on
that default path are also passed, even if they are not eventually used.

### Configuration Files

There are a couple of configurations needed before running:

    * container_info - YAML file containing information on containers being used including those
      for compilers and analyzers. By default, CAP will use the "container_info.yaml" file 
      shipped with this code, which should reside right next to 'CAP.py'. It should contain 
      everything needed for GCC and JavaC compilers and Rose analyzer. Information on how this 
      file is structure and how new containers can be added can be found in the 'Extend Me'
      section
    
    * execution_info - Information about the current CAP execution. By default, this resides 
      within the 'execution_info.py' file located next to the 'CAP.py' file. It exists as a 
      python file as it has been useful to make use of python code to generate some of this 
      information (EG: enumerating many compile_methods in for loops, getting IDE hints,
      etc.). I may want to find a better way of creating this configuration in the future...

      The execution info file must contain the global variables:

        - EXECUTION_UID (str): a unique string identifier of the current execution. This should 
          be the same between multiple threads/nodes working on this current execution. However,
          multiple executions could be run on the same folders or data at the same time without 
          clobbering each other so long as you change this EXECUTION_UID. This value will be 
          used in the naming of a lot of output files/logs and whatnot
      
      It may optionally contain the following variables. If not present, they will be set to 
      their defaults:
        
        - POSTPROCESSING (Optional[Union[str, List[str]]], default=[]): string or list of 
          strings for the postprocessings to apply to analyzer outputs, or None to not apply
          any. Available strings:

          * 'cfg': build a CFG() object, one for each of the normalizers in 
            exec_info['normalizers']
          * 'memcfg': build a MemCFG() object, one for each of the normalizers in 
            exec_info['normalizers']
          * 'stats': build a CFG() object and get the graph statistics with 
            cfg.get_compressed_stats(), one for each of the normalizers in 
            exec_info['normalizers']
      
          NOTE: these will be stored as pickled bytes() objects

        - DROP_COLUMNS (Optional[Union[str, List[str]]], default=[]): by default, all of the 
          data generated is kept. This if not None, can be a string or list of strings of the 
          column or group of columns to drop. You may also pass any columns that would appear 
          in the metadata, and those will be dropped. Metadata columns to drop can be passed 
          either as their original name, or with the prefix 'meta_' as they would appear in the 
          output data. Any columns that do not correspond to data being kept will raise an 
          error, unless they start with the prefix 'meta_', in which case it is assume that that
          column is a possible metadata column which may or may not exist. Available 
          non-metadata columns to drop: 
                  
          'analyzer', 'binaries', 'analyzer_output', 'metadata', 'error', 'compile_stdout', 
          'compile_stderr', 'analyzer_stdout', 'analyzer_stderr', 'compile_time', 
          'analysis_time', 'language_family', 'compiler_family', 'compiler', 'compiler_version',
          'architecture', 'flags'

          There are also a couple of special strings that will drop groups of columns including:

            * 'metadata': drop any metadata that was passed in metadata dictionaries
            * 'compile_info': drop all the compilation info
            * 'timings': drop all the timing information ('compile_time', 'analysis_time')
            * 'stdio': drop all the stdio information ('compile_stderr', 'analyzer_stdout', etc.)
            * 'stdout': drop all the stdout information ('compile_stdout', 'analyzer_stdout')
            * 'stderr': drop all the stderr information ('compile_stderr', 'analyzer_stderr')

          See the README.md for what all of these columns are.
        
        - NORMALIZERS (Optional[Union[str, Normalizer, Iterable[Union[None, str, Normalizer]]]], 
          default=[]): normalizers to use when building postprocessing CFG's. Will build those 
          CFG's once for each of the normalizers here. Can be:

          * None: will not normalize at all (use raw input) for CFG's, will use 'unnormalized' 
            normalization (default BaseNormalizer()) for MemCFG's
          * str: string name of a normalizer to use. See BinCFG documentation for list of 
            available normalizer strings
          * Normalizer: Normalizer-like object to use. See BinCFG documentation
          * Iterable of any of the above: will do one of each normalization for each datapoint
        
        - ANALYZERS (Optional[Union[str, List[str]]], default=[]): string or list of strings for
          the analyzers to use. Should exist in the container_info YAML file. Can be empty if 
          you wish to not analyze files, but only compile
        
        - COMPILE_METHODS (Optional[List[Dict[str, Any]]], default=[]): list of dictionaries of 
          compile methods to apply. See the 'Compile Methods' subsection for more info. Can be 
          empty if using CAP task 'binaries' to analyze pre-compiled binaries

### Containers

You must have containers made before execution. These can currently either be singularity image files within the
'--containers_dir' directory, or prebuilt docker images ready in the global docker. 
All of the code used to build these containers is in the '/singularity' directory at the top of this repository. See
that directory's readme for info on how to set up and build singularity/docker containers used in CAP

The currently available container platforms are:

  - 'docker'
  - 'singularity'

### Input Data

There are multiple forms of input data depending on what CAP-ing you wish to perform. These can often be automatically
detected, see the 'Automatic Detection' subsection for more info

#### Partitioned Data

Assumes your input is partitioned into multiple parquet files based on an integer key 'id'. Makes the following assumptions:

  - The '--task' passed was 'partitioned' or automatically detected to be 'partitioned'
  - There is a parquet file located at '--partitioned_info_path' (defaults to 
    "[input_path]/[exec_uid].parquet" if not passed) which contains the metadata about the 
    source code files being CAP-ed for this execution uid. This file should contain the columns 
    "id" and "programming_language" designating a unique integer id per individual source code, 
    and a string programming language identifier respectively. The "id" is the key that will be 
    used to look up the source code inside the 'partitioned' folder. This file may contain other
    columns as well which will be added to each CAP-ed file as metadata. All other columns will 
    be considered metadata
  - There is a folder located at '--paritioned_dir' (defaults to "[input_path]/partitioned" if
    not passed) which contains one or more parquet files. These parquet files will be 
    partitioned by the integer "id" key such that each file contains a contiguous range of id's.
    Each parquet file should be named "[start]-[end].parquet" where 'start' is the starting id 
    (inclusive) and 'end' is the ending id (exclusive) of the range for that file. Each file 
    should contain the columns "id" for the id key and "source" for the source code. Any other 
    columns will be ignored.

#### Source Code

A single source code file. You can pass either 'source' or 'source-[language]' as the task with '[language]' being the
programming language used (if not there, then the language will be automatically detected). Makes the following assumptions:

  - The '--task' passed was 'source' or 'source-[language]', or was automatically determined to
    be one of those. If '[language]' is not present, then the source code programming language 
    will be automatically determined. See 'Languages and File Types' for info on source code 
    programming languages
  - The '--input_path' points to a single source code file
  - You have passed one or more compile methods. See the 'Compile Methods' subsection for more 
    info

The 'id' column for this will be the filename.

#### Binary

A single precompiled binary file. You can pass either 'binary' or 'binary-[file_type]' as the task with '[file_type]' 
being the type of the binary (currently not used, but may be used in the future). If '[file_type]' is not passed,
it will be automatically detected. Makes the following assumptions:

  - The '--task' passed was 'binary' or 'binary-[file_type]', or was automatically determined to 
    be one of those. If '[file_type]' is not present, then the binary file type will be 
    automatically detected. See 'Languages and File Types' subsections for info on binary file
    types
  - The '--input_path' points to a single precompiled binary file

The 'id' column for this will be the filename.

#### Tabular

A single file containing some tabular data (EG: csv, parquet, etc.). You can pass either 'tabular' or 'tabular-[file_type]'
as the task with '[file_type]' being the type of tabular data. If '[file_type]' is not passed, it will be automatically 
detected. Makes the following assumptions:

  - The '--task' passed was 'tabular' or 'tabular-[file_type]', or was automatically determined 
    to be one of those. If '[file_type]' is not present, then the tabular file type will be 
    automatically detected. See 'Languages and File Types' subsections for info on tabular file 
    types
  - The '--input_path' points to a single tabular file
  - If compiling and analyzing, then this file contains at least the columns 'id', 'source', and
    'programming_language'. Source codes will be compiled and analyzed based on their 
    programming language. Any extra columns will be treated as metadata. If the column 'binary'
    is also present in addition to the 'source' and 'programming_language' columns, then the 
    'binary' column will be ignored and CAP will default to compiling/analyzing instead of 
    purely analyzing
  - If only analyzing, then this file contains at least the columns 'id' and 'binary'. No 
    compilation will be performed. Any extra columns will be treated as metadata. If the file 
    also contains the columns 'source' and 'programming_language', then this option will not be 
    chosen, and CAP will default to compiling/analyzing instead of purely analyzing.

#### Project

A folder containing all the files relevant to a single project. You can pass either 'project' or 'project-[build_type]'
as the task with '[build_type]' being the type of project being built. If '['build_type]' is not passed, it will be
automatically detected. Makes the following assumptions:

  - The '--task' passed was 'project' or 'project-[build_type]', or was automatically determined
    to be one of those. If '[build_type]' is not present, then the project build type will be 
    automatically detected. See 'Languages and File Types' subsections for info on tabular file
    types
  - The '--input_path' points to a directory containing all the files required for the project 
    to be built
  - There exists a special file within this directory that will determine how things are CAP-ed.
    This file can be:

    1. 'CAP.json': a file special to CAP. It should be in JSON format. The object should be a 
       dictionary, and it can have the following keys/values:

      * 'only_cap': a string or list of string filenames relative to this directory for the 
        file/files that should be CAP-ed in this project. This is useful for times when there 
        needs to be multiple files in the same directory as the main CAP file while 
        compiling/analyzing (EG: header files when compiling, shared libraries when analyzing 
        with rose, etc.). Files should only be either source code or binary files, not tabular 
        files or project directories. Files should be auto-detectable
      
      NOTE: dictionary keys will be searched for in the above order. If there are multiple
      conflicting keys, the first one found in the order is what will be used
    
    NOTE: files will be searched for in the above order. If there are multiple conflicting files
    the first one found is what will be used

#### Miscellaneous Folders/Files

A folder containing some number of files or subfolders to CAP. You should pass the 'misc' task or 'misc-recursive' task, 
or if left to 'auto' and another task couldn't be determined, then CAP will default to 'misc' CAP task. The 'misc-recursive'
task is the same as 'misc', but will recursively check subfolders for more files/projects to CAP. Makes the following assumptions:

  - The '--task' passed was 'misc' or 'misc-recursive', or was automatically determined to be 
    that
  - The '--input_path' points to a directory containing some number of files/folders. Files will
    have their types automatically detected, while folders are always assumed to be project 
    folders (See the 'Project' subsection).

The 'id' column for files/projects will be built off of their filepaths relative to the '--input_path' directory like so:

  - source code/binaries: 'file@[filepath]' where '[filepath]' is the path to that file relative
    to the '--input_path' directory
  - tabular data: 'tabular@[filepath]-[id]' where '[filepath]' is the path to that file relative
    to the '--input_path'  directory and '[id]' is the value of the 'id' column in the tabular 
    data
  - projects: 'project@[folderpath]-[binary_filename]' where '[folderpath]' is the path to that 
    project's folder relative to the '--input_path' directory and '[binary_filename]' is the 
    name of the compiled binary built by that project (NOTE: it's possible for a project to 
    build multiple binaries)
  
#### Note on symlinks:

Whenever symlinks are present (either within a project folder, or when CAP-ing a standalone file), the resolved symlink
paths will be added as bound directories when using containers automatically. This will apply recursively to project
folders.

### Languages and File Types

CAP currently has built-in support for a few programming languages and file types (though, others may be added somewhat 
easily, see the 'Extend Me' section for more info).

#### Programming Languages

The following programming languages can be CAP-ed (assuming the relevant containers exist) without any modification to CAP:

  - 'c', 'c++': C and C++ programming languages
  - 'java': Java programming language

These can be automatically detected as source code files as well as handled during compilation and analysis.

#### Binary Types

These are not currently used, however we can currently automatically detect the following binary types based on their
'magic' bytes:

  - 'elf': ELF files
  - 'pe': Windows PE files
  - 'macho': MacOS MachO files
  - 'java': Java Classfiles

#### Tabular Types

The following tabular data types can be automatically detected and handled:

  - 'csv': CSV files
  - 'parquet': Parquet files

#### Project Types

The following project build types are available:

  - 'cap': allows for fine-grained control over what happens based on a CAP.json file located 
    in the directory
  - 'cmake': blah blah blah

### Precomputed Tokens

The '--atomic_data_dir' folder can contain precomputed tokens used for doing BinCFG postprocessing (creating CFG's, 
MemCFG's, etc.). These precomputed token files should have their names start with "precomputed_tokens", and should be
pickle files containing a 2-tuple of (norm, tokens), where 'norm' is the BinCFG normalizer used and 'tokens' is a dictionary
mapping string tokens to their integer value. These are useful as the default AtomicTokenDict in BinCFG can be slow when
trying to atomically update the token dictionary file for many tokens/processes at once. Having a good chunk of the
tokens precomputed can reduce the number of atomic updates required by a lot.

### Output Data

Most output data will appear in the '--output_dir' directory. Each running process will output its own data into one
or more parquet files. It will save data in chunks, saving to a new file every time the current chunk begins to take up
too much memory. Each file will have filepath "[output_dir]/[exec_uid]-[task_id]-[save_idx].parquet", where 'task_id'
is the id (integer) of the current process's task (in case you are using multithreading/multi-node), and 'save_idx' is
the index of the current chunk of data being saved. The parquet file can have the following columns:

  - 'id': the id of the datapoint, usually either int or string
  - 'analyzer': the string name of the analyzer used, or None if no analysis was performed or an
    error occurred before or during analysis.
  - 'language_family': the string language family used when compiling (See 
    'Programming Languages' subsection for list of currently available language families by 
    default), or None if no compilation was performed or an error occurred before or during 
    compilation
  - 'compiler_family': the string compiler family used when compiling. These will be the main 
    names of compiler families in the container_info.yaml file. Will be None if no compilation
    was performed or an error occurred before or during compilation
  - 'compiler': the string name of the compiler used when compiling, or None if no compilation 
    was performed or an error occurred before or during compilation
  - 'compiler_version': version string of the compiler used when compiling, or None if no 
    compilation was performed or an error occurred before or during compilation
  - 'architecture': the string architecture name compiled to when compiling, or None if no 
    compilation was performed or an error occurred before or during compilation
  - 'flags': list of string flags passed to compiler when compiling, or None if no compilation 
    was performed or an error occurred before or during compilation
  - 'binaries': list of all output binaries associated with this cap file. Each binary is a 
    bytes() object containing the full bytes of the compiled binary. For languages like c/c++, 
    this list will likely have only one element (the compiled binary), but for others like Java,
    this list may contain multiple elements (one for each classfile produced).
  - 'binary_md5_hashes': list of md5 hashes of output binaries
  - 'total_size_mb': total size of all binaries in MB
  - 'analyzer_output': the string text output from the analyzer
  - 'error' (List[Optional[str]]): string error message for any error occurr during CAP. Will 
    be None if no error occurred. Specifically, this is an error from within python, not an 
    error from compile/analysis stderr
  - 'compile_stdout' (List[Optional[str]]): string output from stdout during compilation 
    process, or None if no compilation was performed or an error occurred before or during 
    compilation
  - 'compile_stderr' (List[Optional[str]]): string output from stderr during compilation 
    process, or None if no compilation was performed or an error occurred before or during 
    compilation
  - 'analyzer_stdout' (List[Optional[str]]): string output from stdout during analyzer process, 
    or None if no analysis was performed or an error occurred before or during analysis
  - 'analyzer_stderr' (List[Optional[str]]): string output from stderr during analyzer process, 
    or None if no analysis was performed or an error occurred before or during analysis
  - 'compile_time' (List[Optional[float]]): time in seconds required to compile, or None if no 
    compilation was performed or an error occurred before or during compilation
  - 'analysis_time' (List[Optional[float]]): time in seconds required to analyze, or None if no 
    analysis was performed or an error occurred before or during analysis
  - 'cfg_[norm_name]_[idx]': bytes() containing pickled BinCFG CFG() object normalized with the
    normalizer '[norm_name]'. The '[idx]' is just an integer index in the list of normalizers 
    being used in case multiple normalizers are passed which have the same name. One column will
    exist per normalizer used iff the 'cfg' string is present in the `postprocessing` 
    EXECUTION_INFO list
  - 'memcfg_[norm_name]_[idx]': bytes() containing pickled BinCFG MemCFG() object normalized
    with the normalizer '[norm_name]'. The '[idx]' is just an integer index in the list of 
    normalizers being used in case multiple normalizers are passed which have the same name. One
    column will exist per normalizer used iff the 'memcfg' string is present in the 
    `postprocessing` EXECUTION_INFO list
  - 'stats_[norm_name]_[idx]': bytes() containing pickled BinCFG CFG().get_compressed_stats() 
    numpy array normalized with the normalizer '[norm_name]'. The '[idx]' is just an integer 
    index in the list of normalizers being used in case multiple normalizers are passed which 
    have the same name. One column will exist per normalizer used iff the 'stats' string is
    present in the `postprocessing` EXECUTION_INFO list
  - 'meta_[metadata_key]': metadata associated with the CAP-ed datapoint. Each key that was in 
    the metadata will have 'meta_' prepended to it and its value will be stored in its 
    associated column
  
Any of these columns can be dropped and will not appear in the output files if their name appears in the DROP_COLUMNS
parameter in the execution info.

The only other set of possible output data is BinCFG AtomicTokenDict tokens, which will appear in the "--atomic_data_dir"
directory

### Automatic Detection

Files and folders can be automatically detected depending on what task was passed. When the 'auto' task is passed, we
check in order:

  1. Does the '--input_path' point to a directory? If so:
     1.a Are the '--partitioned_dir' and '--partitioned_info_path' passed and valid folders/files? If so, assume task='paritioned'
     1.b Otherwise, check this folder for a special file designating it as a project build folder. EG: a 'CMakeLists.txt' file.
         If so, assume task='project'
     1.c Otherwise, assume task='misc'
  2. Otherwise, assume we are pointing to a file:
     2.a Check if we were pointing to a tabular file. If so, assume task='tabular'
        2.a.0 Does this file end with a known file extension for tabular files? EG: '.csv', '.parquet'
        2.a.1 Does this file start with known magic bytes? EG: "PAR1" for parquet files
     2.b Check if we were pointing to a precompiled binary file. If so, assume task='binary'
        2.b.0 Does this file start with known binary magic bytes? EG: 0xCAFEBABE for java classfiles
     2.c Check if we were pointing to a source code file. If so, assume task='source'
        2.c.0 Does this file contain known strings that uniquely (or likely uniquely) designate this file as a particular
              language? EG: something similar to "public static void main(String[] args)" for Java
     2.d We couldn't auto-detect the file, raise an error


## Running CAP

CAP can be run in two main ways: from the command line, and by importing CAP.py and calling cap_main(). This code doesn't
have the capability to actually execute HPC jobs, that is left to the user. However, there is a run_cap.sh file if you
are by chance using singularity and the SLURM job manager which you could modify.

### Command Line

<pre>

usage: CAP.py [-h] [--atomic_data_dir ATOMIC_DATA_DIR] [--containers_dir CONTAINERS_DIR] [--logs_dir LOGS_DIR] [--output_dir OUTPUT_DIR]
              [--input_path INPUT_PATH] [--partitioned_info_path PARTITIONED_INFO_PATH] [--partitioned_dir PARTITIONED_DIR]
              [--container_info_path CONTAINER_INFO_PATH] [--temp_dir TEMP_DIR] [--execution_info_path EXECUTION_INFO_PATH]
              [--default_dir DEFAULT_DIR] [-n N_JOBS] [-t TASK_ID] [--threads THREADS] [--task TASK] [--container_platform CONTAINER_PLATFORM]
              [--fail_on_error] [--hpc_copy_containers] [--specific_tasks SPECIFIC_TASKS]

Compile. Analyze. Prepare.

optional arguments:
  -h, --help            show this help message and exit
  --atomic_data_dir ATOMIC_DATA_DIR
                        The path to a directory for atomic data
  --containers_dir CONTAINERS_DIR
                        The path to a directory for containers
  --logs_dir LOGS_DIR   The path to a directory for log files
  --output_dir OUTPUT_DIR
                        The path to a directory for output data
  --input_path INPUT_PATH
                        The path to a directory/file for the input data
  --partitioned_info_path PARTITIONED_INFO_PATH
                        The path to a parquet file containing the metadata and id keys for a "partitioned" CAP process
  --partitioned_dir PARTITIONED_DIR
                        The path to a directory containing partitioned data for a "partitioned" CAP process
  --container_info_path CONTAINER_INFO_PATH
                        The path to a YAML file containing the container_info. Defaults to "container_info.yaml" file 
                        assumed to be right next to this file.
  --temp_dir TEMP_DIR   The path to a directory for temporary files
  --execution_info_path EXECUTION_INFO_PATH
                        The path to a python file that should be imported to get the execution information
  --default_dir DEFAULT_DIR
                        The path to a directory for any unpassed default directories. Any missing directories will use 
                        default names and be subdirectories of this one
  -n N_JOBS, --n_jobs N_JOBS
                        The number of jobs running. If this argument is not passed, will first check the os enironment 
                        variable SLURM_ARRAY_TASK_COUNT. If that does't exist, then will assume n_jobs=1
  -t TASK_ID, --task_id TASK_ID
                        The task_id for this process. Should be in the range [0, num_jobs - 1] and unique for each process. 
                        If this argument is not passed, then the SLURM_ARRAY_TASK_ID environment variable will be used.
  --threads THREADS     Number of threads to use for this task.
  --task TASK           Which task to run. Can be: "auto", "partitioned", "source", "source-[language]", "project", 
                        "project-[build_type]", "binary", "binary-[binary_type]", "tabular", "tabular-[file_type]", 
                        "file", "folder", "misc", or "misc-recursive". See README.md for info
  --container_platform CONTAINER_PLATFORM
                        The container platform to use
  --fail_on_error       By default, most errors will be captured and saved silently into the output data during the CAP 
                        process. If this is True, then any error while CAP-ing a file/folder will instead be raised, an 
                        error will be printed to the log files, and that data will not be stored in the output files. 
                        This will not stop the entire CAP process, however, as files and folders will continue to be CAP-ed. 
                        This just makes the errors visible in the logs and doesn't save them along with the output data
  --await_load          If passed, then each thread within an execution will wait to begin loading its data until the 
                        previous thread has completed the data loading process to save memory during the intial 
                        loading/splitting phase
  --hpc_copy_containers
                        If this flag is passed, then it is assumed that we are running on HPC systems, and we should copy
                        container files from the given `containers_dir` into a temporary place on this node's in-memory 
                        filesystem for faster loading of containers. The 'containers' path will be automatically updated
                        to be "[temp_path]/containers" with all containers for the original "--containers_dir" directory
                        being copied into that path
  --specific_tasks SPECIFIC_TASKS
                        Specific task_id's you wish to run. Should be a comma separated list of integer task_id's. It is
                        assumed that if there are multiple tasks, then they should be run in parallel. If this is passed,
                        then you must pass most values directly. EG: `n_jobs` must be passed, `task_id` must not be passed,
                        `threads` should be the same value that was used during full execution and will not specify the 
                        number of threads to use to run these specific tasks (it is only used for proper logging, `task` 
                        must be passed and cannot be 'all', and `task_id_offset` must not be passed or must be set to the 
                        default value of 0.

</pre>

### Calling cap_main()

You can also import from CAP.py the cap_main() function and call that yourself. It has signature/docstring:

def cap_main(paths, exec_info, task, n_jobs=1, threads=1, task_id=0, hpc_copy_containers=False, container_platform=None, 
             specific_tasks=None):
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
    """


## Safety

There are currently some safety concerns with this code. Specifically, while YAML files are safely loaded, we currently
allow for arbitrary python code to be executed while loading the container_info YAML file (specifically, when loading
information about 'compiler' type containers). So long as you are using a trusted container_info file, you will be
fine. See the 'Container Info' subsection in the 'Extend Me' section for more info on why/when this happens (specifically,
the information about "Command Strings").

Another concern may be that the execution_info python file that is used to hold CAP execution information is imported
and run like a normal python file. So again, don't use untrusted files!


## Extend Me

Information on how to extend this code to new languages, analyzers, etc.

Adding new language:
  - Build singularity container for language compiler(s)
  - Add container info to container_info file
  - Add cleaning method to clean_source()
  - Add parser for language family to utils.misc.get_language_family() file
  - Add file format detections to misc
  - Add new compile_LANG_file() method to cap.process_data.compile.py file. See header for methods specs.
  - Add lang to cap.process_data.compile.compile_single_file() method
  - Add info to CAP.py in FILE_EXTENSIONS_TO_LANGUAGE and BINARY_FILE_EXTENSIONS
  - Update README.md and Docstrings :)
Adding new analyzer:
Adding new container platform:

### Container Info

The container_info file contains information about, who would have guessed, containers. It is a YAML file which should
look something like:

```
ContainerName1:
    type: ContainerType1
    container: ContainerPath1
    ...

ContainerName2:
    type: ContainerType2
    container:
      docker: ContainerPath2Docker
      singularity: ContainerPath2Singularity
```

The ContainerName is the name of the container, and how that container will be referenced in CAP code and settings (EG: 
the name of the compiler_family/analyzer being used). 

ContainerType is a string designating the type of that container. Currently available container types are: 'compiler',
'analyzer'. In case I forget to update this file, the definitely-up-to-date list of them can be found in the 
cap.parsing.container_info.CONTAINER_TYPE global variable.

The ContainerPath contains information on how to locate the container being used. It can be a:

  - string: the name of the container to use. For singularity files, this will be the location of the '.sif' file
    relative to the 'containers_dir'. If the name doesn't end in '.sif', then it will be added when looking for a
    singularity image file. For docker images, this will be the name of the docker image used. This way, one could
    enter the string 'rose-analysis', and CAP will look for a '[container_dir]/rose-analysis.sif' file when using
    singularity, or will attempt to use the 'rose-analysis' docker image when using docker.
  - dictionary: the name to use depending on the container platform being used. Currently available container platforms
    are: 'docker', 'singularity'

There may be multiple container informations within a file. Each of them will require different information depending
on what type of containter it is.

#### Analyzer Containers

These objects give information about analyzers, where to find them, how to use them, etc. They should look something
like:

```
AnalyzerName:
    type: 'analyzer'
    container: AnalyzerPath
    analysis_cmd: AnalysisCmd
```

With the analysis_cmd being the command which should be executed within that container in order to produce analysis
output. The input will be placed within a mounted folder at '/mounted/{binary_basename}', and the output is expected
to exist at '/mounted/{analyzer_output_basename}' once analysis is complete. You should use the strings "{binary_basename}"
and "{analyzer_output_basename}" as those will automatically be inserted into the string with .format() when running the analysis.

It is currently expected that the output will be a single file that should be read in as a string (not bytes), and that
the analyzer will not modify the original binary file.

#### Compiler Containers

These objects give information about compiler families and their compilers, versions, architectures, flags, etc. These
can get quite complicated, so there is a lot of syntactic sugar and defaults built in to help ease this and reduce
the size of the files.

##### Basic Structure

Each compiler container object should contain information for one or more compiler 'families' (also called compiler 
'collection' or compiler 'suite', EG: GCC, Clang, etc.). Information is structured as: 
family -> compiler -> version -> architecture (arch). That is, each compiler family has a number of compilers, each of 
those a number of versions, and so on:

    - 'family': a compiler family (collection/suite). EG: GCC, Clang, etc.
    - 'compiler': a single compiler in the family. EG: for GCC, these would be 'gcc', 'g++', 'gfortran', etc.
    - 'version': a single version of a compiler. EG: for gcc, these could be '5.5.0', '7.5.0', '11.3.0', etc.

      NOTE: version strings can optionally start with a 'v' and it will be ignored. EG: 'v5.5.0', 'v7.5.0', etc.
        
      Versions can either take the form of "[VERSION_NUMBER]" or "[VERSION_NUMBER]-[EXTRA_INFO]", where VERSION_NUMBER
      is a '.'-separated list of digits of any positive length (EG: '3', '7.5.0', '18.47.62.1234'), and EXTRA_INFO can 
      be any string to help differentiate versions with the same number. 
        
      Versions are comparable. They are compared first in order of their VERSION_NUMBER's, then their EXTRA_INFO. That
      is, they are compared by the integer values created by splitting thier VERSION_NUMBER on '.'s from left to right,
      then by their EXTRA_INFO as a plain string comparison. The lack of VERSION_NUMBER's at that index would make that
      version smaller than one that does have a VERSION_NUMBER at that index (same goes for EXTRA_INFO). 
      EG: "2" < "2.3.5" < "2.4" < "2.4.0" < "2.4.0-alpha" < "2.4.0-beta" < "2.5" < "3" < "18.5"
    
    - 'arch' (architecture): a single target architecture

At each level in this hierarchy, there are some metadata fields that should/can be present. The types of these fields can be:
    
    - "Optional": Optional, doesn't have to be here
    - "Partially-Optional": Optional only in the sense that this information doesn't necessarily have to be at this
      level, but must exist by the lowest level. EG: the 'binary_name' field must exist by the 'arch' level, but could
      theoretically exist at some level above it and be inherited.
    - "Inheritable": This field is inheritable, meaning all information lower down the hierarchy will by default inherit
      these values from those above it in the hierarchy (if available), and override it. EG: the 'force_flags' field can 
      be present at every level of the family -> compiler -> version -> architecture hierarchy. Those at lower levels 
      (say, 'version') would inherit the 'force_flags' of its parents ('compiler' and 'family' in this example) if it 
      exists in those levels. If the 'force_flags' field was present in this level ('version'), then it would override 
      those from above (how it overrides may be different depending on the field)

By default, if something is not Optional/Partially-Optional, then it is required at that level. If something is not
Inheritable, then it will only be used at that level.

NOTE: Names/values are case-sensitive.

##### Field Information
Information on all the fields that exist in the compiler:

    - 'binary_name' [Partially-Optional, Inheritable]: string name of the binary to use to compile (or path to that 
      compiler binary). Can be inherited, and overridden by lower levels. Needed by the arch-level
    
    - 'container' [Partially-Optional, Inheritable]: string basename (IE: without file extension) for filename of the
      container that contains the compiler binary. Needed by the arch-level
    
    - 'supported_languages' [Partially-Optional, Inheritable]: either string or list of strings for the language/languages
      that the compiler binary is able to compile. Needed by the arch-level. 
      NOTE: it is likely best to do this at the compiler or lower level
    
    - 'use_previous' [Optional]: allows one to use the previous value information at the current hierarchy level as a
      starting point. That information will be copied, and all information in this value will override that. EG: the 'g++' 
      compiler might use the 'gcc' compiler as a use_previous, in which case all information in the 'gcc' compiler will 
      be copied as used as a starting point for the 'g++' compiler, and all information in the 'g++' compiler would then
      override that data. The value can be either:

        * string: the string name of the value at that level in the hierarchy to use, so order in the file wouldn't matter
        * bool (True): only available at 'version' level. Uses the immediately previous version
        
    NOTE: currently it is only possible to use previous values at the same hierarchy level in the same parent
        
    The priority for data at sublevels varies depending on the key. The `flags` and `force_flags` at lower levels
    will override/add to those inherited from upper levels when using use_previous. All other flags will be overriden
    instead by those inherited.
    
    - 'flags' [Optional, Inheritable]: flags that are available to this value and all levels lower in the hierarchy. This
      should be a dictionary with flag/value pairs. Each 'flag' is the name of a compiler flag (Ignoring the first '-' 
      that would be present. If there are multiple '-'s that should be used, then add in all but the first one to the name). 
      Each value can be:

        * null (None): this has the effect of removing/ignoring that flag. Useful for removing deprecated/outdated 
          flags from versions that use_previous
        * boolean: This is a flag that is either present or not present, and has no value. The boolean value determines 
          whether or not there is an associated 'no-' flag as well. If False, then there is no extra flag. If True, then
          an extra possible 'no-' flag will be added by inserting the string 'no-' after the first letter of the flag name. 
          EG: "finline: True" would add the mutually exclusive flags 'finline' and 'fno-inline'.
        * string/int: a value to always use for this flag. Value will be directly appended to the flag name, so delimiters
          must be inserted here as a string if using. EG: to set a flag '--fake_flag=3', one would use "-fake_flag: '=3'",
          or for flag '-m32': "m: '32'" which incidentally is equivalent to "m32: false"
        * list/tuple: the first value should always be a string for the separator that will be inserted between the flag
          name and any values in this list, and all other values are string/int values that are possible options for this
          string. EG: setting the possible flags '-std=c++11', '-std=gnu++11', '-std=c++14', and '-std=gnu++14' would be:
          "std: ['=', 'c++11', 'gnu++11', 'c++14', 'gnu++14']", or using the flags '-m64' and '-m32' would be: 
          "m: ['', 32, 64]"
        
      All string values may also be "command strings" (See: 'Command Strings' section below). These strings may have
      substrings within them surrounded by '$$' (like latex), and those substrings will be evaluated using eval(). There
      are also some added classes for more functionality.

    - 'force_flags' [Optional, Inheritable]: flags that will be automatically set/forced for everything at/lower on the
      hierarchy. These will override flags with the same name in the `flags` key for all levels at/below it. This should
      be a dictionary with flag/value pairs within it:

      ```
      force_flags:
        flag1: value
        flag2: value
        ...
      ```
        
      EG: the 'gcc' compiler version 5 is only fully-compatible with c++ std versions c++98, c++11, c++14,
      gnu++98, gnu++11, gnu++14. So, one might wish to add the following to their compiler info file at the gcc 5.5.0
      version:

      ```
      force_flags:
        std: ['=', 'c++98', 'gnu++98', 'c++11', 'gnu++11', 'c++14', 'gnu++14']
        ...
      ```

      Which would force all arch's in that version to only allow a c++ std version of '-std=c++98', '-std=gnu++98'...

      This field is also inheritable, so those lower down the hierarchy will by default use the force_flags from the
      level above them. This field can also be overridden in that:

        * flags in multiple levels will be overridden in the lowest level. NOTE: setting a flag to null will remove
          that flag from force_flags in that level and those below it
        * flags in a lower level that are not in higher levels will be added to the force_flags for that level and
          those below it
        * flags in a higher level that are not in a lower level will be inherited
        
      EG: in the example above, let's say that the 'i686' architecture binary (for some reason) cannot use the gnu++
      std versions. One could then insert the force_flags in the 'i686' architecture:
        
      ```
      i686:
        force_flags:
            std: ['=', 'c++98', 'c++11', 'c++14']
        ...
      ```

      And that would still use all of the force_flags from the levels above it, but would override the 'std' flag
      for only the 'i686' architecture

##### Complete Compiler Info Structure
The compiler info object should look like:

```
# The name of a single compiler family/collection/suite. EG: 'GCC', 'Clang'
compiler_family:

    # Family-level data
    'use_previous': # [Optional]
    'container': name  # [Partially-Optional, Inheritable]
    'binary_name': name  # [Partially-Optional, Inheritable]
    'supported_languages': langauges  # [Partially-Optional, Inheritable]
    'force_flags':  # [Optional, Inheritable]
        ...
    'flags':  # [Optional, Inheritable]
        ...
    
    # All the compilers available to the current compiler family
    'compilers':

        compiler1:

            # Compiler-level data
            'use_previous': previous_value  # [Optional]
            'container': name  # [Partially-Optional, Inheritable]
            'binary_name': name  # [Partially-Optional, Inheritable]
            'supported_languages': langauges  # [Partially-Optional, Inheritable]
            'force_flags':  # [Optional, Inheritable]
                ...
            'flags':  # [Optional, Inheritable]
                ...
            
            # All the versions in this compiler
            'versions':

                version1:

                    # Version-level data
                    'use_previous': previous_value  # [Optional]
                    'container': name  # [Partially-Optional, Inheritable]
                    'binary_name': name  # [Partially-Optional, Inheritable]
                    'supported_languages': langauges  # [Partially-Optional, Inheritable]
                    'force_flags':  # [Optional, Inheritable]
                        ...
                    'flags':  # [Optional, Inheritable]
                        ...

                    # All of the architectures available in this version
                    'architectures':

                        arch1:

                            # Architecture-level data
                            'use_previous': previous_value  # [Optional]
                            'container': name  # [Partially-Optional, Inheritable]
                            'binary_name': name  # [Partially-Optional, Inheritable]
                            'supported_languages': langauges  # [Partially-Optional, Inheritable]
                            'force_flags':  # [Optional, Inheritable]
                                ...
                            'flags':  # [Optional, Inheritable]
                                ...
```

NOTE: versions (and only versions) can be left empty or set to null to automatically "use_previous: True" with no modifications

##### Variable Paths/Names
Values that are path/name strings ('container', 'binary_name') can use variables in their values for easy/repetative
naming. They should be inserted in strings as bracket variables that will be formatted with .format(). These names will
be inserted right at the end of loading. Kwargs are:

    - 'family'/'compiler'/'version'/'arch'/'architecture': insert the current name of this value. 'arch' and 'architecture'
      are the same. All of these values will be available since they will only be inserted right at the end, after 
      inheriting all the way down to the final level. 'version' will insert the raw version string
    - 'v': insert the cleaned version string (IE: what you would get by calling str(Version(version_str)))
    - 'vX': with `X` being an integer, insert the specified index of the version string. EG: "{v2}" with 
      version = Version("3.7.2.4-aaa") would insert the string '2', "{v0}" would be '3', etc. Will not insert the
      extra version string bits

NOTE: if you wish to actually enter brackets, you can just use "{{...}}" double brackets instead to escape checking
for string kwargs inside

##### Command Strings
String flags may also contain "command strings" within them surrounded by '\$\$' (like latex). The substrings within
the '\$\$'s will be evaluated using python's eval() function and thus must be valid python code.

WARNING: this will execute arbitrary python code, do not load compiler info files from untrusted sources.

These strings will be parsed into CFConcatenate() objects (See compiler_selection.cf_actions for more info)
that will concatenate all pieces of the strings with the evaluated values replacing the sections surrounded by '\$\$'.

EG: a flag like '-flag=valXX' where 'XX' can be any integer value between 0 and 99:

```flag: "=val$$CFRange(0, 100)$$"```

Would make a CFConcatenate() object like:

```CFConcatenate(['=val', CFRange(0, 100)])```

NOTE: if for some reason you wanted to make the literal string "\$\$", you could do something like: 
```$$'$'*2$$``` which would be parsed into the literal string "\$\$"

Currently available objects for command line flags (See compiler_selection.cf_actions for up-to-date objects):

    - CFAction(): the base class for all actions. Derivatives must override '__call__'. It is recommended that they also
      override '__hash__' for consistent use with other CAP tools. See the docs on the CFAction() object for more info

    - CFRandomNoFlag(flag_name: str): Randomly switches between a flag and its 'no-' version. The 'no-' version is the 
      same as the original value, just with the string 'no-' inserted after the first character. IE: 'finline' -> 'fno-inline'

    - CFConstant(const: Union[int, str]): Always return the given value (as a string). Must be a string or integer value.

    - CFChoice(choices: Iterable[Union[str, int, CFAction]]): Randomly choose uniformly from a list of values. Each value 
      can be either a str/int constant or another CFAction().
    
    - CFRange(start: int, end: int): Randomly choose an integer value in the given range from start (inclusive) to end (exclusive)

    - CFConcat(vals: Iterable[Union[str, int, CFAction]], sep: str = ''): Evaluates all values in list/tuple and 
      concatenates them (with optional separator)

Technically, since we call eval() on command strings, you can use whatever variables/imports/etc is available at the
time these objects are evaluated, but do so at your own risk.


# Poseidon

The /poseidon directory contains our current work towards extending the Triton binary analysis tool to implement
linux system emulation. It is still in development.

Triton: https://triton-library.github.io/

# Release

LLNL-CODE-837816