"""Things to help us auto detect files/project inputs"""

import os
import re
import string


# A file which must exist within a directory in order to do a project compilation
# List of tuples of (name, files_list), where files_list is a list of filenames that, if any of them are present, designate
#   the project type
# Project types will be checked in the order they appear here
_KNOWN_PROJECT_FILES = [
    ('cmake', ['CMakeLists.txt']),
    ('cap', ['CAP.json', 'cap.json'])
]

# File extensions used by tabular data formats
# List of tuples of (name, extenions), where extensions is a list of valid file extensions for the given format, including the '.'
# Extensions will be checked in the order they appear here
_KNOWN_TABULAR_FILE_EXTENSIONS = [
    ('parquet', ['.pq', '.parquet']),
    ('csv', ['.csv']),
]

# Magic bytes used by tabular data formats
# List of tuples of (name, magic_lists), where magic_lists is a list of possible bytes magic starting the file
# Magics will be checked in the order they appear here
_KNOWN_TABULAR_MAGIC_BYTES = [
    ('parquet', [b"PAR1"]),
]

# Magic bytes used by binary data formats
# List of tuples of (name, magic_lists), where magic_lists is a list of possible bytes magic starting the file
# Magics will be checked in the order they appear here
_KNOWN_BINARY_MAGIC_BYTES = [
    ('elf', ['\x7FELF']),
    ('pe', [b'MZ']),  # Also handles dll's
    ('macho', [b'\xFE\xED\xFA\xCE', b'\xFE\xED\xFA\xCF', b'\xCE\xFA\xED\xFE', b'\xCF\xFA\xED\xFE']),
    ('java', [b'\xCA\xFE\xBA\xBE']),  # Technically, this may conflict with Apple Fat Binary, but idk if anyone uses that...
]
_MAX_MAGIC_BYTES_SIZE = max(len(e) for _, l in _KNOWN_BINARY_MAGIC_BYTES for e in l) + 2

# File extensions used by source code files
# List of tuples of (name, extenions), where extensions is a list of valid file extensions for the given format, including the '.'
# Extensions will be checked in the order they appear here
_KNOWN_SOURCE_FILE_EXTENSIONS = [
    ('c', ['.c']),
    ('c++', ['.C', '.cc', '.cpp', '.CPP', '.c++', '.C++', '.cp', '.cxx']),
    ('java', ['.java']),
    ('cs', ['.cs']),
]

# A regular expression that matches one or more contiguous whitespace characters, non-capturing
_SPACING_RE = r'(?:[{w}]+)'.format(w=string.whitespace)

# A valid csharp name identifier
_CS_NAME = r'[a-zA-Z_][a-zA-Z_0-9]+'

# Substrings we expect to find within certain source code files, and (hopefully) nowhere else
# List of tuples of (name, reg_list), where reg_list is a list of regular expressions to search for, with the re.DOTALL flag on
# Sources will be checked in the order they appear here
_KNOWN_SOURCE_SUBSTRINGS = [
    # Any (executable) java program should have a valid main entrypoint something like "public static void main(String[]..."
    ('java', [r'public{sp}static{sp}void{sp}main{sp}?\({sp}?String{sp}?\[{sp}?\]'.format(sp=_SPACING_RE)]),
    
    # I think any c# program should look like:
    # namespace [NAME] {
    #   ... class [NAME] {
    #    ...   static ... Main (...
    ('c#', [r'namespace{sp}{cn}{sp}?\{{[^{{]*class{sp}{cn}{sp}?\{{[^{{]*Main{sp}?\('.format(sp=_SPACING_RE, cn=_CS_NAME)]),
]


def get_known_project_type(input_path):
    """Returns the project type, or None if input_path is not a directory or project type can't be inferred
    
    Checks the directory for a known filename that is expected depending on what type of project build method we
    will be using. See _KNOWN_PROJECT_FILES for currently supported files/project build types
    """
    if not os.path.isdir(input_path):
        return None
    
    for k, filenames in _KNOWN_PROJECT_FILES:
        if any(f in filenames for f in os.listdir(input_path)):
            return k
        
    return None


def get_known_tabular_type(input_path):
    """Returns the tabular data file format type, or None if input_path isn't a file or tabular type can't be inferred

    First checks if the file ends with a known extension, then checks for various magic bytes designating the file.
    See _KNOWN_TABULAR_FILE_EXTENSIONS and _KNOWN_TABULAR_MAGIC_BYTES for supported values
    """
    if not os.path.isfile(input_path):
        return None
    
    for k, l in _KNOWN_TABULAR_FILE_EXTENSIONS:
        if any(input_path.endswith(e) for e in l):
            return k
    
    for k, ls in _KNOWN_TABULAR_MAGIC_BYTES:
        for magic in ls:
            with open(input_path, 'rb') as f:
                if f.read(len(magic)) == magic:
                    return k
    
    return None


def get_known_compiled_type(input):
    """Returns the compiled binary file format type, or None if input_path isn't a file or binary type can't be inferred
    
    Checks files for magic bytes. See _KNOWN_BINARY_MAGIC_BYTES for supported values

    Args:
        input (Union[str, bytes]): either a string path to a file to check, or bytes of a binary to check
    """
    if isinstance(input, str):
        if not os.path.isfile(input):
            return None
        with open(input, 'rb') as f:
            input = f.read(_MAX_MAGIC_BYTES_SIZE)
    
    for k, l in _KNOWN_BINARY_MAGIC_BYTES:
        for magic in l:
            if input[:len(magic)] == magic:
                return k
    
    return None


def get_known_source_type(input_path):
    """Returns the source code file type, or None if input_path isn't a file or source code type can't be inferred
    
    Assumes all source code is text. Checks first for file extensions, then substrings to determine type. See extensions
    and substrings in _KNOWN_SOURCE_FILE_EXTENSIONS and _KNOWN_SOURCE_SUBSTRINGS respectively
    """
    if not os.path.isfile(input_path):
        return None
    
    for k, l in _KNOWN_SOURCE_FILE_EXTENSIONS:
        if any(input_path.endswith(e) for e in l):
            return k
    
    with open(input_path, 'r') as f:
        source = f.read()
    for k, l in _KNOWN_SOURCE_SUBSTRINGS:
        if any(re.search(r, source, flags=re.DOTALL) for r in l):
            return k
    
    return None


def auto_detect_file_task(filepath):
    """Automatically detect the cap task for the given file"""
    # Check for known tabular data file formats
    s = get_known_tabular_type(filepath)
    if s is not None:
        return 'tabular-%s' % s
    
    # Check for known compiled binaries
    s = get_known_compiled_type(filepath)
    if s is not None:
        return 'binary-%s' % s
    
    # Check for known source code files
    s = get_known_source_type(filepath)
    if s is not None:
        return 'source-%s' % s

    # Otherwise we have no idea what this task is, return None
    return None


def auto_detect_folder_task(folder_path):
    """Automatically detect the cap task for the given folder (not partitioned)"""
    # Otherwise if this has a known project file in it
    s = get_known_project_type(folder_path)
    if s is not None:
        return 'project-%s' % s
    
    # Otherwise just assume this is a misc CAP process
    return 'misc'


def auto_detect_cap_task(paths):
    """Automaticaly detect the cap task based on the given paths"""
    paths = {'input': paths} if isinstance(paths, str) else paths
    
    # If this is a directory, check for partitioned, folder, or misc tasks
    if os.path.isdir(paths['input']):
        
        # Check if the given/default partitioned paths exist. If so, use them
        if 'partitioned' in paths and os.path.exists(paths['partitioned']) and os.path.exists(paths['partitioned_info']):
            return 'partitioned'
        
        # Otherwise, return the auto detected folder task
        return auto_detect_folder_task(paths['input'])
        
    # Otherwise assume it is a file. Return the auto detected cap file task
    return auto_detect_file_task(paths['input'])
