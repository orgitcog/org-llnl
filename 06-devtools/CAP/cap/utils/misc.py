"""
Miscellaneous utility functions
"""

import re
import os
import string
import traceback
import pandas as pd


# The maximum number of string characters to keep from stdout/stderr from subprocesses
MAX_STD_STR_LEN = 100_000


def get_language_family(language):
    """Returns the language family. Matches pretty generously

    Args:
        language (str): the string programming_language being used
    """
    if any(s in language.lower() for s in ['c++', 'clang++']):
        return 'c++'
    elif re.fullmatch(r'.*C[0-9]+', language) or language.lower() in ['c', 'gnu c']:
        return 'c'
    elif 'java' in language.lower():
        return 'java'
    elif 'c#' in language.lower():
        return 'c#'
    else:
        raise ValueError("Could not parse language: '%s'" % language)


JAVA_CLASSNAME_REGEX = re.compile(r'public{spacing}(?:final{spacing})?class{spacing}([^{bracket} \t\n]*)(?:{spacing}|{bracket})'.format(spacing=r'(?: |\t|\n)+', bracket='{'))
JAVA_RESERVED_KEYWORDS = ['abstract','continue','for','new','switch','assert','default','if','package','synchronized',
                          'boolean','do','goto','private','this','break','double','implements','protected','throw','byte',
                          'else','import','public','throws','case','enum','instanceof','return','transient','catch','extends',
                          'int','short','try','char','final','interface','static','void','class','finally','long','strictfp',
                          'volatile','const','float','native','super','while','true','false','null']
def get_java_classname(source):
    """Returns the name of the first public class found in the given java source"""
    matches = JAVA_CLASSNAME_REGEX.findall(source)
    
    if matches is None or len(matches) == 0:
        raise ValueError("Could not find a match to get the java classname for source file:\n%s" % source)

    group = matches[0]
    if group[0] in "0123456789":
        raise ValueError("Java public class name starts with a digit: %s" % repr(group))
    elif group in JAVA_RESERVED_KEYWORDS:
        raise ValueError("Java public class name was a reserved keyword: %s" % repr(group))
    else:
        for c in group:
            if ord(c) < 128 and c not in string.ascii_letters and c not in string.digits and c not in ['_', '$']:
                raise ValueError("Java public class name contains an invalid character %s: %s" % (c, group))
            elif ord(c) >= 128:
                raise ValueError("Java public class name contains a unicode character that might not be able to be a part of a filename %s: %s" % (c, group))

    return group


_PART_FILE_RE = r'[0-9]+-[0-9]+[.](?:pq|parquet)'
def get_partitioned_source(part_id, part_dir):
    """Returns the source for the given partition_id"""
    files = [tuple(map(int, f.split('.')[0].split('-'))) + (os.path.join(part_dir, f), ) for f in os.listdir(part_dir) if re.fullmatch(_PART_FILE_RE, f)]
    for start, end, f in files:
        if start <= part_id < end:
            chunk = pd.read_parquet(f, columns=['id', 'source'])
            found = chunk[chunk['id'] == part_id]
            if len(found) == 0:
                raise ValueError("Could not find part_id '%d' within partitioned file: %s" % (part_id, repr(f)))
            elif len(found) > 1:
                raise ValueError("Found multiple rows for part_id '%d' within partitioned file: %s" % (part_id, repr(f)))
            return found['source'].iloc[0]
    raise ValueError("Could not find partitioned file containing part_id '%d' in directory: %s" % (part_id, repr(part_dir)))


def get_symlink_binds(bind_cmd, path):
    """Returns a string of space-separated bind commands, one for each symlink that should be bound within the container"""
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = [os.path.join(dirname, f) for dirname, dirs, files in os.walk(path) for f in files]
    else:
        raise NotImplementedError("Unknown file path type for path: %s" % repr(path))
    
    return ' '.join([(bind_cmd.format(host_path=os.path.realpath(p), container_path=os.path.realpath(p))) for p in set(files) if os.path.islink(p)])


class CriticalError(Exception):
    """An error that, should it occurr within an ExceptionLogger, will not be caught. Wrap around the actual error"""
    def __init__(self, error):
        super().__init__('Critical error, unrecoverable:\n%s: %s\nTraceback: %s' % (type(error).__name__, str(error), traceback.format_exc()))
