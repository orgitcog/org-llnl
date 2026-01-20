"""
Utils for cleaning source code before attempting to compile it, or after errors occur
"""

import re
from utils.misc import JAVA_CLASSNAME_REGEX


# You can find this list by doing "javac -X" and looking at the '-Xlint:' keys
JAVA_SUPPRESS_WARNING_STRING = '@SuppressWarnings({"auxiliaryclass", "cast", "classfile", "deprecation", "dep-ann", "divzero", "empty", "exports", "fallthrough", "finally", "module", "opens", "options", "overloads", "overrides", "path", "processing", "rawtypes", "removal", "requires-automatic", "requires-transitive-automatic", "serial", "static", "try", "unchecked", "varargs", "preview"})'
JAVA_REPLACE_SUPPRESS_WARNINGS = re.compile(r'@SuppressWarnings\(".*"\)')

def clean_source(source, language_family):
    """Cleans the given source code
    
    Currently:

        - removes all "#pragma GCC optimize" commands
        - inserts removes all "@SuppressWarnings" annotations and inserts a "@SuppressWarnings(...)" annotation before
          the main class that suppresses all known warnings

    NOTE: Will remove them from strings and whatnot, I don't care enough to try and prevent that because it will
    probably never happen
    
    Args:
        source (str): string source code
        language_family (str): the language family being used
    
    Returns:
        str: the cleaned source code
    """
    if language_family in ['c', 'c++']:
        return re.sub(r'#pragma GCC .*\n', '', source)
    elif language_family in ['java']:
        return JAVA_CLASSNAME_REGEX.sub(lambda match: JAVA_SUPPRESS_WARNING_STRING + "\n" + match.group(0), 
                                        JAVA_REPLACE_SUPPRESS_WARNINGS.sub('', source), count=1)
    elif language_family in ['c#']:
        # Should do a better check here to see if there's anything bad that can happen
        return source
    else:
        raise NotImplementedError(language_family)

