"""Contains the execution info for the current CAP execution

See the README.md, 'Configuration Files' subsection for more info
"""
from bincfg import X86HPCDataNormalizer, X86BaseNormalizer, JavaBaseNormalizer

# The unique identifier for this CAP execution
EXECUTION_UID = 'csharp'

# Postprocessing methods to apply
POSTPROCESSING = ['cfg']

# The columns to drop
DROP_COLUMNS = []

# The normalizer(s) to use
NORMALIZERS = []

# The analyzer(s) to use, should exist inside the container_info file
ANALYZERS = ['rose']

# The compile methods to use. See the README.md, 'Compile Method' section for more info
COMPILE_METHODS = [{
        'family': 'Dotnet',
        'compiler': 'dotnet',
        'version': 8,
        'arch': 'cil',
        'flags': [],
        'force_flags': {},
    },]

# Whether or not to have each process on the same machine wait for previous processes to load their data before loading
#   theirs to save memory
AWAIT_LOAD = False

# Debug number of files to work on. Leave None to do full CAP-ing
DEBUG_NUM_FILES = None