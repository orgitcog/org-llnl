import re
from .base_normalizer import BaseNormalizer
from .base_tokenizer import BaseTokenizer, TokenizationLevel, Tokens, Architectures, TokenMismatchError, UnknownTokenError, get_architecture
from .normalize import normalize_cfg_data, get_normalizer
from .x86 import *
from .java import *
from .multi_normalizer import MultiNormalizer


# Regular expression matching all known return instructions, hopefully without interference
RETURN_INSTRUCTION_RE = re.compile(r'[ \t\n]*(?:(?:{x86})|(?:{java})).*'.format(
    x86=r'(?:lock )*(?:ret[nf]?)',  # Have to allow for lock instructions, as well as retn (return near) and retf (return far)
    java=r'ret|[ilfda]?return',  # 'ret' is deprecated since Java 7 or something, but we should leave it here for backwards compatibility
))
