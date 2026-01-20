import re
import bincfg
from .x86_tokenizer import Tokens, X86_REGISTER_SIZES, X86_MEMORY_SIZES
from ..norm_utils import imm_to_int, IMMEDIATE_VALUE_STR, GENERAL_REGISTER_STR, MEM_SIZE_TOKEN_STR, JUMP_DESTINATION_STR, \
    MEMORY_EXPRESSION_STR, FUNCTION_CALL_STR, MULTI_FUNCTION_CALL_STR, RECURSIVE_FUNCTION_CALL_STR, EXTERNAL_FUNCTION_CALL_STR, \
    INTERNAL_FUNCTION_CALL_STR, DISPLACEMENT_IMMEDIATE_STR, STRING_LITERAL_STR, RE_SPACING


# Regex's to check if registers are general or not, and to remove number information.
RE_GENERAL_REGISTER_MATCH = re.compile(r'r[0-9]+[dwb]?|[re]?[abcd]x|[abcd][lh]|[re]?[sd]il?', flags=re.IGNORECASE)
RE_REMOVE_REGISTER_NUMBER = re.compile(r'\(?[0-9]+\)?')

# Handling memory size information
MEM_SIZE_RE = re.compile(r'(?:v([0-9]+))?([a-z]+)(?:{space}ptr)?'.format(space=RE_SPACING), flags=re.IGNORECASE)


def x86_clean_nop(state):
    """Cleans any line with the opcode 'nop' to only contain the opcode

    Args:
        idx (int): the index in ``line`` of the 'nop' opcode
        line (List[TokenTuple]): a list of (token_type, token) tuples. the current assembly line
        args: unused
        kwargs: unused

    Returns:
        int: integer index in line of last handled token
    """
    old_line = ' '.join([l[2] for l in state.line])
    state.line.clear()
    state.line.append((Tokens.OPCODE, 'nop', old_line))
    return 1


def x86_replace_general_register(self, state):
    """Replaces general registers with a default string and their size, keeping special registers the same (while removing their numbers)

    Args:
        token (str): the current string token

    Returns:
        str: normalized name of register
    """
    return (GENERAL_REGISTER_STR + str(X86_REGISTER_SIZES[state.token.lower()])) if RE_GENERAL_REGISTER_MATCH.fullmatch(state.token) is not None \
        else RE_REMOVE_REGISTER_NUMBER.sub('', state.token)


def x86_memsize_value(self, state):
    """Replaces memory size pointers with 'memsize' followed by the value of that memsize in bytes

    Args:
        token (str): the current string token

    Returns:
        str: normalized memory size string
    """
    vsize, mem_str = MEM_SIZE_RE.fullmatch(state.token).groups()
    mem_size = X86_MEMORY_SIZES[mem_str.lower()] * (1 if vsize is None else int(vsize))
    return MEM_SIZE_TOKEN_STR + str(mem_size)
