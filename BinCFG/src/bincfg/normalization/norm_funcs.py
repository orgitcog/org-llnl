from .base_tokenizer import Tokens
from .base_normalizer import DEFAULT_IMMEDIATE_THRESHOLD
from .norm_utils import imm_to_int, scan_for_token, DISPLACEMENT_IMMEDIATE_STR, IMMEDIATE_VALUE_STR, STRING_LITERAL_STR, \
    JUMP_DESTINATION_STR, MEMORY_EXPRESSION_STR, FUNCTION_CALL_STR, MULTI_FUNCTION_CALL_STR, RECURSIVE_FUNCTION_CALL_STR, \
    EXTERNAL_FUNCTION_CALL_STR, INTERNAL_FUNCTION_CALL_STR
from ..cfg.cfg_edge import EdgeType
import bincfg


def ignore(self, state):
    """Ignores information (if using for rose info, then it will also ignore negatives)"""
    pass


def identity(self, state):
    """Returns the original token"""
    return state.token


def return_token(self, state):
    """Returns the original token"""
    return state.token


def return_dispmem(self, state):
    """Replaces memory addressing displacement values with the string 'dispmem'"""
    return DISPLACEMENT_IMMEDIATE_STR


def replace_immediate(*args, include_negative=False):
    """Builds a function that replaces immediate values with the IMMEDIATE_VALUE_STR. 
    
    This will return a function to be called as a part of a normalizer. This function takes no arguments and only 1 keyword 
    argument: whether or not to include a negative sign '-' in front of the immediate string when the input is negative.

    NOTE: This is meant to be a higher-order function. But, just in case the user forgets that (or is too lazy to add in
    two extra characters to call this function), if you pass multiple args then it will be assumed this is being called 
    as if it is the _repl_func() function below and will simply return the default result

    Args:
        args: args for this function. Ideally empty
        include_negative (bool, optional): if True, will include a negative sign in front of the returned immediate 
            string when the input is negative. Defaults to False.

    Returns:
        Union[Callable[..., str], str]: either a function that will handle immediate strings (if this function was 
            called correctly), or a handled immediate string
    """
    def _ret_imm(self, state):
        return ('-' + IMMEDIATE_VALUE_STR) if include_negative and imm_to_int(state.token) < 0 else IMMEDIATE_VALUE_STR
    return _ret_imm if len(args) == 0 else _ret_imm(*args)


def replace_string_literal(*args, replace_previous_immediate=False):
    """Builds a function that replaces string literal values with the string 'str'
    
    This will return a function to be called as a part of a normalizer. This function takes no arguments and only 1 keyword 
    argument: whether to replace the previous immediate, or keep it and add in a 'str' string

    NOTE: This is meant to be a higher-order function. But, just in case the user forgets that (or is too lazy to add in
    two extra characters to call this function), if you pass multiple args then it will be assumed this is being called 
    as if it is the _repl_func() function below and will simply return the default result

    Args:
        args: args for this function. Ideally empty
        replace_previous_immediate (bool): if True, then any previous immediate value will be replaced with the 'str'
            string, otherwise the 'str' string will just be added

    Returns:
        Union[Callable[..., str], str]: either a function that will handle immediate strings (if this function was 
            called correctly), or a handled immediate string
    """
    def _ret_string_imm(self, state):
        # If we are not replacing any previous immediate, just return now
        if not replace_previous_immediate:
            return STRING_LITERAL_STR
        
        # Scan for a previous immediate token, and if not present, just return the string now
        idx = scan_for_token(state.line, type=Tokens.IMMEDIATE, ignore_type=[Tokens.SPACING], stop_unmatched=True, start=-1, increment=-1)
        if idx is None:
            return STRING_LITERAL_STR
        
        # We have a previous immediate value, replace it with our string, and return None to not insert token
        state.line = state.line[:idx] + [(Tokens.STRING_LITERAL, STRING_LITERAL_STR, state.orig_token)]
        return None
    return _ret_string_imm if len(args) == 0 else _ret_string_imm(*args)


def threshold_immediate(threshold=DEFAULT_IMMEDIATE_THRESHOLD, include_negative=False, imm_str=IMMEDIATE_VALUE_STR):
    """Builds a function that replaces immediate values with `immval` iff abs(immediate) > some threshold

    Args:
        threshold (int): the threshold to use. Any immediates whose absolute values are larger than this threshold
            will be replaced with the `imm_str`
        include_negative (bool): if True, then any immediate that are too large and get replaced will have a negative
            sign added to the front of the replacement string if the immediates were negative
        imm_str (str): the string to replace immediate values with

    Returns:
        Union[Callable[..., str], str]: either a function that will handle thresholded immediate strings (if this 
            function was called correctly), or a handled thresholded immediate string
    """
    if not isinstance(threshold, int):
        raise TypeError('Threshold must be int, instead got `%s`: %s' % (type(threshold).__name__, threshold))

    def _threshold(self, state):
        # If we have the same value as the imm_str, then assume we've already been normalized
        if state.token == IMMEDIATE_VALUE_STR:
            return IMMEDIATE_VALUE_STR
        
        val = imm_to_int(state.token)
        return str(val) if abs(val) <= threshold else ('-' + imm_str) if val < 0 and include_negative else imm_str

    return _threshold


def replace_jump_destination(self, state):
    """Replaces the jump destination immediate with 'jmpdst' iff the jump destination is an immediate value, not a segment address

    Args:
        idx (int): the index in ``line`` of the 'jump' opcode
        line (List[TokenTuple]): a list of (token_type, token) tuples. the current assembly line

    Returns:
        int: integer index in line of last handled token
    """
    state.line[state.token_idx + 1] = (state.line[state.token_idx + 1][0], JUMP_DESTINATION_STR if state.line[state.token_idx + 1][0] == Tokens.IMMEDIATE else \
                                        state.line[state.token_idx + 1][1], state.line[state.token_idx + 1][2])
    return state.token_idx + 2


def replace_memory_expression(*args):
    """Builds a function that replaces memory expressions with the given replacement string
    
    This will return a function to be called as a part of a normalizer. This only takes one argument: the replacement string.
    If no arguments are passed, then the replacement string will default to 'memexpr'

    NOTE: This is meant to be a higher-order function. But, just in case the user forgets that (or is too lazy to add in
    two extra characters to call this function), if you pass multiple args then it will be assumed this is being called 
    as if it is the _repl_func() function below and will simply return the default result

    Args:
        args: args for this function. Ideally either empty to use default memory expression string, or a string to replace
            all memory expressions with.

    Returns:
        Union[Callable[..., None], None]: either a function that will handle memory expressions (if this function was 
            called correctly), or a handled memory expression
    """
    replacement = args[0] if len(args) == 1 else MEMORY_EXPRESSION_STR

    if not isinstance(replacement, str):
        raise TypeError('Replacement must be str, instead got `%s`: %s' % (type(replacement).__name__, replacement))

    def repl_mem(self, state):
        """Replace memory expressions with 'memexpr'"""
        # Using a brand new token so it's not confused with anything else. Keep track of the old value as well
        state.line[state.memory_start] = (Tokens.MEMORY_EXPRESSION, replacement, ' '.join([l[2] for l in state.line[state.memory_start + 1:]]))
        del state.line[state.memory_start + 1:]  # Delete the rest of the line after memory_start index

    return repl_mem if len(args) <= 1 else repl_mem(*args)


def replace_function_call_immediate(*args):
    """Builds a function that replaces function call immediate values with the given replacement string
    
    This will return a function to be called as a part of a normalizer. This only takes one argument: the replacement string.
    If no arguments are passed, then the replacement string will default to 'func'

    NOTE: This is meant to be a higher-order function. But, just in case the user forgets that (or is too lazy to add in
    two extra characters to call this function), if you pass multiple args then it will be assumed this is being called 
    as if it is the _repl_func() function below and will simply return the default result

    Args:
        args: args for this function. Ideally either empty to use default function call string, or a string to replace
            all function callsa with.

    Returns:
        Union[Callable[..., None], None]: either a function that will handle function calls (if this function was 
            called correctly), or a handled function call
    """
    replacement = args[0] if len(args) == 1 else FUNCTION_CALL_STR

    if not isinstance(replacement, str):
        raise TypeError('Replacement must be str, instead got `%s`: %s' % (type(replacement).__name__, replacement))

    def _repl_func(self, state):
        state.line[state.token_idx + 1] = (Tokens.IMMEDIATE, replacement, state.line[state.token_idx + 1][2])
        return state.token_idx + 2

    return _repl_func if len(args) <= 1 else _repl_func(*args)


def special_function_call(self, state, ret_only_call_type=False):
    """Handles special function calls
    
    Special external functions have their name kept. Recursive calls are replaced with 'self', other internal function 
    calls are replaced with 'internfunc', other external function calls are replaced with 'externfunc'. If a block has
    multiple function calls out, then it will be replaced with 'multifunc'.

    NOTE: This can all only happen if cfg and block information is passed. If it is not passed, then all function
    calls will be replaced with 'func'

    Args:
        idx (int): the index in ``line`` of the 'call' opcode
        line (List[TokenTuple]): a list of (token_type, token) tuples. the current assembly line
        special_functions (Set[str]): a set of string special function names.
        cfg (Union[CFG, MemCFG], optional): either a ``CFG`` or ``MemCFG`` object that these lines occur in. Used for 
            determining function calls to self, internal functions, and external functions. If not passed, then these 
            will not be used. Defaults to None.
        block (Union[CFGBasicBlock, int], optional): either a ``CFGBasicBlock`` or integer block_idx in a ``MemCFG`` 
            object. Used for determining function calls to self, internal functions, and external functions. If not 
            passed, then these will not be used. Defaults to None.
        ret_only_call_type (bool): if True, will return only the call type being used as a string. This is only for testing
            purposes and should likely not be used in normalization as this function already can handle the normalizing.
            This will return a string if it is not a special function call (for the appropriate function call type), or
            a tuple with one element for a special function call (the name of the special function). 

    Returns:
        int: integer index in line of last handled token
    """
    idx, line, block, cfg = state.token_idx, state.line, state.block, state.cfg

    # If cfg is None, then we cannot determine intern/extern/self function calls. Just return 'func' for function call
    if cfg is None:
        line[idx + 1] = (Tokens.IMMEDIATE, FUNCTION_CALL_STR, line[idx + 1][2])
        return idx + 2

    # Check that we even need to replace the value. We should only replace immediate values, so check to make sure
    #   the old_token starts with a digit
    if not ret_only_call_type and line[idx + 1][2][0] not in "0123456789":
        return idx + 2

    # Find the function call info in cfg
    if isinstance(cfg, bincfg.MemCFG):
        # Find the outgoing edges of this block. Normally the first edge is the function call edge, however we still need
        #   to check because there can be times where it only contains the normal return edge as the function call
        #   address couldn't be resolved to a known basic block
        func_call_block_inds, edge_types = cfg.get_block_edges_out(block, ret_edge_types=True)

        # Make sure the first value is in fact a function call
        if len(edge_types) > 0 and edge_types[0] == EdgeType.FUNCTION_CALL.value:

            # Check for a self call by comparing the two block's function indices
            func_call_block_idx = func_call_block_inds[0]
            self_call = cfg.block_func_idx[block] == cfg.block_func_idx[func_call_block_idx]
            extern_func_name = cfg.get_block_function_name(func_call_block_idx) if cfg.is_block_extern_function(func_call_block_idx) else None

            # Check for a multi-function call
            multi_call = len(edge_types[edge_types == EdgeType.FUNCTION_CALL.value]) > 1
        
        # Otherwise, we have no clue where the function call goes to, treat it as just some innerfunc
        else:
            self_call = False
            extern_func_name = None
            multi_call = False

    # Otherwise this is a plain CFG. Get the call address from the next immediate value, and look up its info in the CFG
    elif isinstance(cfg, bincfg.CFG):
        fc_out = block.get_sorted_edges(edge_types='function', direction='out')[0]

        if len(fc_out) > 1:
            multi_call = True
        else:
            multi_call = False

            # Need to check that the disassembler was able to figure out the call location
            if len(fc_out) == 0:
                self_call = False
                extern_func_name = None
            else:
                func = cfg.get_block(fc_out[0].to_block).parent_function
                self_call = func.address == block.parent_function.address
                extern_func_name = func.name if func.is_extern_function else None
    
    else:
        raise TypeError("Unknown cfg type for special_function_call normalization: %s" % repr(type(cfg)))

    # Determine what function call strings to return
    # A multi-function call
    if multi_call:
        if ret_only_call_type: return MULTI_FUNCTION_CALL_STR
        line[idx + 1] = (line[idx + 1][0], MULTI_FUNCTION_CALL_STR, line[idx + 1][2])

    # A recursive function call
    elif self_call:
        if ret_only_call_type: return RECURSIVE_FUNCTION_CALL_STR
        line[idx + 1] = (line[idx + 1][0], RECURSIVE_FUNCTION_CALL_STR, line[idx + 1][2])

    # An external function call
    elif extern_func_name is not None:
        extern_func_name = extern_func_name.split('@')[0]
        if extern_func_name in self.special_functions:
            if ret_only_call_type: return (extern_func_name,)
            line[idx + 1] = (line[idx + 1][0], '{' + extern_func_name + '}', line[idx + 1][2])
        else:
            if ret_only_call_type: return EXTERNAL_FUNCTION_CALL_STR
            line[idx + 1] = (line[idx + 1][0], EXTERNAL_FUNCTION_CALL_STR, line[idx + 1][2])

    # An internal function call
    else:
        if ret_only_call_type: return INTERNAL_FUNCTION_CALL_STR
        line[idx + 1] = (line[idx + 1][0], INTERNAL_FUNCTION_CALL_STR, line[idx + 1][2])

    return idx + 2
