"""
An assortment of helper/utility functions for tokenization/normalization.
"""

import bincfg
import re
import traceback
import json
from functools import reduce


# The string/char that is inserted at the start and end of special tokens to signify them as special (and to make sure
#   they don't conflict with other actual tokens and whatnot)
SPECIAL_TOKEN_DESIGNATOR = '#'

# Token to insert at the start of each instruction for opcode-level tokenization
INSTRUCTION_START_TOKEN = '{0}start_instr{0}'.format(SPECIAL_TOKEN_DESIGNATOR)

# Token that is used to identify the start of a split immediate token
SPLIT_IMMEDIATE_TOKEN = '{0}split_imm{0}'.format(SPECIAL_TOKEN_DESIGNATOR)

# Constant string values for normalized tokens
STRING_LITERAL_STR = '{0}str{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
IMMEDIATE_VALUE_STR = '{0}immval{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
FUNCTION_CALL_STR = '{0}func{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
RECURSIVE_FUNCTION_CALL_STR = '{0}self{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
INTERNAL_FUNCTION_CALL_STR = '{0}innerfunc{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
EXTERNAL_FUNCTION_CALL_STR = '{0}externfunc{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
MULTI_FUNCTION_CALL_STR = '{0}multifunc{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
JUMP_DESTINATION_STR = '{0}jmpdst{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
MEMORY_EXPRESSION_STR = '{0}memexpr{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
GENERAL_REGISTER_STR = '{0}reg{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
SEGMENT_STR = '{0}seg{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
SEGMENT_ADDRESS_STR = '{0}segaddr{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
MEM_SIZE_TOKEN_STR = '{0}memptr{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
DISPLACEMENT_IMMEDIATE_STR = '{0}dispmem{0}'.format(SPECIAL_TOKEN_DESIGNATOR)
LARGE_BLOCK_STR = '{0}large_block{0}'.format(SPECIAL_TOKEN_DESIGNATOR)


# Some helpful regular expressions

# Possible immediate values: hexadecimal, octal, decimal (NOTE: the need to be processed in that order so decimal immediate
#    doesn't take up the initial '0' in front of hexadecimal/octal)
RE_IMM_HEX = r'-?0x[0-9a-f]+'
RE_IMM_OCT = r'-?0o[0-7]+'
RE_IMM_BIN = r'-?0b[01]+'
RE_IMM_INT = r'-?[0-9]+'
RE_IMMEDIATE = r'(?:{hex}|{oct}|{bin}|{int})'.format(hex=RE_IMM_HEX, oct=RE_IMM_OCT, int=RE_IMM_INT, bin=RE_IMM_BIN)

# Match string literals. Allows for strings starting/ending with either single or double quotes, and one can escape
#   quotes with \' or \", and can escape the escape with \\
# Also matches '#str#' as a string literal
RE_STRING_LITERAL = r'(?:"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')|{str}'.format(str=STRING_LITERAL_STR)

# Various symbol characters
RE_PLUS_SIGN = r'\+'
RE_TIMES_SIGN = r'\*'
RE_OPEN_BRACKET = r'\['
RE_CLOSE_BRACKET = r'\]'
RE_COLON = r':'

# Spacing and newline characters
RE_SPACING = r'[, \t.]+'
RE_NEWLINE = r'[|\n]'

# Extra information given by disassemblers
# We could just assume there will only ever be one set of disassembler information per line, and greedily match from
#   every open bracket '<' to the furthest end bracket '>', but I'm not sure this will always be the case.
# We first check for any string literals before looking at end brackets '>', in case those appear within the strings.
# So, we have the following restrictions on how disassembler information may be formatted:
#   1. All info is within '<>' brackets
#   2. You may nest '>' brackets within disassembler info, so long as they are always opened previously, and can go up to
#       a max depth of _DIS_INFO_MAX_REC_DEPTH
#   3. You may have any number of '<' brackets
#   4. Any '<>' brackets within well-formatted strings inside the disassembler info will have no effect as those are
#       treated as plain strings within the disassembler info
# This means the following disassembler infos are valid:
#   * <>
#   * <info>
#   * <lots of <<<<<<< things>
#   * <string="this is <> weird >>>>> string <<<<<">
#   * <nested <brackets> within data <are <ok> up to a certain <depth>>
# While the following are invalid and will break things:
#   * <
#   * <data>>
#   * <super<deep<nested<...<thing>>...>>
# NOTE: this recursive depth restriction exists because 1. I can't find any way of doing recursive regex's with python's
#   re module and 2. They'd probably be slower even if they did exist since we probably don't need that much depth
# NOTE: currently, the largest depth I've seen is 1 from something like: "invokespecial 0x0001<java/lang/object::<init>>"
DISINFO_START = '<'
DISINFO_END = '>'
_DIS_INFO_MAX_REC_DEPTH = 2  # The depth is 0 for no nested, +1 for each depth of nesting. EG: 2 would match up to "<0 <1 <2> 1> 0>"
RE_DISASSEMBLER_INFO = reduce(lambda rs, s: rs % s, [r'{ds}(?:{strlit}|%s|[^{de}])*{de}'] * _DIS_INFO_MAX_REC_DEPTH + [r'{ds}(?:{strlit}|[^{de}])*{de}'])\
    .format(strlit=RE_STRING_LITERAL, ds=DISINFO_START, de=DISINFO_END)


_RAISE_ERR = object()
def imm_to_int(token, on_err=_RAISE_ERR):
    """Convert the given value to integer
    
    If token is an integer, returns token. Otherwise, converts a string token to an integer, then back to a string, 
        accounting for hexadecimal, decimal, octal, and binary values

    Args:
        token (Union[str, int]): the immediate token to convert to integer
        on_err (Optional[Any]): if passed, then this value will be returned if there is an error while trying to parse
            the immediate value. Otherwise the error will just be raised like normal

    Returns:
        int: integer value of given token
    """
    try:
        return int(token, 0) if isinstance(token, str) else token
    except Exception:
        if on_err is _RAISE_ERR:
            raise
        return on_err


def scan_for_token(token_list, type=None, token=None, stop_on_type=None, stop_on_token=None, ignore_type=None, ignore_token=None, 
                   stop_unmatched=False, match_re=False, ignore_re_case=True, start=0, increment=1, wrap=True, max_matches=1, 
                   ret_list=False, ret='index', on_no_match=None):
    """Scans the given token list looking for a specific token(s) or token type(s)

    Will return None if no match is found.

    Detects tokens in the order:

        1. 'ignore' tokens
        2. 'stop' tokens
        3. accepted tokens (from `type` or `token` parameters)
    
    So, if one passes multiple parameters that conflict with one another, the above ordering is what takes precedent.
    
    Args:
        token_list (List[Tuple[str, str, ...]]): the list of tokens. Each element should be a tuple of (token_type, token, ...).
            The first element is the type of the token, second is the string token, and anything else is ignored. This means
            this function can work with either the 2-tuple token lists from Tokenizer() objects as well as the 3-tuple
            token lists from Normalizer() objects.
        type (Optional[Union[str, Iterable[str]]]): the type or types of tokens to return. Can be a string to only return one
            type of token, or an iterable of strings to return the first token found that has any of those types. If 
            `token` is not None, then the returned token must also match that argument.
            NOTE: you can match "not X" by using python re's negative lookahead: r'(?![X]).*', where '[X]' is the thing to not match
        token (Optional[Union[str, Iterable[str]]]): the token to return. Can be a string to only return one matching token, or
            an iterable of strings to return the first token found that matches any of those tokens. If `type` is None, then
            the returned token must also match that type.
            NOTE: you can match "not X" by using python re's negative lookahead: r'(?![X]).*', where '[X]' is the thing to not match
        stop_on_type (Optional[Union[str, Iterable[str]]]): if a token of this type is found, then we immediately stop
            searching and return whatever we currently have. Can be a string to only stop at one type of token, or an 
            iterable of strings to stop at the first token found that has any of those types.
            NOTE: you can match "not X" by using python re's negative lookahead: r'(?![X]).*', where '[X]' is the thing to not match
        stop_on_token (Optional[Union[str, Iterable[str]]]): if this token is found, then we immediately stop
            searching and return whatever we currently have. Can be a string to only stop at one token, or an 
            iterable of strings to stop at the first token found that matches any of these.
            NOTE: you can match "not X" by using python re's negative lookahead: r'(?![X]).*', where '[X]' is the thing to not match
        ignore_type (Optional[Union[str, Iterable[str]]]): ignores token types. Can be a string to only ignore one token type, or an 
            iterable of strings to ignore any token types that match any of these. These tokens will not be added to return
            lists or considered tokens to keep. Since these are checked before 'stop' token types, this will override
            the stopping on any tokens also matched with `stop_on_type`.
            NOTE: you can match "not X" by using python re's negative lookahead: r'(?![X]).*', where '[X]' is the thing to not match
        ignore_token (Optional[Union[str, Iterable[str]]]): ignores tokens. Can be a string to only ignore one token, or an 
            iterable of strings to ignore any tokens that match any of these. These tokens will not be added to return
            lists or considered tokens to keep. Since these are checked before 'stop' token types, this will override
            the stopping on any tokens also matched with `stop_on_token`.
            NOTE: you can match "not X" by using python re's negative lookahead: r'(?![X]).*', where '[X]' is the thing to not match
        stop_unmatched (bool): if True, will stop on the first unmatched token. IE: a token that was not ignored, was not
            already stopped on, and was not considered a token to keep
        match_re (bool): if True, will assume any match values in `type` or `token` are to be considered regular expressions to fullmatch()
        ignore_re_case (bool): if True, will pass re.IGNORECASE as a flag when making the regular expressions
        start (int): the index to start at within token_list
        increment (int): the increment to use when searching for tokens. Set to a negative number to move backwards through the list
            NOTE: if returning multiple values, they will be returned in the order they appear in the input list, regardless
            of the `increment` value
        wrap (bool): if True, then the initial `start` index will be wrapped to the length of the `token_list`. If False,
            then an initial `start` index that is out of bounds of the `token_list` will immediately stop.
        max_matches (Union[int, None]): the number of matches to find. If 1, then values will be returned as normal. If >1, then this will
            search through the list finding up to `max_matches` matching tokens and return their `ret` values as a list in the
            order that they were found. If None, then all matches found will be returned
            NOTE: if `max_matches` != 1, then the return value will always either be None if no matches were found, or a list (even if
            only one match was found)
        ret_list (bool): if True, will always return a list, even if only a single return value was present
        ret (Union[str, Iterable[str]]): what value(s) to return. Can be a single string to return a single value, or an 
            iterable of strings to return multiple values as a tuple in the order they were passed. Valid strings:

            - 'index': return the index in `token_list` of the matched token
            - 'type': return the token type of the matched token
            - 'token': return the string token that was matched
            - 'all': return all of the above. If in a passed list, ignores all other values in the list. Will return values
              in the order above.
        
        on_no_match (Optional[Any]): value to return if there were no matches found. Defaults to None
    
    Returns:
        Union[None, int, str, Tuple, List]: None if no match is found, or one of the return types designated by `ret` argument,
            or a tuple of multiple return values if user passed multiple values in `ret`, or a list of one of the previous
            if collecting matches for multiple tokens.
            NOTE: if returning multiple values, they will be returned in the order they appear in the input list, regardless
            of the `increment` value
    """
    # Compile into RE's if using
    re_flags = re.IGNORECASE if ignore_re_case else 0
    _mre = lambda val: [(re.compile(t, flags=re_flags) if match_re and not isinstance(t, re.Pattern) else t) for t in 
                        ([] if val is None else [val] if isinstance(val, str) else list(val))]
    type, token, stop_on_type, stop_on_token, ignore_type, ignore_token = \
        map(_mre, [type, token, stop_on_type, stop_on_token, ignore_type, ignore_token])

    # Make sure the user passed at least one type or token
    if len(type) == 0 and len(token) == 0:
        raise ValueError("Must pass at least one `type` or `token` to match to")
    
    # Make sure some integer values are good
    if increment == 0:
        raise ValueError("`increment` cannot be 0")
    max_matches = 2**100000 if max_matches is None else max_matches
    if max_matches < 1:
        raise ValueError("`num_matches` must be >= 1")
    
    # Make sure the user passed a valid return value
    ret = [ret] if isinstance(ret, str) else list(ret)
    ret = [r.lower() for r in ret]
    for i, r in enumerate(ret):
        if r in ['ind', 'idx', 'index', 'loc']:
            ret[i] = 0
        elif r in ['type', 'token_type']:
            ret[i] = 1
        elif r in ['token', 'token_string']:
            ret[i] = 2
        elif r in ['all']:
            ret = list(range(3))
            break
        else:
            raise ValueError("Unknown return type: %s" % repr(r))
    
    # If the token_list is empty, return None. Otherwise, insert the indices into the token_list at the beginning
    if len(token_list) == 0:
        return None
    token_list = [[i, t[0], t[1]] for i, t in enumerate(token_list)]
    
    # Function to check string matches depending on whether it is a regular expression or just normal string
    _str_match = lambda _token, _match: _match.fullmatch(_token) is not None if isinstance(_match, re.Pattern) else (_token == _match)
    
    # Iterate through the token_list finding all matches until reaching the end
    idx = (start % len(token_list)) if wrap else start
    ret_inds = []
    while 0 <= idx < len(token_list):
        # Add in the increment now to make the code nicer
        curr_idx = idx
        idx += increment

        # Check if this element is one that should be ignored, and skip it if so
        if any(any(_str_match(token_list[curr_idx][i + 1], t) for t in arr) for i, arr in enumerate([ignore_type, ignore_token])):
            continue
        # Check if this element matches a stop type or token, and break if so
        elif any(any(_str_match(token_list[curr_idx][i + 1], t) for t in arr) for i, arr in enumerate([stop_on_type, stop_on_token])):
            break
        # Check if this element doesn't match either type or token, and skip it if so (or, break if stop_unmatched=True)
        elif any(len(arr) > 0 and not any(_str_match(token_list[curr_idx][i + 1], t) for t in arr) for i, arr in enumerate([type, token])):
            if stop_unmatched:
                break
            continue

        # Otherwise, this matches! Add it into the list, and check if we have found our max number of matches
        ret_inds.append(curr_idx)
        if len(ret_inds) >= max_matches:
            break
    
    # Now that we have all of the matched indices, make the return objects
    _make_ret = lambda i: token_list[i][ret[0]] if len(ret) == 1 else tuple(token_list[i][r] for r in ret)
    _rev = lambda l: list(reversed(l)) if increment < 0 else l
    ret = on_no_match if len(ret_inds) == 0 else _make_ret(ret_inds[0]) if max_matches == 1 else _rev([_make_ret(i) for i in ret_inds])

    return ret if isinstance(ret, list) or not ret_list else [ret]


def get_normalizer(normalizer):
    """Returns the normalizer being used.

    Args:
        normalizer (Union[str, Normalizer, type]): either a ``Normalizer`` object (IE: has a callable 'normalize' function), 
            or a string name of a built-in normalizer to use, or a type of a normalizer to instantiate with no args/kwargs
            passed. Accepted strings include: 'innereye', 'deepbindiff', 'safe', 'deepsemantic', 'unnormalized', 
            'compressed_stats', 'hpc_data'

    Raises:
        ValueError: for unknown string name of normalizer
        TypeError: if `normalizer` was not a string or ``Normalizer`` object

    Returns:
        Normalizer: a ``Normalizer`` object
    """
    
    if isinstance(normalizer, str):
        norm_str = normalizer.lower()

        # You can specify the opcode/instruction level tokenization
        tl_names = bincfg.TokenizationLevel.AUTO.value + bincfg.TokenizationLevel.INSTRUCTION.value + bincfg.TokenizationLevel.OPCODE.value
        matched = re.fullmatch(r'(.*)[\-_](%s)' % '|'.join(tl_names), norm_str)
        if matched is not None:
            tl = 'op' if matched.groups()[1] in bincfg.TokenizationLevel.OPCODE.value else \
                'inst' if matched.groups()[1] in bincfg.TokenizationLevel.INSTRUCTION.value else 'auto'
            norm_str = matched.groups()[0]
        else:
            tl = 'auto'

        if norm_str.endswith("_normalizer"):
            norm_str, *_ = norm_str.rpartition("_normalizer")
        elif norm_str.endswith("_norm"):
            norm_str, *_ = norm_str.rpartition("_norm")

        known_isa = None
        for s in ['x86', 'java']:
            if norm_str.startswith(s):
                _, known_isa, norm_str = norm_str.partition(s)
        
        if norm_str.startswith('_'):
            norm_str = norm_str[1:]
        
        def _check_isa(needed, allow_none=False):
            if not allow_none and known_isa is None:
                raise ValueError("Ambiguous normalizer string: %s, must pass ISA name in front (IE: 'x86_%s')" % (repr(norm_str), norm_str))
            
            if isinstance(needed, str):
                needed = (needed,)
            
            if known_isa is not None and known_isa not in needed:
                raise ValueError("Normalizer %s can only be used with %s ISA's, not %s" % (repr(norm_str), repr(needed), repr(known_isa)))

        
        if norm_str in ['innereye', 'inner', 'innereyenormalizer']:
            _check_isa('x86', allow_none=True)
            return bincfg.normalization.X86InnerEyeNormalizer(tokenization_level=tl)
        elif norm_str in ['deepbindiff', 'bindiff', 'deepbin', 'deepbindiffnormalizer']:
            _check_isa('x86', allow_none=True)
            return bincfg.normalization.X86DeepBinDiffNormalizer(tokenization_level=tl)
        elif norm_str in ['safe', 'safenormalizer']:
            _check_isa('x86', allow_none=True)
            return bincfg.normalization.X86SafeNormalizer(tokenization_level=tl)
        elif norm_str in ['deepsem', 'deepsemantic', 'semantic', 'deepsemanticnormalizer']:
            _check_isa('x86', allow_none=True)
            return bincfg.normalization.X86DeepSemanticNormalizer(tokenization_level=tl)
        
        elif norm_str in ['none', 'unnorm', 'unnormalized', 'base', 'basenormalizer']:
            _check_isa(('x86', 'java'), allow_none=False)
            return bincfg.normalization.X86BaseNormalizer(tokenization_level=tl) if known_isa == 'x86' else\
                bincfg.normalization.JavaBaseNormalizer(tokenization_level=tl)
        elif norm_str in ['compressed', 'stats', 'comp_stats', 'compressed_stats', 'statistics', 'compressedstats', 'compressedstatsnormalizer']:
            _check_isa(('x86', 'java'), allow_none=False)
            if known_isa == 'java':
                raise NotImplementedError("Need to implement the java compressed stats normalizer")
            return bincfg.normalization.X86CompressedStatsNormalizer(tokenization_level=tl) if known_isa == 'x86' else\
                bincfg.normalization.JavaCompressedStatsNormalizer(tokenization_level=tl)
        elif norm_str in ['hpc', 'hpc_data', 'hpcdata', 'hpcdatanormalizer', 'hpcdatanorm', 'hpcnorm', 'hpcnormalizer']:
            _check_isa(('x86', 'java'), allow_none=False)
            if known_isa == 'java':
                raise NotImplementedError("Need to implement the java hpcdata normalizer")
            return bincfg.normalization.X86HPCDataNormalizer(tokenization_level=tl) if known_isa == 'x86' else\
                bincfg.normalization.JavaHPCDataNormalizer(tokenization_level=tl)
        else:
            raise ValueError("Unknown normalization string: '%s'" % normalizer)
    
    elif isinstance(normalizer, type):
        try:
            return get_normalizer(normalizer())
        except Exception as e:
            raise ValueError("Could not build a default normalizer from type: %s\nError Message: %s\n Traceback:%s"
                             % (repr(normalizer.__name__), e, traceback.format_exc()))
    
    elif hasattr(normalizer, 'normalize') and callable(normalizer.normalize):
        return normalizer
    
    else:
        raise TypeError("Unknown normalizer type: '%s'" % normalizer)


def parse_disinfo_json(string):
    """Attempts to pase a JSON object inside of disassembler info tokens
    
    Assumes the `DISINFO_START` and `DISINFO_END` have already been removed from the string.

    Args:
        string (str): the string to attempt to parse into json
    
    Returns:
        Union[None, JSONObject]: returns the resulting JSON object, or None if the string could not be parsed as JSON
    """
    try:
        return json.loads(string)
    except Exception:
        return None
