"""Class for tokenizing assembly lines, as well as other tokenization constants"""

import re
from ..utils import ParameterSaver, eq_obj, hash_obj
from .norm_utils import *
from enum import Enum
from ..utils.type_utils import *


# Token type names
class Tokens:
    INSTRUCTION_ADDRESS = 'inst_addr'
    INSTRUCTION_START = 'inst_start'
    SPLIT_IMMEDIATE = 'split_imm'
    DISASSEMBLER_INFO = 'disassembler_info'
    NEWLINE = 'newline'
    SPACING = 'spacing'

    OPEN_BRACKET = 'open_bracket'
    CLOSE_BRACKET = 'close_bracket'
    PLUS_SIGN = 'plus_sign'
    TIMES_SIGN = 'times_sign'
    COLON = 'colon'

    INSTRUCTION_PREFIX = 'prefix'
    OPCODE = 'opcode'
    REGISTER = 'register'
    IMMEDIATE = 'immediate'

    MEMORY_SIZE = 'memory_size'
    MEMORY_EXPRESSION = 'memory_expression'
    BRANCH_PREDICTION = 'branch_prediction'
    STRING_LITERAL = 'string_literal'
    SEGMENT_ADDRESS = 'segment_address'

    MISMATCH = 'mismatch'


class TokenizationLevel(Enum):
    """Different levels to perform tokenization"""
    OPCODE = ['op', 'opcode', 'operand', 'opcodes', 'operands']
    INSTRUCTION = ['inst', 'instruction', 'line', 'instructions', 'lines']
    AUTO = ['auto', 'automatic', 'default']


class Architectures(Enum):
    """Known (but not necessarily supported) architectures"""
    X86 = ['x86', 'i686', 'x86_64']
    JAVA = ['java', 'java_bytecode']


def get_architecture(arch: 'Union[str, Architectures]') -> 'Architectures':
    """Returns the architecture
    
    Args:
        arch (Union[str, Architectures])
    """
    if isinstance(arch, Architectures):
        return arch
    elif isinstance(arch, str):
        arch = arch.lower().replace('-', '_')
        for a in Architectures:
            if any(arch == v for v in a.value):
                return a
        raise ValueError("Unknown architecture string: %s" % repr(arch))
    else:
        raise TypeError("Cannot get architecture from object of type: %s" % repr(type(arch).__name__))


class TokenMismatchError(Exception):
    pass

class UnknownTokenError(Exception):
    pass


# Special tokens to insert into tokenizer by default
SPECIAL_TOKENS_START = [
    (Tokens.STRING_LITERAL, RE_STRING_LITERAL),
    (Tokens.DISASSEMBLER_INFO, RE_DISASSEMBLER_INFO),
    (Tokens.INSTRUCTION_START, INSTRUCTION_START_TOKEN),
    (Tokens.SPLIT_IMMEDIATE, SPLIT_IMMEDIATE_TOKEN),
    (Tokens.PLUS_SIGN, RE_PLUS_SIGN),
    (Tokens.TIMES_SIGN, RE_TIMES_SIGN),
    (Tokens.OPEN_BRACKET, RE_OPEN_BRACKET),
    (Tokens.CLOSE_BRACKET, RE_CLOSE_BRACKET),
    (Tokens.COLON, RE_COLON),
    (Tokens.SPACING, RE_SPACING),
    (Tokens.NEWLINE, RE_NEWLINE),
    (Tokens.IMMEDIATE, RE_IMMEDIATE),
]
SPECIAL_TOKENS_END = [
    (Tokens.MISMATCH, r'.'),
]


# An indication that the default newline tuple should be used when tokenizing/normalizing
_USE_DEFAULT_NT = object()


def parse_tokenization_level(tokenization_level, auto_tl):
    """Returns the bincfg.TokenizationLevel enum based on the given tokenization_level.

    Args:
        tokenization_level (Union[bincfg.TokenizationLevel, str]): either a string tokenization level, or a class from the 
            bincfg.TokenizationLevels enum
        auto_tl (bincfg.TokenizationLevel): the default tokenization level to use if we get an 'auto' tokenization level

    Returns:
        bincfg.TokenizationLevel: a class from the ``bincfg.TokenizationLevels`` enum
    """
    if not isinstance(auto_tl, TokenizationLevel) or auto_tl == TokenizationLevel.AUTO:
        raise TypeError("`auto_tl` must be a bincfg.TokenizationLevel, and cannot be bincfg.TokenizationLevel.AUTO. Got: %s" % repr(auto_tl))
    
    if isinstance(tokenization_level, str):
        tl = tokenization_level.lower().replace('-', '_')
        for l in TokenizationLevel:
            if tl in l.value:
                ret = l
                break
        else:
            raise ValueError("Unknown tokenization_level string: '%s'" % tokenization_level)
    elif isinstance(tokenization_level, TokenizationLevel):
        ret = tokenization_level
    else:
        raise TypeError("Unknown tokenization_level type: '%s'" % type(tokenization_level))
    
    # Check for auto
    if ret is TokenizationLevel.AUTO:
        return auto_tl
    
    return ret


class BaseTokenizer(metaclass=ParameterSaver):
    """A default class to tokenize instructions

    Should be subclassed once for each instruction set, providing the tokens being used.

    Many functions may be overriden to change tokenization behavior. These functions all start with the name `token\_...`
    and take as input a single state dictionary and return either a string for the next token to append to the current
    line being tokenized, or None to not add anything to the line. The state dictionary contains the following:

        - 'tokenizer' (BaseTokenizer): this tokenizer 
        - 'kwargs' (Dict[str, Any]): dictionary of extra kwargs passed to the initial call to the `tokenize` function
        - 'all_strings' (List[str]): list of input strings (args) passed to the initial call to the `tokenize` function
        - 'token_handlers' (Dict[str, Callable[]]): dictionary mapping token types to the function that handles that token
        - 'sentence' (List[Tuple[str, str]]): list of processed token tuples to return, each a tuple of (token\_name, token)
        - 'newline_tup' (Union[None, Tuple[str, str]]): token tuple to add at the end of each line to indicate a new line
        - 'match_instruction_address' (bool): whether or not we are matching instruction addresses
        - 'split_imm' (bool): whether or not we are currently handling an immediate token that was split 
        - 'line' (List[Tuple[str, str]]): the current line of tokens we are working on
        - 'string' (str): the current string being tokenized
        - 'token_type' (str): the type of the 'token', should be from `bincfg.normalization.base_tokenizer.Tokens`
        - 'token' (str): the currently matched token string
        - 'match' (re.Match): the re match object that matched this token
    
    Some extra functions are available for overriding including:

        - handle_line(): called at the end of each line being tokenized (an individual string passed to the tokenizer)
        - handle_sentence(): called at the end of each sentence being tokenized (aggregation of all lines passed to the tokenizer)
    
    Each instruction set architecture (ISA) should have its own ``Tokenizer`` class that inherits from ``BaseTokenizer``. The 
    tokenization process uses python's ``re`` module to perform tokenization, converting strings into streams of 
    (token\_name, token\_string) tuples. For more information on how to use regex to create tokenizers, see: 
    https://docs.python.org/3/library/re.html#writing-a-tokenizer
    
    TOKENIZATION PROCESS
    
        1. Clean the incomming instruction strings using the passed `clean_instruction_func`
        2. Iterate through the strings finding all tokens
        
           a. Each token is sent to its corresponding token handler function
           b. At the end of each 'line' (EG: end of a passed `string`, reaching Tokens.NEWLINE token, etc.), that line
              is handled with the `handle_line()` function
           c. All tokens are added to the same return 'sentence', even if multiple strings in `strings` were passed

        3. After all strings have been tokenized and lines handled, the final return 'sentence' is sent to `handle_sentence()`
    
    SPECIAL TOKENS

    There are some 'special tokens' that are assumed to exist for all ISA's as they are a part of the tokenization
    process itself. These tokens will be inserted into the passed `tokens` parameter at the beginning of the list
    (IE: they are the first tokens searched for), except for the 'mismatch' token which is inserted at the end,
    and are inserted in the following order:

        1. String literals (Tokens.STRING_LITERAL) - matches strings which can start/end with matching single or double 
           quotes, and can escape inner quotes with \\' or \\", and can escape the escape character with \\\\. Any extra
           escape characters (not behind a ' or " or \\) will be left as-is.
        2. Disassembler information (Tokens.DISASSEMBLER_INFO) - matches disassembler information of the form "<...>". 
           This info must be within open/close angle brackets. It is also possible to nest angle brackets within the 
           disassembler info up to a maximum current depth of 3. IE: we can match the following:
        
            * "<no angle brackets inside>" - depth of 1
            * "<angle <brackets> depth <2>>" - depth of 2
            * "<level <3 angle <bracket>> depth>" - depth of 3
        
           We also do not check that every open has a matching close, just that every close has a matching open. So, the
           following could still be matched:

            * "<lots of <<<<<<< things>"
        
           However, missing or unmatched ending angle brackets will fail, as well as very deep nesting:

            - "<" : no matching '>' only for the first occurance of '<'
            - "<data>>" : no matching '<' for both of the '>' brackets
            - "<super<deep<nested<...<thing>>...>>" : too large nesting depth

           String literals are checked first within the disassembler info so that any end brackets '>' within the strings
           won't affect the parsing of the disassembler info.
        
           This limit on nesting depth is present due to the inability for python's re engine to handle recursive matching
           of nested brackets, and I can't think of any way to implement it entirely within re's (which is needed in order
           to continue using the python re tokenization method). I don't see any reason why this would be needed as we
           already go down to a depth of 3 to handle more than what I would expect as output from disassemblers, and if
           the user is inserting information themselves, they could simply input the information within the brackets
           using a different delimiter and parse it themselves by overriding things like `token_disassembler_info()`
           and `handle_disassembler_info` in the `Tokenizer` and `Normalizer` classes respectively. If a larger
           depth is needed, one can manually alter the `_DIS_INFO_MAX_REC_DEPTH` variable at the top of this file. It will
           increase the valid nesting depth at the cost of slower regular expression matching for disassembler info.
        3. Instruction start token "#start_instr#" (Tokens.INSTRUCTION_START) - used to determine when instructions 
           start/stop when using an op-level tokenization scheme. When tokenizing, we need to know when a new instruction 
           is started to decide if an immediate value found should be considered an instruction address or just a plain 
           immediate. New instructions occur whenever we reach a newline token, an instruction start token, or the start 
           of a new string passed in the args of the `tokenize()` method. This instruction start token is removed when 
           found, and won't appear during normalization.
        4. Split immediate token "#split_imm#" (Tokens.SPLIT_IMMEDIATE) - used to designate a split immediate value. This 
           is useful for reducing the number of unique tokens present while keeping full immediate information. When using 
           split immediates during normalization, immediate values with more digits than some threshold will be split into multiple
           immediate tokens and placed one after the other, prepended with this "#split_imm#" token. In order to keep
           that output as renormalizable, the tokenizer, when finding one of these split immediate tokens, will concatenate
           all of the following immediate tokens until reaching some non-immediate (and, non-spacing) token to rebuild
           the original immediate token. This split immediate token is removed when found, and won't appear during normalization
        5. Plus sign (Tokens.PLUS_SIGN) - '+'
        6. Times sign (Tokens.TIMES_SIGN) - '*'
        7. Open bracket (Tokens.OPEN_BRACKET) - '['
        8. Close bracket (Tokens.CLOSE_BRACKET) - ']'
        9. Colon (Tokens.COLON) - ':'
        10. Spacing (Tokens.SPACING) - One or more space ' ', comma ',', or tab '\\t' characters in a row
        11. Newline (Tokens.NEWLINE) - Either the newline character '\\n' or a pipe character '|'
        12. Immediate values (Tokens.IMMEDIATE) - any integer immediate value in hex, decimal, octal, or binary. Hex values
            must start with '0x', octal with '0o', and binary with '0b'
        13. Mismatch token (Tokens.MISMATCH) - matches any character. Inserted at the very end of `tokens` and is used 
            to designate the start of an unknown token or character so that can be handled (by default, an error is raised)

    If you wish to keep some of the above tokens, but overwrite others, you can set that token's regex in the passed
    `tokens` parameter, and that will overwrite these special tokens. You may also set it to None to not insert it at all.

    INSTRUCTION ADDRESSES

    If match_instruction_address=True when tokenizing, the tokenizer will attempt to match instruction addresses at the
    beginning of each line. If there is an immediate value at the start of a line (IE: start of a string in `strings`, 
    or immediately after a Tokens.NEWLINE or Tokens.INSTRUCTION_START [ignoring any Tokens.SPACING]), then that token
    will be converted into a Tokens.INSTRUCTION_ADDRESS token. If there is a Tokens.COLON immediately after that token 
    (again, ignoring any Tokens.SPACING), then that first Tokens.COLON match will be appended to that Tokens.INSTRUCTION_ADDRESS 
    token, removing any Tokens.SPACING inbetween them. For example, using the x86 tokenization scheme:

        - "0x1234: add rax rax" -> [(Tokens.INSTRUCTION_ADDRESS, '0x1234:'), ...]
        - "  0x1234     : add rax rax" -> [(Tokens.SPACING, '  '), (Tokens.INSTRUCTION_ADDRESS, '0x1234:'), ...]
        - "0x1234 add rax rax" -> [(Tokens.INSTRUCTION_ADDRESS, '0x1234'), ...]
    
           
    Parameters
    ----------
    tokens: `Optional[List[Tuple[str, str]]]`
        the tokens to use. Should be a list of 2-tuples. Each tuple is a pair of (name, regex) where
        name is the string name of the token, and regex is a regular expression to find that token. These
        tuples should be ordered in the preferred order to search for tokens. If None, then this will default to 
        self.DEFAULT_TOKENS (which should be set when defining the class)
    token_handlers: `Optional[Dict[str, Callable[[Dict[str, Any]], Union[None, str]]]]`
        optional dictionary mapping token type strings to functions to handle those token types when tokenizing. This is
        intended to be used when you wish to add entirely new token types not present in `bincfg.normalization.base_tokenizer.Tokens`.
        If you wish to change the behavior of handling an already-present token type, just override that token handler function.
        These will override the default token handlers.
    insert_special_tokens: `bool`
        by default, some special tokens will be inserted at the front of `tokens` (see the 'special tokens' listed above).
        If you wish to stop this from happening, you can set `insert_special_tokens` to False
    case_sensitive: `bool`
        If True, then regular expressions will be matched exactly as they appear. If False, then the re.IGNORECASE flag
        will be passed when compiling the regular expressions
    """

    DEFAULT_NEWLINE_TUPLE = (Tokens.NEWLINE, '\n')
    """The default (token_type, token) tuple to use for newlines"""

    ARCHITECTURE = None
    """The architecture this tokenizer works on"""

    def __init__(self, tokens=None, token_handlers=None, insert_special_tokens=True, case_sensitive=False):
        self.tokens = tokens if tokens is not None else self.DEFAULT_TOKENS
        self.case_sensitive = case_sensitive

        # Insert the special tokens, make sure user values override regex's and positions of default tokens, and remove
        #   any tokens that the user has set to None
        if insert_special_tokens:
            user_tokens = set(t[0] for t in self.tokens)
            self.tokens = [t for t in SPECIAL_TOKENS_START if t[0] not in user_tokens] + self.tokens + \
                [t for t in SPECIAL_TOKENS_END if t[0] not in user_tokens]
            self.tokens = [t for t in self.tokens if t[1] is not None]

        self.token_handlers = token_handlers if token_handlers is not None else {}
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Initializes the tokenizer from self.tokens"""
        flags = (re.M|re.UNICODE)
        flags = (flags|re.IGNORECASE) if not self.case_sensitive else flags
        self.tokenizer = re.compile('|'.join([("(?P<%s>%s)" % pair) for pair in self.tokens]), flags=flags)
    
    def tokenize(self, *strings, newline_tup=_USE_DEFAULT_NT, match_instruction_address=True, **kwargs):
        """Tokenizes the input
        
        Subclasses should override any self.token_* methods they wish to inject behavior into. Each one of those functions
        takes in a 'state' dictionary as input and should return either a new string token or None to use the old token.

        See the docs for :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info on how tokenization
        works, how to create subclasses, etc.

        Args:
            strings (str): arbitrary number of strings to tokenize.
            newline_tup (Optional[Tuple[str, str]]): the tuple to insert inbetween each passed string, or None to not 
                insert anything. Defaults to `self.__class__.DEFAULT_NEWLINE_TUPLE`.
            match_instruction_address (bool, optional): if True, will match instruction addresses. If there is an immediate
                value at the start of a line (IE: start of a string in `strings`, or immediately after a Tokens.NEWLINE
                or Tokens.INSTRUCTION_START [ignoring any Tokens.SPACING]), then that token will be converted into a
                Tokens.INSTRUCTION_ADDRESS token. If there is a Tokens.COLON immediately after that token (again, ignoring
                any Tokens.SPACING), then that first Tokens.COLON match will be appended to that Tokens.INSTRUCTION_ADDRESS 
                token, removing any Tokens.SPACING inbetween them. For example, using the x86 tokenization scheme:

                    - "0x1234: add rax rax" -> [(Tokens.INSTRUCTION_ADDRESS, '0x1234:'), ...]
                    - "  0x1234     : add rax rax" -> [(Tokens.SPACING, '  '), (Tokens.INSTRUCTION_ADDRESS, '0x1234:'), ...]
                    - "0x1234 add rax rax" -> [(Tokens.INSTRUCTION_ADDRESS, '0x1234'), ...]
                
            kwargs (Any): extra kwargs to store in the tokenizer state, for use in child classes

        Returns:
            List[Tuple[str, str]]: list of (token_type, token) tuples
        """
        if len(strings) == 0:
            return []
        
        # Replace the newline_tup if using
        newline_tup = self.DEFAULT_NEWLINE_TUPLE if newline_tup is _USE_DEFAULT_NT else newline_tup
        
        # If the first element starts with an `INSTRUCTION_START` token, then concatenate all of the strings together
        if strings[0].strip().startswith(INSTRUCTION_START_TOKEN):
            strings = [' '.join(strings)]
        
        # Dictionary mapping token types to self functions to handle those tokens. Only have to add tokens to handle
        token_handler_dict = {
            Tokens.INSTRUCTION_ADDRESS: self.token_instruction_address,
            #Tokens.INSTRUCTION_START: None,
            #Tokens.SPLIT_IMMEDIATE: None,
            Tokens.DISASSEMBLER_INFO: self.token_disassembler_info,
            Tokens.NEWLINE: self.token_newline,
            Tokens.SPACING: self.token_spacing,

            Tokens.OPEN_BRACKET: self.token_all_symbols,
            Tokens.CLOSE_BRACKET: self.token_all_symbols,
            Tokens.PLUS_SIGN: self.token_all_symbols,
            Tokens.TIMES_SIGN: self.token_all_symbols,
            Tokens.COLON: self.token_all_symbols,

            Tokens.INSTRUCTION_PREFIX: self.token_instruction_prefix,
            Tokens.OPCODE: self.token_opcode,
            Tokens.REGISTER: self.token_register,
            Tokens.IMMEDIATE: self.token_immediate,

            Tokens.MEMORY_SIZE: self.token_memory_size,
            #Tokens.MEMORY_EXPRESSION: 'memory_expression'
            Tokens.BRANCH_PREDICTION: self.token_branch_prediction,
            Tokens.STRING_LITERAL: self.token_string_literal,

            Tokens.MISMATCH: self.token_mismatch,
        }
        token_handler_dict.update(self.token_handlers)
        
        token_state = {'kwargs': kwargs, 'all_strings': strings, 'token_handlers': token_handler_dict, 'sentence': [], 
                       'newline_tup': newline_tup, 'match_instruction_address': match_instruction_address, 'tokenizer': self}
        
        for string in strings:
            token_state.update({'previous_newline': True, 'split_imm': False, 'line': [], 'string': string, 'matched_ia_colon': False})

            for mo in self.tokenizer.finditer(string):
                token_state['token_type'], token_state['token'], token_state['match'] = mo.lastgroup, mo.group(), mo
                
                # Handle all the various tokens. Keep all tokens
                new_token = token_state['token_handlers'][token_state['token_type']](token_state) if token_state['token_type'] in token_state['token_handlers'] \
                    else token_state['token'] if token_state['token_type'] in [Tokens.SPLIT_IMMEDIATE, Tokens.INSTRUCTION_START] \
                    else self.token_unknown(token_state)
                        
                if new_token is not None:
                    token_state['line'].append((token_state['token_type'], new_token))
            
            # If there are split immediates, parse them out now in line
            token_state['line'] = self._merge_split_immediates(token_state['line'])

            # Check for instruction addresses
            if token_state['match_instruction_address']:
                token_state['line'] = self._check_instruction_address(token_state['line'])

            # Handle the new line and add on the newline_tup if using, and we don't have an empty sentence
            new_line = self.handle_line(token_state)
            token_state['sentence'] += new_line if new_line is not None else token_state['line']
            if token_state['newline_tup'] is not None and len(token_state['sentence']) > 0:
                token_state['sentence'].append(token_state['newline_tup'])
        
        return self.handle_sentence(token_state)
    
    def _merge_split_immediates(self, line):
        """Merges any split immediates into one, ignoring any spacing inbetween them"""
        ret_line, idx = [], 0
        while idx < len(line):
            # If this isn't a split immediate, just go ahead to the next token
            if line[idx][0] not in [Tokens.SPLIT_IMMEDIATE]:
                ret_line.append(line[idx])
                idx += 1
                continue

            # This is a split immediate token, scan ahead for all immediate tokens (including instruction address), 
            #   ignoring spacing, until reaching a non-immediate token. Concatenate these into a single immediate value
            imm_inds = scan_for_token(line, type=Tokens.IMMEDIATE, ignore_type=Tokens.SPACING, stop_unmatched=True,
                                      start=idx + 1, max_matches=None, wrap=False, on_no_match=[])
            if len(imm_inds) > 0:
                ret_line.append((line[imm_inds[0]][0], ''.join([line[i][1] for i in imm_inds])))
            idx = max(imm_inds + [idx]) + 1
        
        return ret_line
    
    def _check_instruction_address(self, line):
        """Scans through the current line checking for possible instruction address locations"""
        _newlines = [Tokens.NEWLINE, Tokens.INSTRUCTION_START]
        ret_line, idx = [], 0
        while idx < len(line):
            token_tup, token_type = line[idx], line[idx][0]

            # If we are at the start of a line, check for instruction address
            if idx == 0 or token_type in _newlines:
                inst_addr_inds = scan_for_token(line, type=Tokens.IMMEDIATE, ignore_type=Tokens.SPACING, stop_unmatched=True, 
                                                start=(idx+1) if token_type in _newlines else idx, wrap=False, on_no_match=[], ret_list=True)
                
                if len(inst_addr_inds) > 0 and imm_to_int(line[inst_addr_inds[0]][1]) > 0:
                    # Check for possible colon token
                    inst_addr_inds += scan_for_token(line, type=Tokens.COLON, ignore_type=Tokens.SPACING, stop_unmatched=True, 
                                                     start=max(inst_addr_inds)+1, wrap=False, on_no_match=[], ret_list=True)
                    
                    # Build the instruction address token. Insert any tokens before the start if present
                    for i in range(idx if token_type in _newlines + [Tokens.SPACING] else (idx + 1), min(inst_addr_inds)):
                        ret_line.append(line[i])
                    token_tup = (Tokens.INSTRUCTION_ADDRESS, ''.join([line[i][1] for i in inst_addr_inds]))
                    idx = max(inst_addr_inds)

            # Add in this token and increment the index
            ret_line.append(token_tup)
            idx += 1

        return ret_line
    
    def handle_line(self, state):
        """Handles a single line (one string passed to the tokenizer)
        
        Each line could contain newlines and whatnot, but no newline_tup's will have been inserted.

        Subclasses may override this function for more behavior, but it defaults to just returning the passed line.

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            List[Tuple[str, str]]: list of (token_type, token) tuples for this line
        """
        return state['line']
    
    def handle_sentence(self, state):
        """Handles an entire sentence (aggregation of all strings passed to one call of this tokenizer)
        
        Inbetween each line, a newline_tup will have already been inserted (if using)

        Subclasses may override this function for more behavior, but it defaults to just returning the passed sentence

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            List[Tuple[str, str]]: the final list of tokens
        """
        return state['sentence']

    def token_branch_prediction(self, state):
        """Handles any branch_prediction tokens
        
        This can be overriden by subclasses for more functionality, but defaults to just returning the original token

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            Union[str, None]: either a string token for the next token to append to line, or None to not append anything
        """
        return state['token']

    def token_immediate(self, state):
        """Handles any immediate tokens
        
        This can be overriden by subclasses for more functionality, but defaults to just returning the original token

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            Union[str, None]: either a string token for the next token to append to line, or None to not append anything
        """
        return state['token']

    def token_instruction_prefix(self, state):
        """Handles any instruction_prefix tokens
        
        This can be overriden by subclasses for more functionality, but defaults to just returning the original token

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            Union[str, None]: either a string token for the next token to append to line, or None to not append anything
        """
        return state['token']

    def token_memory_size(self, state):
        """Handles any memory_size tokens
        
        This can be overriden by subclasses for more functionality, but defaults to just returning the original token

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            Union[str, None]: either a string token for the next token to append to line, or None to not append anything
        """
        return state['token']

    def token_opcode(self, state):
        """Handles any opcode tokens
        
        This can be overriden by subclasses for more functionality, but defaults to just returning the original token

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            Union[str, None]: either a string token for the next token to append to line, or None to not append anything
        """
        return state['token']

    def token_register(self, state):
        """Handles any register tokens
        
        This can be overriden by subclasses for more functionality, but defaults to just returning the original token

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            Union[str, None]: either a string token for the next token to append to line, or None to not append anything
        """
        return state['token']

    def token_disassembler_info(self, state):
        """Handles any disassembler information tokens
        
        This can be overriden by subclasses for more functionality, but defaults to just returning the original token

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            Union[str, None]: either a string token for the next token to append to line, or None to not append anything
        """
        return state['token']
    
    def token_string_literal(self, state):
        """Handles any string literals
        
        This can be overriden by subclasses for more functionality, but defaults to just returning the original token

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            Union[str, None]: either a string token for the next token to append to line, or None to not append anything
        """
        return state['token']
    
    def token_spacing(self, state):
        """Handles any spacing tokens
        
        This can be overriden by subclasses for more functionality, but defaults to just returning the original token

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            Union[str, None]: either a string token for the next token to append to line, or None to not append anything
        """
        return state['token']
    
    def token_newline(self, state):
        """Handles any newline tokens
        
        This can be overriden by subclasses for more functionality, but defaults to just returning the original token

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            Union[str, None]: either a string token for the next token to append to line, or None to not append anything
        """
        return state['token']
    
    def token_instruction_address(self, state):
        """Handles any instruction address tokens
        
        This can be overriden by subclasses for more functionality, but defaults to just returning the original token

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            Union[str, None]: either a string token for the next token to append to line, or None to not append anything
        """
        return state['token']
    
    def token_all_symbols(self, state):
        """Handles all symbol tokens ('+', '*', '[', ']', ':')
        
        This can be overriden by subclasses for more functionality, but defaults to just returning the original token, 
        except for colons ':', for which we check if the previous non-spacing token was an immediate value. If so, and
        `match_instruction_address` is True, then we append any inbetween spacing and the colon to that immediate and 
        replace its type with Token.INSTRUCTION_ADDRESS.

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info
        
        Returns:
            Union[str, None]: either a string token for the next token to append to line, or None to not append anything
        """
        return state['token']
    
    def token_mismatch(self, state):
        """What to do when there is a token mismatch in a string
        
        This can be overriden by subclasses for more functionality, bet defaults to raising a ``TokenMismatchError`` with 
        info on the mismatch

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info

        Raises:
            TokenMismatchError: by default
        """
        raise TokenMismatchError("Mismatched token '%s' at index %d in string: %s" % (state['token'], state['match'].start(), repr(state['string'])))
    
    def token_unknown(self, state):
        """What to do when there is a token type that we don't know how to handle
        
        This can be overriden by subclasses for more functionality, bet defaults to raising a ``UnknownTokenError`` with 
        info on the unknown token

        Args:
            state (Dict): dictionary of current state. See :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info

        Raises:
            UnknownTokenError: by default
        """
        raise UnknownTokenError("Unknown token type %s, token: %s" % (repr(state['token_type']), repr(state['token'])))

    def __call__(self, *strings, newline_tup=_USE_DEFAULT_NT, match_instruction_address=True, **kwargs):
        """Tokenizes the input
        
        Subclasses should override any self.token_* methods they wish to inject behavior into. Each one of those functions
        takes in a 'state' dictionary as input and should return either a new string token or None to use the old token.

        See the docs for :func:`~bincfg.normalization.base_tokenizer.BaseTokenizer` for more info on how tokenization
        works, how to create subclasses, etc.

        Args:
            strings (str): arbitrary number of strings to tokenize.
            newline_tup (Optional[Tuple[str, str]]): the tuple to insert inbetween each passed string, or None to not 
                insert anything. Defaults to `self.__class__.DEFAULT_NEWLINE_TUPLE`.
            match_instruction_address (bool, optional): if True, will match instruction addresses. If there is an immediate
                value at the start of a line (IE: start of a string in `strings`, or immediately after a Tokens.NEWLINE
                or Tokens.INSTRUCTION_START [ignoring any Tokens.SPACING]), then that token will be converted into a
                Tokens.INSTRUCTION_ADDRESS token. If there is a Tokens.COLON immediately after that token (again, ignoring
                any Tokens.SPACING), then that first Tokens.COLON match will be appended (along with any inbetween Tokens.SPACING)
                to that Tokens.INSTRUCTION_ADDRESS token. For example, using the x86 tokenization scheme:

                    - "0x1234: add rax rax" -> [(Tokens.INSTRUCTION_ADDRESS, '0x1234:'), ...]
                    - "  0x1234     : add rax rax" -> [(Tokens.SPACING, '  '), (Tokens.INSTRUCTION_ADDRESS, '0x1234     :'), ...]
                    - "0x1234 add rax rax" -> [(Tokens.INSTRUCTION_ADDRESS, '0x1234'), ...]
                
            kwargs (Any): extra kwargs to store in the tokenizer state, for use in child classes

        Returns:
            List[Tuple[str, str]]: list of (token_type, token) tuples
        """
        return self.tokenize(*strings, newline_tup=newline_tup, match_instruction_address=match_instruction_address, **kwargs)
    
    def __repr__(self) -> str:
        if eq_obj(self.tokens, self.DEFAULT_TOKENS):
            tokens_str = ''
        elif len(repr(self.tokens)) > 30:
            tokens_str = '...'
        else:
            tokens_str = 'tokens=' + repr(self.tokens)
        return self.__class__.__name__ + "(%s)" % tokens_str
    
    def __str__(self) -> str:
        return self.__class__.__name__
    
    def __eq__(self, other):
        return type(self) == type(other) and eq_obj(self.tokens, other.tokens) and self.case_sensitive == other.case_sensitive \
            and eq_obj(self.token_handlers, other.token_handlers)
    
    def __hash__(self):
        return hash_obj([type(self).__name__, self.tokens, self.case_sensitive, self.token_handlers], return_int=True)

