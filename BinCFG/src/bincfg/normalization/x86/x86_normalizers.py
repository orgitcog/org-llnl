"""
A bunch of builtin normalization methods based on literature.

NOTE: some of these are slightly modified from their original papers either for code purposes, or because we are using
decompiled binaries instead of compiled assembly and thus lose out on some information (EG: symbol information
for jump instructions)
"""
from ..base_normalizer import TokenizationLevel, BaseNormalizer, LIBC_FUNCTION_NAMES
from .x86_tokenizer import X86_DEFAULT_TOKENIZER, X86_RE_SEGMENT_REG
from .x86_norm_funcs import *
from ..norm_utils import SPLIT_IMMEDIATE_TOKEN, scan_for_token, RE_SPACING
from ..norm_funcs import *
from ...utils import eq_obj, hash_obj


# Used to replace '{spacing}ptr' in a memory size expression
REPL_SPACE_PTR = re.compile(r'{space}ptr'.format(space=RE_SPACING), flags=re.IGNORECASE)


class X86BaseNormalizer(BaseNormalizer):
    """Base class for x86 normalizers.
    
    Performs an 'unnormalized' normalization, removing what is likely extraneous information, and providing a base class
    for other x86 normalization methods to inherit from.
    
    Parameters
    ----------
    tokenizer: `Tokenizer`
        the tokenizer to use
    token_handlers: `Optional[Dict[str, Callable[[NormalizerState], Union[None, str]]]]`
        optional dictionary mapping string token types to functions to handle those tokens. These will override any
        token handlers that are used by default (IE: all of the `self.handle_*` functions). Functions should take one
        arg (the current normalizer state) as input and return either the next string token to add to the current line,
        or None to not add anything. This is useful for adding more methods to handle new token types that are not builtin.
    token_sep: `Optional[str]`
        the string to use to separate each token in returned instruction lines. Only used if tokenization_level is 
        'instruction'. If None, then a default value will be used (' ' for unnormalized using BaseNormalizer(), '_' 
        for everything else)
    tokenization_level: `Optional[Union[TokenizationLevel, str]]`
        the tokenization level to use for return values. Can be a string, or a ``TokenizationLevel`` type. Strings can be:

            - 'op': tokenized at the opcode/operand level. Will insert a 'INSTRUCTION_START' token at the beginning of
              each instruction line
            - 'inst'/'instruction': tokenized at the instruction level. All tokens in each instruction line are joined
              together using token_sep to construct the final token
            - 'auto': pick the default value for this normalization technique

    anonymize_tokens: `bool`
        if True, then tokens will be annonymized by taking their 4-byte shake_128 hash. Why does this exist? Bureaucracy.
    """

    DEFAULT_TOKENIZATION_LEVEL = TokenizationLevel.INSTRUCTION
    """The default tokenization level used for this normalizer"""

    renormalizable = True
    """Whether or not this normalization method can be renormalized later by other normalization methods"""

    tokenizer = None
    """The tokenizer used for this normalizer"""

    token_sep = None
    """The separator string used for this normalizer. Will default to ' '"""

    tokenization_level: TokenizationLevel
    """The tokenization level to use for this normalizer"""

    def __init__(self, tokenizer=None, token_handlers=None, token_sep=None, tokenization_level=TokenizationLevel.AUTO, anonymize_tokens=False):
        # Add the handle_segment_address() function
        token_handlers = {} if token_handlers is None else token_handlers
        token_handlers.setdefault(Tokens.SEGMENT_ADDRESS, self.handle_segment_address)

        super().__init__(tokenizer=tokenizer if tokenizer is not None else X86_DEFAULT_TOKENIZER, token_handlers=token_handlers,
                         token_sep=token_sep if token_sep is not None else ' ',
                         tokenization_level=tokenization_level, anonymize_tokens=anonymize_tokens)
        self.register_opcode_handler(r'(?:far)?callq?', self.opcode_function_call)
        self.register_opcode_handler(r'j.*|farjmp', self.opcode_jump)
        self.register_opcode_handler(r'nop.*', x86_clean_nop)
    
    def opcode_function_call(self, state):
        """Handles function call opcodes, defaults to doing nothing"""
        return None
    
    def opcode_jump(self, state):
        """Handles jump opcodes, defaults to doing nothing"""
        return None
    
    def handle_all_symbols(self, state):
        """Handles all symbols. We use this to keep track of when memory expressions start/end"""
        # Check for memory address brackets
        if state.token_type in [Tokens.OPEN_BRACKET]:
            state.set(memory_start=len(state.line))
        elif state.token_type in [Tokens.CLOSE_BRACKET]:
            state.line.append((state.token_type, state.token, state.token))
            self.handle_memory_expression(state)
            state.set(memory_start=None)
            return None  # Don't append the token since we've already done that

        # Check for segment addresses before memory expressions
        elif state.token_type in [Tokens.COLON]:
            idx = scan_for_token(state.line, type=Tokens.REGISTER, token=X86_RE_SEGMENT_REG, ignore_type=[Tokens.SPACING], 
                                 stop_unmatched=True, match_re=True, start=-1, increment=-1)
            if idx is not None:
                state.line, state.token_type, state.token, state.orig_token = state.line[:idx], Tokens.SEGMENT_ADDRESS, \
                    state.line[idx][1] + state.token, ''.join([t[2] for t in state.line[idx:]]) + state.orig_token
                return self._handle_token(state, insert_token=False).token
        
        return state.token
    
    def handle_segment_address(self, state):
        """Handles a segment address. Defaults to returning the original token

        Should return either the token to add to the current line, or None to not add any token

        Args:
            state (NormalizerState): dictionary of current state information. See ``bincfg.normalization.base_normalizer.NormalizerState``
        """
        return state.token
    
    def handle_memory_size(self, state):
        """Handles a memory size. Removes any '{spacing}ptr' where {spacing} is any amount of spacing

        Should return either the token to add to the current line, or None to not add any token

        Args:
            state (NormalizerState): dictionary of current state information. See ``bincfg.normalization.base_normalizer.NormalizerState``
        """
        return REPL_SPACE_PTR.sub('', state.token)
    
    def handle_memory_expression(self, state):
        """Handles memory expressions. Splits values up into 'base', 'index', 'scale', and 'displacement'

        Values:

            - 'base' (B): a register holding the starting base location. Handled by self.handle_memory_base()
            - 'index' (I): a register that is added to a base register. Handled by self.handle_memory_index()
            - 'scale' (S): an immediate value that is multiplied onto 'index', should only be 1, 2, 4, or 8. Handled by self.handle_memory_scale()
            - 'displacement' (D): and immediate value that acts as a displacement to the memory address (or in some cases,
              the literal memory address itself). Handled by self.handle_memory_displacement()
        
        Acceptable x86 memory addressing modes:

            * [D]
            * [B]
            * [B + I]
            * [B + D]
            * [B + I + D]
            * [B + I*S]
            * [I*S + D]
            * [B + I*S + D]
        
        NOTE: You can have other formats that don't fit known addressing modes, but the values might not be handled
        properly. Specifically, the first register found will be considered the 'base', and all subsequent are considered
        'index', etc.
        
        The respective handle_memory_*() methods will be called with the same parameters as this function, but 'memory_start'
        will instead be the starting index of that token, and 'token' will be the original token value (IE: before being
        handled)

        Args:
            state (NormalizerState): dictionary of current state information. See ``bincfg.normalization.base_normalizer.NormalizerState``
        """
        found_base, found_mult, found_scale = False, False, False
        for i, (token_type, new_token, old_token) in enumerate(state.line[state.memory_start:]):
            state.token_type, state.orig_token, state.token, state.token_idx = token_type, old_token, old_token, state.memory_start + i

            # Ignore spacing and whatnot
            if token_type in [Tokens.SPACING, Tokens.DISASSEMBLER_INFO, Tokens.OPEN_BRACKET, Tokens.PLUS_SIGN]:
                continue

            # If this is a register, check if it is an index or base register
            elif token_type == Tokens.REGISTER:
                if found_base:
                    new_token = self.handle_memory_index(state)
                else:
                    new_token = self.handle_memory_base(state)
                    found_base = True
            
            # If we find a times sign, then update that state
            elif token_type == Tokens.TIMES_SIGN:
                found_mult = True
                continue
            
            # If this is an immediate value, check if it should be a scale or a displacement
            elif token_type == Tokens.IMMEDIATE:
                if found_mult and not found_scale:
                    new_token = self.handle_memory_scale(state)
                    found_scale = True
                else:
                    new_token = self.handle_memory_displacement(state)
            
            state.line[state.token_idx] = (token_type, new_token, old_token)
        
        state.memory_start = None
    
    def handle_memory_base(self, state):
        """Handles the 'base' section of a memory addressing. Defaults to returning it processed as a register

        Should return either the token to add to the current line, or None to not add any token

        Args:
            state (NormalizerState): dictionary of current state information. See bincfg.normalization.base_normalizer.NormalizerState
        """
        return self.handle_register(state)
    
    def handle_memory_index(self, state):
        """Handles the 'index' section of a memory addressing. Defaults to returning it processed as a register

        Should return either the token to add to the current line, or None to not add any token

        Args:
            state (NormalizerState): dictionary of current state information. See bincfg.normalization.base_normalizer.NormalizerState
        """
        return self.handle_register(state)
    
    def handle_memory_scale(self, state):
        """Handles the 'scale' section of a memory addressing. Defaults to returning it processed as an immediate

        Should return either the token to add to the current line, or None to not add any token

        Args:
            state (NormalizerState): dictionary of current state information. See bincfg.normalization.base_normalizer.NormalizerState
        """
        return self.handle_immediate(state)
    
    def handle_memory_displacement(self, state):
        """Handles the 'displacement' section of a memory addressing. Defaults to returning it processed as an immediate

        Should return either the token to add to the current line, or None to not add any token

        Args:
            state (NormalizerState): dictionary of current state information. See bincfg.normalization.base_normalizer.NormalizerState
        """
        return self.handle_immediate(state)

        
class X86InnerEyeNormalizer(X86BaseNormalizer):
    """A normalizer based on the Innereye method
    
    From the InnerEye paper: https://arxiv.org/pdf/1808.04706.pdf

    Rules:

        * Constant values are ignored and replaced with 'immval' or '-immval' for negative values
        * Function names are ignored and replaced with 'func'
        * Strings are 'str'
        * Jump destinations are 'immval'
        * Registers are left as-is 
        * Doesn't say anything about memory sizes, so they are ignored
        * Doesn't say anything about segment addresses, so they are ignored
        * Doesn't say anything about branch predictions, so they are ignored
        * Tokens are at the instruction-level
    
    
    Parameters
    ----------
    tokenizer: `Tokenizer`
        the tokenizer to use
    token_handlers: `Optional[Dict[str, Callable[[NormalizerState], Union[None, str]]]]`
        optional dictionary mapping string token types to functions to handle those tokens. These will override any
        token handlers that are used by default (IE: all of the `self.handle_*` functions). Functions should take one
        arg (the current normalizer state) as input and return either the next string token to add to the current line,
        or None to not add anything. This is useful for adding more methods to handle new token types that are not builtin.
    token_sep: `Optional[str]`
        the string to use to separate each token in returned instruction lines. Only used if tokenization_level is 
        'instruction'. If None, then a default value will be used (' ' for unnormalized using BaseNormalizer(), '_' 
        for everything else)
    tokenization_level: `Optional[Union[TokenizationLevel, str]]`
        the tokenization level to use for return values. Can be a string, or a ``TokenizationLevel`` type. Strings can be:

            - 'op': tokenized at the opcode/operand level. Will insert a 'INSTRUCTION_START' token at the beginning of
              each instruction line
            - 'inst'/'instruction': tokenized at the instruction level. All tokens in each instruction line are joined
              together using token_sep to construct the final token
            - 'auto': pick the default value for this normalization technique

    anonymize_tokens: `bool`
        if True, then tokens will be annonymized by taking their 4-byte shake_128 hash. Why does this exist? Bureaucracy.
    """

    DEFAULT_TOKENIZATION_LEVEL = TokenizationLevel.INSTRUCTION
    """"""
    renormalizable = False
    """"""

    handle_immediate = replace_immediate(include_negative=True)
    """"""
    handle_memory_size = ignore
    """"""
    handle_string_literal = replace_string_literal
    """"""
    handle_segment_address = ignore
    """"""
    handle_branch_prediction = ignore
    """"""
    opcode_function_call = replace_function_call_immediate(FUNCTION_CALL_STR)
    """"""

class X86DeepBinDiffNormalizer(X86BaseNormalizer):
    """A normalizer based on the Deep Bin Diff method
    
    From the DeepBinDiff paper: https://www.ndss-symposium.org/wp-content/uploads/2020/02/24311-paper.pdf

    Rules:

        * Constant values are ignored and replaced with 'immval'
        * General registers are renamed based on length, special ones are left as-is (with number information removed.
            EG: st5 -> st, rax -> reg8, r14d -> reg4, rip -> rip, zmm13 -> zmm)
        * Memory expressions are replaced with 'memexpr'
        * Can't really tell what's supposed to be done with function calls, will just assume they should be 'call immval'
        * Jump destinations are 'immval'
        * Strings are left as-is (Kinda bad, but they are doing binary diffing and not binary similarity, so I'll let it slide)
        * Doesn't say anything about segment addresses, so they are ignored
        * Doesn't say anything about branch predictions, so they are ignored
        * Tokens are at the op-level

    
    Parameters
    ----------
    replace_strings: `bool`
        if True, then strings will be replaced with a 'str' token. Default is False which is the default deepbindiff
        behavior in the paper and leaves full strings as individual tokens
    tokenizer: `Tokenizer`
        the tokenizer to use
    token_handlers: `Optional[Dict[str, Callable[[NormalizerState], Union[None, str]]]]`
        optional dictionary mapping string token types to functions to handle those tokens. These will override any
        token handlers that are used by default (IE: all of the `self.handle_*` functions). Functions should take one
        arg (the current normalizer state) as input and return either the next string token to add to the current line,
        or None to not add anything. This is useful for adding more methods to handle new token types that are not builtin.
    token_sep: `Optional[str]`
        the string to use to separate each token in returned instruction lines. Only used if tokenization_level is 
        'instruction'. If None, then a default value will be used (' ' for unnormalized using BaseNormalizer(), '_' 
        for everything else)
    tokenization_level: `Optional[Union[TokenizationLevel, str]]`
        the tokenization level to use for return values. Can be a string, or a ``TokenizationLevel`` type. Strings can be:

            - 'op': tokenized at the opcode/operand level. Will insert a 'INSTRUCTION_START' token at the beginning of
              each instruction line
            - 'inst'/'instruction': tokenized at the instruction level. All tokens in each instruction line are joined
              together using token_sep to construct the final token
            - 'auto': pick the default value for this normalization technique

    anonymize_tokens: `bool`
        if True, then tokens will be annonymized by taking their 4-byte shake_128 hash. Why does this exist? Bureaucracy.
    """

    DEFAULT_TOKENIZATION_LEVEL = TokenizationLevel.OPCODE
    """"""
    renormalizable = False
    """"""

    def __init__(self, replace_strings=False, tokenizer=None, token_handlers=None, token_sep=None, tokenization_level=TokenizationLevel.AUTO, 
                 anonymize_tokens=False):
        super().__init__(tokenizer=tokenizer, token_handlers=token_handlers, token_sep=token_sep, tokenization_level=tokenization_level, anonymize_tokens=anonymize_tokens)
        
        self._replace_strings = replace_strings
        self.handle_string_literal = replace_string_literal if replace_strings else self.handle_string_literal

    
    handle_immediate = replace_immediate
    """"""
    handle_memory_size = ignore
    """"""
    handle_register = x86_replace_general_register
    """"""
    handle_memory_expression = replace_memory_expression(MEMORY_EXPRESSION_STR)
    """"""
    handle_segment_address = ignore
    """"""
    handle_branch_prediction = ignore
    """"""
    opcode_function_call = replace_function_call_immediate(IMMEDIATE_VALUE_STR)
    """"""

    def __eq__(self, other):
        return super().__eq__(other) and eq_obj(self, other, selector='._replace_strings')
    
    def __hash__(self):
        return hash_obj([super().__hash__(), self._replace_strings], return_int=True)


class X86SafeNormalizer(X86BaseNormalizer):
    """A normalizer based on the SAFE method
    
    From the SAFE paper: https://github.com/gadiluna/SAFE

    Rules:

        * All base memory addresses (IE: memory addresses that are constant values) are replaced with 'dispmem'
          NOTE: they only specify that this is used for base memory addresses. I'm not sure if they mean any displacement
          values or only those that are alone with no other registers and whatnot. It doesn't help that their implementation
          seems to have some bugs here (see below). So, I assume it is to be used for any displacement values, hence
          the string they are replaced with
        * All immediate values greater than some threshold (`safe_threshold` parameter, they use 5000 in the paper) are 
          replaced with 'immval'. Any immediate values smaller than said threshold (including those that are targets
          of call/jump instructions) are left alone
        * Memory sizes are ignored
        * Doesn't say anything about registers, so they are left as-is
        * Doesn't say anything about segment addresses, so they are left as-is
        * Doesn't say anything about branch predictions, so they are ignored
        * Strings would be ignored, just using the immediate values associated with their memory address
        * Tokens are at the instruction-level
    
    NOTE: the code from the safe paper has at least one bug I've found (specifically, in their Radare2 analyzer code which
    is the only analyzer code I could find in their repo, even though they say they also use Angr), specifically when it 
    comes to memory expressions. They do not consider some of the possible memory addressing methods that are allowed
    in x86_64 binaries. For example:

    Original Disassembly                Their Result                    Probably Intended Result
    lea esi, [esi + ecx*2]              X_lea_esi,_[esi*2+0]            X_lea_esi,_[esi+ecx*2+0]
    lea ebx, [ebx + esi*4 + 0x10]       X_lea_ebx,_[ebx*4+16]           X_lea_ebx,_[ebx+esi*4+16]
    lea edi, [eax*4 + 0x419f40]         X_lea_edi,_[MEM]                X_lea_edi,_[eax*4+MEM]

    They also seem to have problems with immediate/displacement values in those memory addresses:

    Original Disassembly                Their Result                    Probably Intended Result
    add byte [ebp + 0x4d8de455], cl     X_add_[ebp*1+1301144661],_cl    X_add_[ebp*1+MEM],_cl

    So, I take a benefit-of-the-doubt approach to this normalizer and assume the authors did not intend for this to
    happen. Instructions are normalized taking into account these possible memory addressing methods. Any displacement
    values that are larger than the safe_threshold are converted into the dispmem token.

    
    Parameters
    ----------
    imm_threshold: `int`
        immediate values whose absolute value is <= imm_threshold will be left alone, those above it will be replaced 
        with the string 'immval'
    tokenizer: `Tokenizer`
        the tokenizer to use
    token_handlers: `Optional[Dict[str, Callable[[NormalizerState], Union[None, str]]]]`
        optional dictionary mapping string token types to functions to handle those tokens. These will override any
        token handlers that are used by default (IE: all of the `self.handle_*` functions). Functions should take one
        arg (the current normalizer state) as input and return either the next string token to add to the current line,
        or None to not add anything. This is useful for adding more methods to handle new token types that are not builtin.
    token_sep: `Optional[str]`
        the string to use to separate each token in returned instruction lines. Only used if tokenization_level is 
        'instruction'. If None, then a default value will be used (' ' for unnormalized using BaseNormalizer(), '_' 
        for everything else)
    tokenization_level: `Optional[Union[TokenizationLevel, str]]`
        the tokenization level to use for return values. Can be a string, or a ``TokenizationLevel`` type. Strings can be:

            - 'op': tokenized at the opcode/operand level. Will insert a 'INSTRUCTION_START' token at the beginning of
              each instruction line
            - 'inst'/'instruction': tokenized at the instruction level. All tokens in each instruction line are joined
              together using token_sep to construct the final token
            - 'auto': pick the default value for this normalization technique

    anonymize_tokens: `bool`
        if True, then tokens will be annonymized by taking their 4-byte shake_128 hash. Why does this exist? Bureaucracy.
    """

    DEFAULT_TOKENIZATION_LEVEL = TokenizationLevel.INSTRUCTION
    """"""
    renormalizable = False
    """"""

    def __init__(self, imm_threshold=DEFAULT_IMMEDIATE_THRESHOLD, tokenizer=None, token_handlers=None, token_sep=None, 
                 tokenization_level=TokenizationLevel.AUTO, anonymize_tokens=False):

        self._imm_threshold = imm_threshold
        super().__init__(tokenizer=tokenizer, token_handlers=token_handlers, token_sep=token_sep, tokenization_level=tokenization_level, anonymize_tokens=anonymize_tokens)

        self.handle_immediate = threshold_immediate(self._imm_threshold)
        self.handle_memory_displacement = threshold_immediate(self._imm_threshold, DISPLACEMENT_IMMEDIATE_STR)

    handle_memory_size = ignore
    """"""
    handle_string_literal = ignore
    """"""
    handle_branch_prediction = ignore
    """"""

    def __eq__(self, other):
        return super().__eq__(other) and eq_obj(self, other, '._imm_threshold')
    
    def __hash__(self):
        return hash_obj([super().__hash__(), self._imm_threshold], return_int=True)
    

class X86DeepSemanticNormalizer(X86BaseNormalizer):
    """A normalizer based on the Deepsemantic method
    
    from the DeepSemantic paper:
    https://arxiv.org/abs/2106.05478

    Rules:

        * Immediates can fall into multiple categories:
            a. Function calls:

               - libc function name(): "libc[name]" (instead, we use "{name}")
               - recursive call: 'self'
               - function within the binary: 'innerfunc'
               - function outside the binary: 'externfunc'
               - NOTE: they do not take into account call tables that could theoretically call both inner and extern
                 functions. So, when this rather rare even occurs, it is given the token 'multifunc'

            b. Jump (branching) family: "jmpdst"

            c. Reference: (NOTE: This might not be done for all disassebly output, such as ROSE, since they don't always have
               this information readily available)
               
               - String literal: 'str'
               - Statically allocated variable: "dispbss"
               - Data (data other than a string): "dispdata"

            d. Default (all other immediate values): "immval"

        * Registers can fall into multiple categories:
            a. Stack/Base/Instruction pointer: Keep track of type and size
               [e|r]*[b|s|i]p[l]*  ->  [s|b|i]p[1|2|4|8]
            b. Special purpose (IE: flags): Keep track of type
               cr[0-15], dr[0-15], st([0-7]), [c|d|e|f|g|s]s  ->  reg[cr|dr|st], reg[c|d|e|f|s]s
            c. AVX registers: Keep track of type
               [x|y|z]*mm[0-7|0-31]  ->  reg[x|y|z]*mm
            d. General purpose registers: Keep track of size
               [e|r]*[a|b|c|d|si|di][x|l|h]*, r[8-15][b|w|d]*  ->  reg[1|2|4|8]
        
        * Pointers can fall into multiple categories:
            a. Direct, small: keep track of size
               byte,word,dword,qword,ptr  ->  memptr[1|2|4|8]
            b. Direct, large: keep track of size
               tbyte,xword,[x|y|z]mmword  ->  memptr[10|16|32|64]
            c. Indirect, string: 
               [base+index*scale+displacement]  ->  [base+index*scale+dispstr]  (NOTE: we use 'str' instead of dispstr)
            d. Indirect, not string:
               [base+index*scale+displacement]  ->  [base+index*scale+disp]

               NOTE: for our purposes, we don't necessarily always have base, index, scale, and displacement present, and
               they may appear in a different (but deterministic) order. It shouldn't really change anything for any 
               models, just how the tokens are formatted

               NOTE: it looks like the 'scale' values are their original immediate values and not replaced with the
               'immval' string, so that is taken into account as well
        
        * Tokenized at instruction-level
        * Doesn't say anything about segment addresses, so they are left as-is
        * Doesn't say anything about branch predictions, so they are ignored
    
    
    Parameters
    ----------
    special_functions: `Optional[Set[str]]`
        a set of special function names. All external functions whose name (ignoring any '@plt' at the end) is in this 
        set will have their name kept, otherwise they will be replaced with 'externfunc'. If None, will attempt to load 
        the default special function names from :func:`~bincfg.utils.cfg_utils.get_special_function_names`. If you do not
        wish to use any special function names, then pass an empty set.
    tokenizer: `Tokenizer`
        the tokenizer to use
    token_handlers: `Optional[Dict[str, Callable[[NormalizerState], Union[None, str]]]]`
        optional dictionary mapping string token types to functions to handle those tokens. These will override any
        token handlers that are used by default (IE: all of the `self.handle_*` functions). Functions should take one
        arg (the current normalizer state) as input and return either the next string token to add to the current line,
        or None to not add anything. This is useful for adding more methods to handle new token types that are not builtin.
    token_sep: `Optional[str]`
        the string to use to separate each token in returned instruction lines. Only used if tokenization_level is 
        'instruction'. If None, then a default value will be used (' ' for unnormalized using BaseNormalizer(), '_' 
        for everything else)
    tokenization_level: `Optional[Union[TokenizationLevel, str]]`
        the tokenization level to use for return values. Can be a string, or a ``TokenizationLevel`` type. Strings can be:

            - 'op': tokenized at the opcode/operand level. Will insert a 'INSTRUCTION_START' token at the beginning of
              each instruction line
            - 'inst'/'instruction': tokenized at the instruction level. All tokens in each instruction line are joined
              together using token_sep to construct the final token
            - 'auto': pick the default value for this normalization technique

    anonymize_tokens: `bool`
        if True, then tokens will be annonymized by taking their 4-byte shake_128 hash. Why does this exist? Bureaucracy.
    """

    DEFAULT_TOKENIZATION_LEVEL = TokenizationLevel.INSTRUCTION
    """"""
    renormalizable = False
    """"""

    def __init__(self, special_functions=None, tokenizer=None, token_handlers=None, token_sep=None, tokenization_level=TokenizationLevel.AUTO):
        self.special_functions = LIBC_FUNCTION_NAMES if special_functions is None else set(special_functions)
        super().__init__(tokenizer=tokenizer, token_handlers=token_handlers, token_sep=token_sep, tokenization_level=tokenization_level)
    
    handle_register = x86_replace_general_register
    """"""
    handle_immediate = replace_immediate
    """"""
    handle_memory_size = x86_memsize_value
    """"""
    handle_string_literal = replace_string_literal
    """"""
    handle_memory_scale = lambda self, state: str(imm_to_int(state.token))
    """"""
    handle_branch_prediction = ignore
    """"""
    opcode_function_call = special_function_call
    """"""
    opcode_jump = replace_jump_destination
    """"""

    def __eq__(self, other):
        return super().__eq__(other) and eq_obj(self, other, selector='special_functions')
    
    def __hash__(self):
        return hash_obj([super().__hash__(), self.special_functions], return_int=True)


class X86CompressedStatsNormalizer(X86BaseNormalizer):
    """A normalizer I created for use in CFG.get_compressed_stats()

    Rules:

        * Immediates are replaced with immediate string (including negative)
        * function calls are either self vs. intern vs. extern func, no special functions
        * jump destinations are 'jmpdst'
        * registers are handled the same as deepsem/deepbindiff
        * memory pointers/memory expressions are handled the same as in deepsemantic
        * Tokenized at the instruction-level
        * segment addresses are ignored
        * branch predictions are ignored
    
    Parameters
    ----------
    special_functions: `Optional[Set[str]]`
        a set of special function names. All external functions whose name (ignoring any '@plt' at the end) is in this 
        set will have their name kept, otherwise they will be replaced with 'externfunc'. If None, will attempt to load 
        the default special function names from :func:`~bincfg.utils.cfg_utils.get_special_function_names`. If you do not
        wish to use any special function names, then pass an empty set.
    tokenizer: `Tokenizer`
        the tokenizer to use
    token_handlers: `Optional[Dict[str, Callable[[NormalizerState], Union[None, str]]]]`
        optional dictionary mapping string token types to functions to handle those tokens. These will override any
        token handlers that are used by default (IE: all of the `self.handle_*` functions). Functions should take one
        arg (the current normalizer state) as input and return either the next string token to add to the current line,
        or None to not add anything. This is useful for adding more methods to handle new token types that are not builtin.
    token_sep: `Optional[str]`
        the string to use to separate each token in returned instruction lines. Only used if tokenization_level is 
        'instruction'. If None, then a default value will be used (' ' for unnormalized using BaseNormalizer(), '_' 
        for everything else)
    tokenization_level: `Optional[Union[TokenizationLevel, str]]`
        the tokenization level to use for return values. Can be a string, or a ``TokenizationLevel`` type. Strings can be:

            - 'op': tokenized at the opcode/operand level. Will insert a 'INSTRUCTION_START' token at the beginning of
              each instruction line
            - 'inst'/'instruction': tokenized at the instruction level. All tokens in each instruction line are joined
              together using token_sep to construct the final token
            - 'auto': pick the default value for this normalization technique

    anonymize_tokens: `bool`
        if True, then tokens will be annonymized by taking their 4-byte shake_128 hash. Why does this exist? Bureaucracy.
    """

    DEFAULT_TOKENIZATION_LEVEL = TokenizationLevel.INSTRUCTION
    """"""
    renormalizable = False
    """"""

    def __init__(self, special_functions=None, tokenizer=None, token_handlers=None, token_sep=None, tokenization_level=TokenizationLevel.AUTO):

        self.special_functions = set() if special_functions is None else special_functions
        super().__init__(tokenizer=tokenizer, token_handlers=token_handlers, token_sep=token_sep, tokenization_level=tokenization_level)
            
    handle_register = x86_replace_general_register
    """"""
    handle_memory_size = x86_memsize_value
    """"""
    handle_immediate = replace_immediate(include_negative=True)
    """"""
    handle_string_literal = replace_string_literal
    """"""
    handle_segment_address = ignore
    """"""
    handle_branch_prediction = ignore
    """"""
    opcode_function_call = special_function_call
    """"""
    opcode_jump = replace_jump_destination
    """"""

    def __eq__(self, other):
        return super().__eq__(other) and eq_obj(self, other, selector='special_functions')
    
    def __hash__(self):
        return hash_obj([super().__hash__(), self.special_functions], return_int=True)


class X86HPCDataNormalizer(X86BaseNormalizer):
    """A special normalizer meant for use in HPC compile jobs
    
    This normalizer is made to reduce the total number of new tokens as much as possible while still being able to fully
    reproduce original BaseNormalizer output, and while trying to minimize the number of tokens per assembly line as
    much as possible.

    Since immediate values make up the vast majority of 'unique' tokens (or are the root cause of there being so many)
    in the BaseNormalizer, this is all that is changed. Specifically:

        - immediate values get split into multiple tokens. EG: '123456789' -> '1234', '5678', '9' if using `num_digits` of 4
        - negatives stay connected to tokens. EG: '-54321' -> '-543', '21' if using `num_digits` of 4
        - Before some split immediate values, a 'split immediate' token is inserted for later tokenization to know that the
          following immediate values should all be concatenated together. This is only inserted when default behavior
          would produce the wrong values (EG: whenever a token is split, whenever a non-split token has a split token
          before it, etc.)
    
    NOTE: this should only be used with the 'opcode' tokenization level as it provides no benefit otherwise
    

    Parameters
    ----------
    num_digits: `int`
        the number of digits to use before splitting. This will include the minus sign as a digit.
    replace_strings: `bool`
        if True, then strings will be replaced with a 'str' token. Default is False which is the default deepbindiff
        behavior in the paper and leaves full strings as individual tokens
    tokenizer: `Tokenizer`
        the tokenizer to use
    token_handlers: `Optional[Dict[str, Callable[[NormalizerState], Union[None, str]]]]`
        optional dictionary mapping string token types to functions to handle those tokens. These will override any
        token handlers that are used by default (IE: all of the `self.handle_*` functions). Functions should take one
        arg (the current normalizer state) as input and return either the next string token to add to the current line,
        or None to not add anything. This is useful for adding more methods to handle new token types that are not builtin.
    token_sep: `Optional[str]`
        the string to use to separate each token in returned instruction lines. Only used if tokenization_level is 
        'instruction'. If None, then a default value will be used (' ' for unnormalized using BaseNormalizer(), '_' 
        for everything else)
    tokenization_level: `Optional[Union[TokenizationLevel, str]]`
        the tokenization level to use for return values. Can be a string, or a ``TokenizationLevel`` type. Strings can be:

            - 'op': tokenized at the opcode/operand level. Will insert a 'INSTRUCTION_START' token at the beginning of
              each instruction line
            - 'inst'/'instruction': tokenized at the instruction level. All tokens in each instruction line are joined
              together using token_sep to construct the final token
            - 'auto': pick the default value for this normalization technique

    anonymize_tokens: `bool`
        if True, then tokens will be annonymized by taking their 4-byte shake_128 hash. Why does this exist? Bureaucracy.
    """

    DEFAULT_TOKENIZATION_LEVEL = TokenizationLevel.OPCODE
    """"""

    @property
    def renormalizable(self):
        """We are only losslessly renormalizable if self._replace_strings == False"""
        return hasattr(self, '_replace_strings') and not self._replace_strings

    def __init__(self, num_digits=4, replace_strings=False, tokenizer=None, token_handlers=None, token_sep=None, tokenization_level=TokenizationLevel.AUTO):
        if num_digits < 1:
            raise ValueError("`num_digits` must be >= 1")
        
        self._num_digits = num_digits
        self._replace_strings = replace_strings
        super().__init__(tokenizer=tokenizer, token_handlers=token_handlers, token_sep=token_sep, tokenization_level=tokenization_level)

        self.handle_string_literal = replace_string_literal(replace_previous_immediate=True) if replace_strings else super().handle_string_literal

    def finalize_instruction(self, state):
        """Handles a single instruction. Calls super()'s handle_instruction, then performs the immediate splitting
        
        Args:
            state (NormalizerState): dictionary of current state information. See ``bincfg.normalization.base_normalizer.NormalizerState``
        """
        super().finalize_instruction(state)

        new_inst = []
        prev_split = False
        for token_type, new_token, old_token in state.line:
            if token_type in [Tokens.IMMEDIATE] and (len(new_token) > self._num_digits or prev_split):
                
                new_inst.append((Tokens.SPLIT_IMMEDIATE, SPLIT_IMMEDIATE_TOKEN, SPLIT_IMMEDIATE_TOKEN))
                while len(new_token) > 0:
                    new_inst.append((Tokens.IMMEDIATE, new_token[:self._num_digits], new_token[:self._num_digits]))
                    new_token = new_token[self._num_digits:]

                prev_split = True
            else:
                new_inst.append((token_type, new_token, old_token))
                prev_split = False
        
        state.line = new_inst

    def __eq__(self, other):
        return super().__eq__(other) and eq_obj(self, other, selector='._num_digits') and eq_obj(self, other, selector='._replace_strings')
    
    def __hash__(self):
        return hash_obj([super().__hash__(), self._num_digits, self._replace_strings], return_int=True)
    