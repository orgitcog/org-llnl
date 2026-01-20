from ..base_normalizer import BaseNormalizer
from ..base_tokenizer import TokenizationLevel
from .java_tokenizer import JAVA_DEFAULT_TOKENIZER
from ..norm_funcs import threshold_immediate, replace_immediate
from ...utils import hash_obj


# All of these are the 'invoke' commands
JAVA_FUNCTION_CALL_RE = r'invoke(?:virtual|interface|special|static|dynamic)'

# Three types: conditional (the 'if' ones), tabular (switch/lookup tables), and unconditional (goto, jsr, etc)
# This one does technically match to 4 instructions that don't exist: if_acmp[lt,le,gt,ge], but that should be fine, this looks nicer
_JAVA_CONDITIONAL_JUMP_RE = r'if(?:{null}|(?:_[ia]cmp)?{comps})'.format(
    null=r'(?:non)?null',
    comps=r'(?:eq|ne|lt|le|gt|ge)',
)
_JAVA_UNCONDITIONAL_JUMP_RE = r'(?:goto|jsr)(?:_w)?'
_JAVA_SWITCH_JUMP_RE = r'(?:table|lookup)switch'
JAVA_JUMP_RE = r'{cond}|{switch}|{uncond}'.format(cond=_JAVA_CONDITIONAL_JUMP_RE, uncond=_JAVA_UNCONDITIONAL_JUMP_RE, switch=_JAVA_SWITCH_JUMP_RE)


class JavaBaseNormalizer(BaseNormalizer):
    """A base class for a normalization method. 
    
    Performs an 'unnormalized' normalization, removing what is likely extraneous information, and providing a base class
    for other normalization methods to inherit from.
    
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
    """The separator string used for this normalizer
    
    Will default to ' ' for ``BaseNormalizer``, and '_' for all other normalizers.
    """

    tokenization_level: TokenizationLevel
    """The tokenization level to use for this normalizer"""

    def __init__(self, tokenizer=None, token_handlers=None, token_sep=None, tokenization_level=TokenizationLevel.AUTO, anonymize_tokens=False):
        super().__init__(tokenizer=tokenizer if tokenizer is not None else JAVA_DEFAULT_TOKENIZER, token_handlers=token_handlers, 
                         token_sep=token_sep if token_sep is not None else ' ',
                         tokenization_level=tokenization_level, anonymize_tokens=anonymize_tokens)
        self.register_opcode_handler(JAVA_FUNCTION_CALL_RE, self.opcode_function_call)
        self.register_opcode_handler(JAVA_JUMP_RE, self.opcode_jump)
    
    def opcode_function_call(self, state):
        """Handles function call opcodes, defaults to doing nothing"""
        return None
    
    def opcode_jump(self, state):
        """Handles jump opcodes, defaults to doing nothing"""
        return None


class JavaReplaceImmediateNormalizer(JavaBaseNormalizer):
    """Replaces all immediate values over some threshold with the immediate token
    
    Parameters
    ----------
    imm_threshold: `Optional[int]`
        all immediate values whose absolute value is greater than this threshold will be replaced with the immediate
        value token. If None or < 0, then all immediates will be replaced no matter the size
    include_negative: `bool`
        if True, then a negative sign will be added to the front of all replaced immediate tokens that are negative
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
    def __init__(self, imm_threshold=None, include_negative=True, tokenizer=None, token_handlers=None, token_sep=None, 
                 tokenization_level=TokenizationLevel.AUTO, anonymize_tokens=False):
        super().__init__(tokenizer=tokenizer, token_handlers=token_handlers, token_sep=token_sep, 
                         tokenization_level=tokenization_level, anonymize_tokens=anonymize_tokens)

        self.imm_threshold = -1 if imm_threshold is None or imm_threshold < 0 else imm_threshold
        self.include_negative = include_negative

        self.handle_immediate = replace_immediate(include_negative=self.include_negative) if self.imm_threshold == -1 else \
            threshold_immediate(imm_threshold, include_negative)
    
    def __eq__(self, other):
        return super().__eq__(other) and self.imm_threshold == other.imm_threshold and self.include_negative == other.include_negative
    
    def __hash__(self):
        return super().__hash__() + hash_obj([self.imm_threshold, self.include_negative], return_int=True)
