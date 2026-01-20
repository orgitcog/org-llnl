from ..base_tokenizer import RE_DISASSEMBLER_INFO, RE_SPACING, RE_NEWLINE, Tokens , BaseTokenizer, Architectures


# The delimiter lookahead. Allows for lookahead to make sure there is some delimiter, and we don't greedily grab 
#   registers/values/etc in case their names interfere with anything else.
# Can be a spacing, newline, or end of string
JAVA_RE_DELIMITER = r'{spacing}|{newline}|{dis_info}|$'.format(spacing=RE_SPACING, newline=RE_NEWLINE, dis_info=RE_DISASSEMBLER_INFO)

# Check for any opcode mnemonic. For now, we just get any and all characters that could correspond to an opcode, 
#   instruction prefix, or branch prediction. These will be sorted out later when parsing. Must be followed by delimiter
JAVA_RE_OPCODE = r'(?:[a-z][a-z0-9_]*)(?={delim})'.format(delim=JAVA_RE_DELIMITER)

# Check for the one 'prefix' opcode: wide
JAVA_RE_PREFIX = r'wide(?={delim})'.format(delim=JAVA_RE_DELIMITER)

# Match names to tokens, and define the order in which they should be matched
JAVA_DEFAULT_TOKENS = [
    (Tokens.INSTRUCTION_PREFIX, JAVA_RE_PREFIX),  # The 'wide' instruction prefix
    (Tokens.OPCODE, JAVA_RE_OPCODE),  # Opcode. Should go last since it's the most general matching. 
]
"""Default list of (token_type, regex) token tuples to match to"""


class JavaBaseTokenizer(BaseTokenizer):
    """A default class to tokenize java bytecode line input
    
    The tokenizer will tokenize essentially anything, so long as it fits known tokens.

    Known Tokens:

        * All of the default tokens from ``BaseTokenizer``
        * Instruction prefix: the 'wide' prefix
        * Opcode: any alpha-numeric + underscore substring

    Anything that does not fit one of the above tokens will be considered a 'token mismatch'


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
        If True, then it is assumed that all regular expressions will exactly match case. If False, then it is assumed
        that all regular expressions only handle lowercase strings, and all incoming instructions will be converted to lowercase
    """
    DEFAULT_TOKENS = JAVA_DEFAULT_TOKENS

    ARCHITECTURE = Architectures.JAVA
    """The architecture this tokenizer works on"""
    

JAVA_DEFAULT_TOKENIZER = JavaBaseTokenizer()
