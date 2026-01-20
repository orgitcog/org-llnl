"""Testing BaseTokenizer"""

import pytest
import pickle
from bincfg import Tokens, BaseTokenizer, TokenMismatchError, UnknownTokenError
from bincfg.normalization.norm_utils import INSTRUCTION_START_TOKEN, SPLIT_IMMEDIATE_TOKEN


# For inserting a new token type into the tests
FAKE_TOKEN_TYPE = 'fake'
def token_fake(state):
    return 'FAKEFAKE_' + state['token'].partition('_')[-1]
def pretend_call_token_fake(string):
    return token_fake({'token': string})


# Whether or not to pickle the tokenizers to make sure pickling doesn't have any affect on outcome
def _pickle(tokenizer, pickled):
    if pickled == 'pickled':
        return pickle.loads(pickle.dumps(tokenizer))
    return tokenizer


@pytest.fixture(scope='module')
def fake_tokenizer():
    """A fake tokenizer for an entirely new language
    
    The tokenizer follows the following rules:

        * Normal instruction addresses
        * normal immediates of bin, hex, octal, decimal
        * string literals
        * disassembler information
        * all symbols '+', '[', ']', etc.
        * Spacing is any amount of space ' ' or '&' characters in a row
        * normal newlines
        * FAKE tokens: type=FAKE_TOKEN_TYPE, anything that matches r'fake_[a-z]+'
        * We DO NOT allow for the split immediate or start instruction tokens
        * Opcodes are all alpha-numeric substrings that start with an alphabet character
        * we mismatch on all tokens that don't fit above and are not '%', IE: all '%' characters should be immediately ignored
    
    This tokenizer includes special tokens, and is NOT case-sensitive
    """
    tokens = [
        (Tokens.SPACING, r'[ &]+'),
        (FAKE_TOKEN_TYPE, r'fake_[a-z]+'),
        (Tokens.SPLIT_IMMEDIATE, None),
        (Tokens.INSTRUCTION_START, None),
        (Tokens.OPCODE, r'[a-z][a-z0-9]+'),
        (Tokens.MISMATCH, r'[^%].'),
    ]

    return BaseTokenizer(tokens=tokens, token_handlers={FAKE_TOKEN_TYPE: token_fake}, insert_special_tokens=True, case_sensitive=False)


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_construct_new_tokenizer(fake_tokenizer, pickled):
    """We can even construct this new tokenizer"""
    fake_tokenizer = _pickle(fake_tokenizer, pickled)
    assert isinstance(fake_tokenizer, BaseTokenizer)


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_kept_special_tokens(fake_tokenizer, pickled):
    """The special tokens that we kept are still there and working"""
    fake_tokenizer = _pickle(fake_tokenizer, pickled)
    
    # Immediates of different types
    assert fake_tokenizer.tokenize("1234 0x1234 0b011001 1234 0o777") == [
        (Tokens.INSTRUCTION_ADDRESS, '1234'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x1234'), (Tokens.SPACING, ' '), 
        (Tokens.IMMEDIATE, '0b011001'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '1234'), (Tokens.SPACING, ' '), 
        (Tokens.IMMEDIATE, '0o777'), (Tokens.NEWLINE, '\n')
    ]

    # String literals
    assert fake_tokenizer.tokenize("test 'this is a % $ & string fake_abc' 1234") == [
        (Tokens.OPCODE, 'test'), (Tokens.SPACING, ' '), (Tokens.STRING_LITERAL, "'this is a % $ & string fake_abc'"),
        (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '1234'), (Tokens.NEWLINE, '\n')
    ]

    # Disassembler info, symbols, and newlines
    assert fake_tokenizer.tokenize("test <this is \"rose\" & 1234 info> + *[]: \n newline + | more newline") == [
        (Tokens.OPCODE, 'test'), (Tokens.SPACING, ' '), (Tokens.DISASSEMBLER_INFO, '<this is \"rose\" & 1234 info>'),
        (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'), (Tokens.SPACING, ' '), (Tokens.TIMES_SIGN, '*'), (Tokens.OPEN_BRACKET, '['),
        (Tokens.CLOSE_BRACKET, ']'), (Tokens.COLON, ':'), (Tokens.SPACING, ' '), (Tokens.NEWLINE, '\n'), (Tokens.SPACING, ' '),
        (Tokens.OPCODE, 'newline'), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'), (Tokens.SPACING, ' '), (Tokens.NEWLINE, '|'),
        (Tokens.SPACING, ' '), (Tokens.OPCODE, 'more'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'newline'), (Tokens.NEWLINE, '\n')
    ]
    

@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_overridden_tokens(fake_tokenizer, pickled):
    """The tokens we override (either with new regex, or with None to delete) work properly"""
    fake_tokenizer = _pickle(fake_tokenizer, pickled)

    # Removed tokens
    with pytest.raises(TokenMismatchError):
        fake_tokenizer.tokenize("this&should {start} fail".format(start=INSTRUCTION_START_TOKEN))
    with pytest.raises(TokenMismatchError):
        fake_tokenizer.tokenize("this&should 1234 also{split} fail".format(split=SPLIT_IMMEDIATE_TOKEN))
    
    # Override space
    assert fake_tokenizer.tokenize("test&split&1234&&&&space  &&") == [
        (Tokens.OPCODE, 'test'), (Tokens.SPACING, '&'), (Tokens.OPCODE, 'split'), (Tokens.SPACING, '&'), (Tokens.IMMEDIATE, '1234'), 
        (Tokens.SPACING, '&&&&'), (Tokens.OPCODE, 'space'), (Tokens.SPACING, '  &&'), (Tokens.NEWLINE, '\n')
    ]

    # Override mismatch
    assert fake_tokenizer.tokenize("test skip%ped%%%values7%%&%") == [
        (Tokens.OPCODE, 'test'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'skip'), (Tokens.OPCODE, 'ped'), 
        (Tokens.OPCODE, 'values7'), (Tokens.SPACING, '&'), (Tokens.NEWLINE, '\n')
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_new_token_type(fake_tokenizer, pickled):
    """Inserting a new token type and token handler function"""
    fake_tokenizer = _pickle(fake_tokenizer, pickled)

    assert fake_tokenizer.tokenize("new fake_token token") == [
        (Tokens.OPCODE, 'new'), (Tokens.SPACING, ' '), (FAKE_TOKEN_TYPE, pretend_call_token_fake('fake_token')),
        (Tokens.SPACING, ' '), (Tokens.OPCODE, 'token'), (Tokens.NEWLINE, '\n')
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_case_insensitive(fake_tokenizer, pickled):
    """Case insensitivity"""
    fake_tokenizer = _pickle(fake_tokenizer, pickled)

    assert fake_tokenizer.tokenize("Add 1234 FaKe_Apples") == [
        (Tokens.OPCODE, 'Add'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '1234'), (Tokens.SPACING, ' '), 
        (FAKE_TOKEN_TYPE, pretend_call_token_fake('FaKe_Apples')), (Tokens.NEWLINE, '\n')
    ]


@pytest.fixture(scope='module')
def fake_tokenizer_other():
    """A different fake tokenizer for an entirely new language
    
    The tokenizer follows the following rules:

        * case-sensitive
        * no special tokens inserted
        * We only check for 'fake' tokens, split immediate, opcode tokens, and mismatch tokens
        * We also insert a second 'fake' token '(', but forget to put it into token_handlers
    
    """
    tokens = [
        (FAKE_TOKEN_TYPE, r'fAke_[a-z]+'),
        (Tokens.SPLIT_IMMEDIATE, SPLIT_IMMEDIATE_TOKEN),
        (Tokens.OPCODE, r'[a-z][a-z0-9A-Z]+'),
        ('unknown', r'\('),
        (Tokens.MISMATCH, r'.'),
    ]

    return BaseTokenizer(tokens=tokens, token_handlers={FAKE_TOKEN_TYPE: token_fake}, insert_special_tokens=False, case_sensitive=True)


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_no_other_tokens(fake_tokenizer_other, pickled):
    """None of the normal special tokens exists"""
    fake_tokenizer_other = _pickle(fake_tokenizer_other, pickled)

    with pytest.raises(TokenMismatchError):
        fake_tokenizer_other.tokenize("1234")
    with pytest.raises(TokenMismatchError):
        fake_tokenizer_other.tokenize("test 'a' test")
    with pytest.raises(TokenMismatchError):
        fake_tokenizer_other.tokenize("test <disinfo>")
    with pytest.raises(TokenMismatchError):
        fake_tokenizer_other.tokenize("{start}".format(start=INSTRUCTION_START_TOKEN))


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_case_sensitive(fake_tokenizer_other, pickled):
    "Case-sensitivity"
    fake_tokenizer_other = _pickle(fake_tokenizer_other, pickled)

    assert fake_tokenizer_other.tokenize("testCaseSenSitive") == [
        (Tokens.OPCODE, 'testCaseSenSitive'), (Tokens.NEWLINE, '\n')
    ]
    assert fake_tokenizer_other.tokenize("fAke_abc") == [
        (FAKE_TOKEN_TYPE, pretend_call_token_fake('fAke_abc')), (Tokens.NEWLINE, '\n')
    ]

    with pytest.raises(TokenMismatchError):
        fake_tokenizer_other.tokenize("Test")
    with pytest.raises(TokenMismatchError):
        fake_tokenizer_other.tokenize("fake_abc")


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_inserted_split_imm(fake_tokenizer_other, pickled):
    """Inserting another token thing"""
    fake_tokenizer_other = _pickle(fake_tokenizer_other, pickled)

    # Split immediate is handled silently
    assert fake_tokenizer_other.tokenize("testTest{split}test".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        (Tokens.OPCODE, 'testTest'), (Tokens.OPCODE, 'test'), (Tokens.NEWLINE, '\n')
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_unknown_token(fake_tokenizer_other, pickled):
    """Forgot to insert the new token_type into the token handlers"""
    fake_tokenizer_other = _pickle(fake_tokenizer_other, pickled)

    with pytest.raises(UnknownTokenError):
        fake_tokenizer_other.tokenize("test()")


class FakeTokenizer(BaseTokenizer):

    DEFAULT_NEWLINE_TUPLE = (Tokens.SPLIT_IMMEDIATE, '#')

    DEFAULT_TOKENS = [
        (Tokens.REGISTER, r'(?:[re][abcd]x)'),
        (Tokens.OPCODE, r'[a-z][a-z0-9]+'),
    ]

    def __init__(self, tokens=None, token_handlers=None, insert_special_tokens=True, case_sensitive=False):
        super().__init__(tokens=tokens, token_handlers=token_handlers, insert_special_tokens=insert_special_tokens, case_sensitive=case_sensitive)

        # Doing it this way so we can check pickling works correctly
        def token_opcode(self, state):
            if 'kwargs_test' in state['kwargs']:
                return state['kwargs']['kwargs_test']
            return super().token_opcode(state)
        self.token_opcode = lambda *args, **kwargs: token_opcode(self, *args, **kwargs)
    
    def token_mismatch(self, state):
        if 'test_mismatch' in state['kwargs']:
            return state['token']
        super().token_mismatch(state)


@pytest.fixture(scope='module')
def tokenizer():
    return FakeTokenizer()
    

@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_default_class_values(tokenizer, pickled):
    """The default values for newline tuples, tokens, etc. are used correctly"""
    tokenizer = _pickle(tokenizer, pickled)

    assert tokenizer.tokenize("1234 add rax, rax opcode") == [
        (Tokens.INSTRUCTION_ADDRESS, '1234'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), 
        (Tokens.REGISTER, 'rax'), (Tokens.SPACING, ', '), (Tokens.REGISTER, 'rax'), (Tokens.SPACING, ' '), 
        (Tokens.OPCODE, 'opcode'), tokenizer.DEFAULT_NEWLINE_TUPLE,
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_override_default_class_values_when_calling(tokenizer, pickled):
    """Overriding default class values when calling the `.tokenize()` method"""
    tokenizer = _pickle(tokenizer, pickled)

    new_newline_tup = (Tokens.MISMATCH, 'newline')
    assert tokenizer.tokenize("1234 add rax", match_instruction_address=False, newline_tup=new_newline_tup) == [
        (Tokens.IMMEDIATE, '1234'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), 
        (Tokens.REGISTER, 'rax'), new_newline_tup,
    ]

    # None newline tup doesn't insert it
    new_newline_tup = None
    assert tokenizer.tokenize("1234 add rax", match_instruction_address=False, newline_tup=new_newline_tup) == [
        (Tokens.IMMEDIATE, '1234'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), 
        (Tokens.REGISTER, 'rax'),
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_override_functions(tokenizer, pickled):
    """Overriding builtin functions"""
    tokenizer = _pickle(tokenizer, pickled)

    assert tokenizer.tokenize("test # mismatch", test_mismatch=True) == [
        (Tokens.OPCODE, 'test'), (Tokens.SPACING, ' '), (Tokens.MISMATCH, '#'), (Tokens.SPACING, ' '), 
        (Tokens.OPCODE, 'mismatch'), tokenizer.DEFAULT_NEWLINE_TUPLE,
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_passed_kwargs(tokenizer, pickled):
    """Extra passed kwargs to the `.tokenize()` function end up in state['kwargs']"""
    tokenizer = _pickle(tokenizer, pickled)

    assert tokenizer.tokenize("test 1234 opcode", kwargs_test='kwargs') == [
        (Tokens.OPCODE, 'kwargs'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '1234'), (Tokens.SPACING, ' '),
        (Tokens.OPCODE, 'kwargs'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_passed_tokens(pickled):
    """When we pass our own tokens to the FakeTokenizer() constructor"""
    # Don't pass the registers, everything is an opcode
    tokenizer = FakeTokenizer(tokens=[
        (Tokens.OPCODE, r'[a-z][a-z0-9]+'),
    ])
    tokenizer = _pickle(tokenizer, pickled)

    assert tokenizer.tokenize("add rax, rax") == [
        (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'rax'), (Tokens.SPACING, ', '),
        (Tokens.OPCODE, 'rax'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_strings_star_args(tokenizer, pickled):
    """Tests sending strings as both a single string, and a star-args of multiple strings"""
    tokenizer = _pickle(tokenizer, pickled)

    strings = [
        'add rax, rax',
        'hey rose <information> "string"',
        'an + lot  .. **of: : weird -1234 symbols',
        'multiple \n new rax ebx lines|12343 together',
    ]
    
    old_newline = tokenizer.DEFAULT_NEWLINE_TUPLE
    tokenizer.DEFAULT_NEWLINE_TUPLE = (Tokens.NEWLINE, '\n')

    result = [
        (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'), (Tokens.SPACING, ', '), 
        (Tokens.REGISTER, 'rax'), tokenizer.DEFAULT_NEWLINE_TUPLE, 
        
        (Tokens.OPCODE, 'hey'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'rose'), (Tokens.SPACING, ' '), 
        (Tokens.DISASSEMBLER_INFO, '<information>'), (Tokens.SPACING, ' '), ('string_literal', '"string"'), 
        tokenizer.DEFAULT_NEWLINE_TUPLE, 
        
        (Tokens.OPCODE, 'an'), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'lot'), 
        (Tokens.SPACING, '  .. '), (Tokens.TIMES_SIGN, '*'), (Tokens.TIMES_SIGN, '*'), (Tokens.OPCODE, 'of'), (Tokens.COLON, ':'), 
        (Tokens.SPACING, ' '), (Tokens.COLON, ':'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'weird'), (Tokens.SPACING, ' '), 
        (Tokens.IMMEDIATE, '-1234'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'symbols'), tokenizer.DEFAULT_NEWLINE_TUPLE, 
        
        (Tokens.OPCODE, 'multiple'), (Tokens.SPACING, ' '), (Tokens.NEWLINE, '\n'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'new'), 
        (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'ebx'), (Tokens.SPACING, ' '), 
        (Tokens.OPCODE, 'lines'), (Tokens.NEWLINE, '|'), (Tokens.INSTRUCTION_ADDRESS, '12343'), (Tokens.SPACING, ' '), 
        (Tokens.OPCODE, 'together'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]
    
    assert tokenizer.tokenize(*strings) == result
    assert tokenizer.tokenize('\n'.join(strings)) == result

    tokenizer.DEFAULT_NEWLINE_TUPLE = old_newline


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_default_immediate(tokenizer, pickled):
    """A bunch of different positive and negative immediate values, all in different formats (bin, hex, etc.)"""
    tokenizer = _pickle(tokenizer, pickled)

    vals = [0, 1, 10, 1234, 1000000000000, 3 ** 2033]
    for val in vals:
        for mult in ['', '-']:
            for format in ['0b{0:b}', '0x{0:x}', '0o{0:o}', '{0:d}']:
                string = mult + format.format(val)
                assert tokenizer.tokenize("add rax, eax%s-10 [eax]" % string) == [
                    (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'), (Tokens.SPACING, ', '),
                    (Tokens.REGISTER, 'eax'), (Tokens.IMMEDIATE, string), (Tokens.IMMEDIATE, '-10'), (Tokens.SPACING, ' '),
                    (Tokens.OPEN_BRACKET, '['), (Tokens.REGISTER, 'eax'), (Tokens.CLOSE_BRACKET, ']'), tokenizer.DEFAULT_NEWLINE_TUPLE,
                ], "failed on %s" % {'val': val, 'mult': mult, 'format': format}


@pytest.mark.parametrize('flip', [True, False])
@pytest.mark.parametrize('string,succeed', [
    ("''", True),
    ("'apples'", True),
    ("'weird1234 string   @%@&**^#$# << >> ,. <234>'", True),
    ("'\\'\"escapes\"\t\\'\\\\'", True),
    ('"', False),
    ('"forgot to close', False),
    ('"escaped \\"end\\"', False),
])
@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_default_string_literals(tokenizer, flip, string, succeed, pickled):
    """String literals are parsed correctly
    
    The parameterize holds all the strings that need to be parsed correctly.

    This will also test flipping all of the single and double quotes within the strings.
    """
    tokenizer = _pickle(tokenizer, pickled)

    if flip:
        string = ''.join([('"' if c == "'" else "'" if c == '"' else c) for c in string])
    
    if succeed:
        assert tokenizer.tokenize("add %s, 1234" % string) == [
            (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.STRING_LITERAL, string), (Tokens.SPACING, ', '), 
            (Tokens.IMMEDIATE, '1234'), tokenizer.DEFAULT_NEWLINE_TUPLE
        ], "Failed on flip=%s of string: %s" % (flip, repr(string))
    else:
        with pytest.raises(TokenMismatchError):
            tokenizer.tokenize("add %s, 1234" % string)


@pytest.mark.parametrize('string,succeed', [
    ('<>', True),
    ('<<<<<apples>', True),
    ('<nested <bra<ck>ets>>', True),
    ('<inner "string with\\" < and > inside\\\\">', True),
    ('<', False),
    ('<unmatched', False),
    ('>', False),
    ('<Multiple> end>', False),
])
@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_disassembler_info(tokenizer, string, succeed, pickled):
    """Disassembler info is parsed, including any substrings, and nested depth of 3 max"""
    tokenizer = _pickle(tokenizer, pickled)

    if succeed:
        assert tokenizer.tokenize("add rax, ecx%s" % string) == [
            (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'),  (Tokens.SPACING, ', '), 
            (Tokens.REGISTER, 'ecx'), (Tokens.DISASSEMBLER_INFO, string), tokenizer.DEFAULT_NEWLINE_TUPLE
        ]
    else:
        with pytest.raises(TokenMismatchError):
            tokenizer.tokenize("add rax, ecx%s" % string)


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_split_immediates(tokenizer, pickled):
    """Split immediate tokens, and their fails (which will fail silently)"""
    tokenizer = _pickle(tokenizer, pickled)

    # Normal splitting, we don't take up any extra immediates
    assert tokenizer.tokenize("add 376{split}123 456 rax 7532".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '376'), (Tokens.IMMEDIATE, '123456'), (Tokens.SPACING, ' '), 
        (Tokens.REGISTER, 'rax'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '7532'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]

    # Multiple split immediate tokens
    assert tokenizer.tokenize("add 376{split}123 456 rax {split}   \t7532 1234567".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '376'), (Tokens.IMMEDIATE, '123456'), (Tokens.SPACING, ' '), 
        (Tokens.REGISTER, 'rax'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '75321234567'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]

    # Single-token split immediates
    assert tokenizer.tokenize("add {split}123".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '123'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]

    # Broken split immediate tokens that don't have immediates, right next to each other, and at end of line
    assert tokenizer.tokenize("add {split}{split}123 123rax{split}   {split}".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '123123'), (Tokens.REGISTER, 'rax'), 
        (Tokens.SPACING, '   '), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]

    # Negatives and hex/bin/octal split immediates
    assert tokenizer.tokenize("add {split}-0x123 456 {split}0b10011010 0".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '-0x123456'), (Tokens.SPACING, ' '),
        (Tokens.IMMEDIATE, '0b100110100'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]


    # Split immediates as instruction address
    assert tokenizer.tokenize("{split}10: add".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        (Tokens.INSTRUCTION_ADDRESS, '10:'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'add'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]
    assert tokenizer.tokenize("{split}10 10 942 : add".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        (Tokens.INSTRUCTION_ADDRESS, '1010942:'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'add'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_symbols_and_spacing(tokenizer, pickled):
    """Symbols and spacing"""
    tokenizer = _pickle(tokenizer, pickled)

    assert tokenizer.tokenize("  \t\n||+rax*[123]]: :") == [
        (Tokens.SPACING, '  \t'), (Tokens.NEWLINE, '\n'), (Tokens.NEWLINE, '|'), (Tokens.NEWLINE, '|'), (Tokens.PLUS_SIGN, '+'),
        (Tokens.REGISTER, 'rax'), (Tokens.TIMES_SIGN, '*'), (Tokens.OPEN_BRACKET, '['), (Tokens.IMMEDIATE, '123'),
        (Tokens.CLOSE_BRACKET, ']'), (Tokens.CLOSE_BRACKET, ']'), (Tokens.COLON, ':'), (Tokens.SPACING, ' '),
        (Tokens.COLON, ':'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_instruction_address(tokenizer, pickled):
    """Instruction addresses with colon, assuming match_instruction_address=True (False is tested elsewhere)"""
    tokenizer = _pickle(tokenizer, pickled)

    # Plain instruction address
    assert tokenizer.tokenize("1234 add rax") == [
        (Tokens.INSTRUCTION_ADDRESS, '1234'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '),
        (Tokens.REGISTER, 'rax'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]

    # With colon and newline
    assert tokenizer.tokenize("\n  1234: add rax") == [
        (Tokens.NEWLINE, '\n'), (Tokens.SPACING, '  '), (Tokens.INSTRUCTION_ADDRESS, '1234:'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'add'), 
        (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]

    # With colon and spacing and another colon
    assert tokenizer.tokenize("  1234  \t:: add rax") == [
        (Tokens.SPACING, '  '), (Tokens.INSTRUCTION_ADDRESS, '1234:'), (Tokens.COLON, ':'), (Tokens.SPACING, ' '), 
        (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]

    # Don't let negatives be instruction addresses
    assert tokenizer.tokenize("-1234: add rax") == [
        (Tokens.IMMEDIATE, '-1234'), (Tokens.COLON, ':'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'add'), 
        (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]

    # Starting newline, but no address
    assert tokenizer.tokenize("\n :1234: add rax") == [
        (Tokens.NEWLINE, '\n'), (Tokens.SPACING, ' '), (Tokens.COLON, ':'), (Tokens.IMMEDIATE, '1234'), (Tokens.COLON, ':'), (Tokens.SPACING, ' '), 
        (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]

    # Split tokens as instruction address
    assert tokenizer.tokenize("{split}1234: add rax".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        (Tokens.INSTRUCTION_ADDRESS, '1234:'), (Tokens.SPACING, ' '), 
        (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]
    assert tokenizer.tokenize("{split}  1234  \t3 {split}: add rax".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        (Tokens.INSTRUCTION_ADDRESS, '12343:'), (Tokens.SPACING, ' '), 
        (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'), tokenizer.DEFAULT_NEWLINE_TUPLE
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_basic_tokenizer_builtin(tokenizer, pickled):
    """Builting functions that should just work, not gonna check their values right now"""
    tokenizer = _pickle(tokenizer, pickled)

    # __call__ should just be the same as `.tokenize()`, makes sure kwargs are sent, and pickling works with parameter saver
    assert tokenizer("add", match_instruction_address=False, newline_tup=(Tokens.NEWLINE, '\n'), kwargs_test='value') == [
        (Tokens.OPCODE, 'value'), (Tokens.NEWLINE, '\n')
    ]

    # Just want to call them and make sure they don't break
    assert isinstance(str(tokenizer), str)
    assert isinstance(repr(tokenizer), str)
    assert tokenizer == tokenizer
    assert isinstance(hash(tokenizer), int)

    # Pickling preseves equality
    assert tokenizer == pickle.loads(pickle.dumps(tokenizer))


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_empty_string(tokenizer, pickled):
    """Tokenize the empty string"""
    tokenizer = _pickle(tokenizer, pickled)

    assert tokenizer.tokenize("") == []
    assert tokenizer.tokenize() == []
    assert tokenizer.tokenize("{split}{split}".format(split=SPLIT_IMMEDIATE_TOKEN)) == []


def _equality_fake_func(state):
    return None
def _equality_fake_func2(state):
    return None
_equality_var = _equality_fake_func
_EQUALITY_TEST_TOKENIZERS = [
    ({}, 'a', 531367615942336365),
    ({}, 'a', 531367615942336365),
    ({'tokens': FakeTokenizer.DEFAULT_TOKENS + [('fakefake', 'fake')]}, 'b', 1122623343755692881),
    ({'case_sensitive': True}, 'c', 158729947333387147), 
    ({'case_sensitive': True}, 'c', 158729947333387147), 
    ({'token_handlers': {Tokens.OPCODE: _equality_fake_func2}}, 'd', 678502768402453625),
    ({'token_handlers': {Tokens.OPCODE: _equality_fake_func}}, 'd', 678502768402453625),
    ({'token_handlers': {Tokens.OPCODE: _equality_var}}, 'd', 678502768402453625),
]
@pytest.mark.parametrize('t2,v2,h2', _EQUALITY_TEST_TOKENIZERS)
@pytest.mark.parametrize('t1,v1,h1', _EQUALITY_TEST_TOKENIZERS)
def test_example_tokenizer_eq_and_hash(t1, v1, h1, t2, v2, h2, print_hashes):
    """Equality and whatnot"""
    if not isinstance(t1, BaseTokenizer):
        t1 = FakeTokenizer(**t1)
    if not isinstance(t2, BaseTokenizer):
        t2 = FakeTokenizer(**t2)

    if print_hashes:
        print(__file__, v1, hash(t1), v2, hash(t2))
    else:  # Don't test while printing
        assert hash(t1) == h1
        assert hash(t2) == h2

    if v1 == v2:
        assert t1 == t2, "failed %s == %s" % (v1, v2)
        assert t2 == t1, "failed %s == %s" % (v1, v2)
        assert hash(t1) == hash(t2), "failed hash %s == %s" % (v1, v2)
        assert pickle.loads(pickle.dumps(t1)) == t2
        assert pickle.loads(pickle.dumps(t2)) == t1
    else:
        assert t1 != t2, "failed %s != %s" % (v1, v2)
        assert t2 != t1, "failed %s != %s" % (v1, v2)
        assert hash(t1) != hash(t2), "failed hash %s != %s" % (v1, v2)
        assert pickle.loads(pickle.dumps(t1)) != t2
        assert pickle.loads(pickle.dumps(t2)) != t1
