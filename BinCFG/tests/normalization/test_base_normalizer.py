"""Testing BaseNormalizer"""

import pytest
import pickle
import hashlib
from bincfg import BaseTokenizer, BaseNormalizer, Tokens, TokenizationLevel
from bincfg.normalization.norm_utils import INSTRUCTION_START_TOKEN, SPLIT_IMMEDIATE_TOKEN


# Whether or not to pickle the tokenizers to make sure pickling doesn't have any affect on outcome
def _pickle(normalizer, pickled):
    if pickled == 'pickled':
        return pickle.loads(pickle.dumps(normalizer))
    return normalizer


class FakeTokenizer(BaseTokenizer):
    """Same FakeTokenizer from test_base_tokenizer.py, but I don't wanna import it for minor changes between them"""

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
            return state['kwargs']['test_mismatch']
        super().token_mismatch(state)


class FakeNormalizer(BaseNormalizer):
    DEFAULT_TOKENIZATION_LEVEL = TokenizationLevel.INSTRUCTION

    def __init__(self, tokenizer, repl_all=False, token_handlers=None, token_sep=' ', tokenization_level=TokenizationLevel.AUTO, anonymize_tokens=False):
        super().__init__(tokenizer=tokenizer, token_handlers=token_handlers, token_sep=token_sep, 
                         tokenization_level=tokenization_level, anonymize_tokens=anonymize_tokens)
        
        # Replace all the handlers with something that just returns <token_type> if repl_all=True
        if repl_all:
            for attr in dir(self):
                if attr.startswith('handle_'):
                    setattr(self, attr, lambda self, state: "<%s>" % state.token_type)
    
    def handle_branch_prediction(self, state):
        return 'bp'

    def handle_mismatch(self, state):
        if 'test_mismatch' in state['kwargs']:
            return state['token']
        super().token_mismatch(state)


@pytest.fixture(scope='module')
def n():
    return FakeNormalizer(tokenizer=FakeTokenizer())


def _handle_opcode_init_norm(state):
    return state.token + "HANDLED"
@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_init_normalizer(pickled):
    """Test passing different kwargs to the normalizer init function"""

    n = FakeNormalizer(FakeTokenizer(), token_handlers={Tokens.OPCODE: _handle_opcode_init_norm}, token_sep='_', 
                       tokenization_level='opcode', anonymize_tokens=False)
    
    n = _pickle(n, pickled)

    assert n.normalize("add 10") == [INSTRUCTION_START_TOKEN, 'addhandled', '10']


def _token_mismatch_override_handle_functions(state):
    return state['token']
def _token_unknown_override_handle_functions(state):
    return state['token']
def _token_opcode_override_handle_functions(state):
    return 'OPCODE'
@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_override_handle_functions(pickled):
    """Overriding the built-in handling functions"""
    tokenizer = FakeTokenizer(tokens=[('fake', r'\(\)')] + FakeTokenizer.DEFAULT_TOKENS)
    tokenizer.token_mismatch = _token_mismatch_override_handle_functions
    tokenizer.token_unknown = _token_unknown_override_handle_functions
    n = FakeNormalizer(tokenizer, repl_all=True, token_handlers={Tokens.OPCODE: _token_opcode_override_handle_functions})
    
    n = _pickle(n, pickled)

    assert n.normalize("1234: add10+*[]:eax-0x33{split}\n<disinfo>\"str\\\"ing\"&()".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        "<inst_addr> <spacing> opcode <plus_sign> <times_sign> <open_bracket> <close_bracket> <colon> <register> <immediate> <newline>",
        "<disassembler_info> <string_literal> <mismatch> <fake> <newline>"
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_pass_kwargs_to_tokenizer(n, pickled):
    """Kwargs should be passed to tokenizer on call"""
    n = _pickle(n, pickled)

    assert n.normalize("1234 add 0b1010", match_instruction_address=False, kwargs_test="value") == [
        '1234 value 10'
    ]


def _setad_roh_test(state):
    state['line'][state['token_idx']] = (Tokens.OPCODE, 'ADDTOKEN', 'ADDTOKEN')
@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_register_opcode_handlers(pickled):
    """Registering opcode handlers"""
    n = FakeNormalizer(FakeTokenizer())
    n.register_opcode_handler(r'ad+', _setad_roh_test)

    n = _pickle(n, pickled)

    assert n.normalize("add 10 ad call rax addddd") == [
        "addtoken 10 addtoken call rax addtoken"
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_split_immediates(n, pickled):
    """Split immediates are parsed correctly"""
    n = _pickle(n, pickled)

    assert n.normalize("add {split}10, rax {split}0b10 10{split}".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        "add 10 rax 10"
    ]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_normalize_empty_string(pickled):
    """Empty string and no strings for both tokenization values"""
    n = FakeNormalizer(FakeTokenizer(), tokenization_level='inst')
    n = _pickle(n, pickled)
    assert n.normalize("") == [""], 'inst empty string'
    assert n.normalize() == [], 'inst no strings'
    n = FakeNormalizer(FakeTokenizer(), tokenization_level='op')
    n = _pickle(n, pickled)
    assert n.normalize("") == [""], 'op empty string'
    assert n.normalize() == [], 'op no strings'


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_builtin_tokens(n, pickled):
    """Tests basic builtin tokens"""
    n = _pickle(n, pickled)
    assert n.normalize("10: add -0xb rax []*+:  \t'str'<remove>") == [
        "add -11 rax [ ] * + : \"str\""
    ]

    assert n.normalize("test \"string\\\"\" and 'strings\"'") == ["test \"string\\\"\" and \"strings\\\"\""]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_split_immediate(n, pickled):
    """Tests split immediates are normalized correctly"""
    n = _pickle(n, pickled)
    assert n.normalize("{split}10 10:add {split}10  \t20 {split}".format(split=SPLIT_IMMEDIATE_TOKEN)) == ["add 1020"]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_empty_string(n, pickled):
    """Tests empty strings are handled correctly"""
    n = _pickle(n, pickled)
    assert n.normalize("") == [""]
    assert n.normalize() == []
    assert n.normalize("{split}{split}".format(split=SPLIT_IMMEDIATE_TOKEN)) == [""]


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_immediates_converted_to_decimal(n, pickled):
    """Tests empty strings are handled correctly"""
    n = _pickle(n, pickled)

    vals = [0, 1, 10, 1234, 1000000000000, 3 ** 2033]
    for val in vals:
        for mult in ['', '-']:
            for format in ['0b{0:b}', '0x{0:x}', '0o{0:o}', '{0:d}']:
                assert n.normalize("1234: add %s rax" % (mult + format.format(val))) == [
                    "add %d rax" % (val * (-1 if mult == '-' else 1))
                ], "%d, %s, %s" % (val, repr(mult), repr(format))


@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_builtin_methods(n, pickled):
    """Builtin dunder methods"""
    n = _pickle(n, pickled)

    assert isinstance(hash(n), int)
    assert isinstance(str(n), str)
    assert isinstance(repr(n), str)

    # __call__ passes all needed kwargs
    assert n(INSTRUCTION_START_TOKEN, "1234: add {split}10 340, rax {inst} mov rax 13 & <disinfo> \"apples\"".format(split=SPLIT_IMMEDIATE_TOKEN, inst=INSTRUCTION_START_TOKEN), 
             "{inst}add".format(inst=INSTRUCTION_START_TOKEN), cfg=None, block=None, match_instruction_address=False, test_mismatch='$') == \
        ["1234 : add 10340 rax", "mov rax 13 $ \"apples\"", "add"]
    

@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_disassembler_info(n, pickled):
    """Disassembler information is parsed correcting"""
    n = _pickle(n, pickled)

    # Immediates are inserted correctly
    assert n.normalize("1234: add rax, 10<11>") == ['add rax 11']
    assert n.normalize("1234: add rax, 10<-0x11'aa'>") == ['add rax -17']
    assert n.normalize("1234: add rax, 10<-0x11 asdfsadj>") == ['add rax -17']
    assert n.normalize("1234: add rax, 10<-0b11><-4>") == ['add rax -4']

    # Strings are inserted correctly
    assert n.normalize("1234: add rax, 10<'aa'>") == ['add rax 10 \"aa\"']
    assert n.normalize("1234: add rax, 10<\"aa\">") == ['add rax 10 \"aa\"']
    assert n.normalize("1234: add rax, 10<\"aa\" asdfuiasdnj>") == ['add rax 10 \"aa\"']
    assert n.normalize("1234: add rax, 10<\"aa\"12>") == ['add rax 10 \"aa\"']

    # Json is parsed and handled
    assert n.normalize("1234: add rax<{\"who knows\": 10, \"a\": [1, 2, 3]}><>, 10<{\"immediate\": 3}>") == ['add rax 3']

    # Json 'insert' and 'insert_type'
    assert n.normalize("1234: add rax, 10<{\"insert\": 17}>") == ['add rax 10 17']
    assert n.normalize("1234: add rax, 10<{\"insert\": 17, \"insert_type\": \"opcode\"}>") == ['add rax 10 17']
    assert n.normalize("1234: add rax, 10<{\"insert\": \"17\"}>") == ['add rax 10 17']
    assert n.normalize("1234: add rax, 10<{\"insert\": \"\\\"17\\\"\"}>") == ['add rax 10 \"17\"']
    assert n.normalize("1234: add rax, 10<{\"insert\": 17, \"insert_type\": \"disassembler_info\"}>") == ['add rax 10']
    assert n.normalize("1234: add rax, 10<{\"insert\": \"17\", \"insert_type\": \"branch_prediction\"}>") == ['add rax 10 bp']
    assert n.normalize("1234: add rax, 10<{\"insert\": \"17\", \"insert_type\": false}>") == ['add rax 10 17']
    assert n.normalize("1234: add rax, 10<{\"insert\": \"&&&\", \"insert_type\": false}>") == ['add rax 10 &&&']
    assert n.normalize("1234: add rax, 10<{\"insert\": \"'&&&'\"}>") == ['add rax 10 "&&&"']

    # Fails if it's unknown and you passed raise_unk_di=True
    with pytest.raises(ValueError):
        n.normalize("1234: add 1234<unknown>", raise_unk_di=True)


def _test_hash(string):
    hasher = hashlib.shake_128()
    hasher.update(string.encode('utf-8'))
    return hasher.hexdigest(4)
@pytest.mark.parametrize('pickled', ['normal', 'pickled'])
def test_anonymize_tokens(n, pickled):
    """Tests the anonymize tokens does hashing right"""
    n = FakeNormalizer(FakeTokenizer(), anonymize_tokens=True)

    assert n.normalize("1234: add rax, 10<'aa'>") == [_test_hash('add rax 10 "aa"')]
    assert n.normalize("10: add -0xb rax []*+:  \t'str'<remove>") == [_test_hash("add -11 rax [ ] * + : \"str\"")], FakeNormalizer(FakeTokenizer(), anonymize_tokens=False).normalize("10: add -0xb rax []*+:  \t'str'<remove>")
    assert n.normalize("1234 add 0b1010", match_instruction_address=False, kwargs_test="value") == [_test_hash('1234 value 10')]

    tokenizer = FakeTokenizer(tokens=[('fake', r'\(\)')] + FakeTokenizer.DEFAULT_TOKENS)
    tokenizer.token_mismatch = _token_mismatch_override_handle_functions
    tokenizer.token_unknown = _token_unknown_override_handle_functions
    n = FakeNormalizer(tokenizer, repl_all=True, token_handlers={Tokens.OPCODE: _token_opcode_override_handle_functions}, anonymize_tokens=True)
    
    n = _pickle(n, pickled)

    assert n.normalize("1234: add10+*[]:eax-0x33{split}\n<disinfo>\"str\\\"ing\"&()".format(split=SPLIT_IMMEDIATE_TOKEN)) == [
        _test_hash("<inst_addr> <spacing> opcode <plus_sign> <times_sign> <open_bracket> <close_bracket> <colon> <register> <immediate> <newline>"),
        _test_hash("<disassembler_info> <string_literal> <mismatch> <fake> <newline>")
    ]


def _equality_fake_func(state):
    return None
def _equality_fake_func2(state):
    return None
def _equality_fake_func3(state):
    return state.token
_equality_var = _equality_fake_func
_EQUALITY_TEST_TOKENIZERS = [
    ({}, {}, 'a', 150203515121960375),
    ({}, {}, 'a', 150203515121960375),
    ({'tokens': FakeTokenizer.DEFAULT_TOKENS + [('fakefake', 'fake')]}, {}, 'b', 1882530265114452202),
    ({'tokens': FakeTokenizer.DEFAULT_TOKENS + [('fakefake', 'fake')]}, {'tokenization_level': TokenizationLevel.INSTRUCTION}, 'b', 1882530265114452202),
    ({'tokens': FakeTokenizer.DEFAULT_TOKENS + [('fakefake', 'fake')]}, {'tokenization_level': TokenizationLevel.OPCODE}, 'c', 1441071216850926297),
    ({'case_sensitive': True}, {'repl_all': True}, 'd', 1121705704022107297), 
    ({'case_sensitive': True}, {'repl_all': True}, 'd', 1121705704022107297), 
    ({'token_handlers': {Tokens.OPCODE: _equality_fake_func2}}, {'anonymize_tokens': True}, 'e', 2078370539328523636),
    ({'token_handlers': {Tokens.OPCODE: _equality_fake_func}}, {'anonymize_tokens': True, 'token_handlers': {Tokens.OPCODE: _equality_fake_func}}, 'f', 2244523516573442261),
    ({'token_handlers': {Tokens.OPCODE: _equality_var}}, {'anonymize_tokens': True, 'token_handlers': {Tokens.OPCODE: _equality_var}}, 'f', 2244523516573442261),
    ({'token_handlers': {Tokens.OPCODE: _equality_var}}, {'anonymize_tokens': True, 'token_handlers': {Tokens.OPCODE: _equality_fake_func2}}, 'f', 2244523516573442261),
    ({'token_handlers': {Tokens.OPCODE: _equality_var}}, {'anonymize_tokens': True, 'token_handlers': {Tokens.OPCODE: _equality_fake_func3}}, 'g', 330360741807403499),
    ({'case_sensitive': True}, {'repl_all': True, 'token_sep': '_'}, 'h', 129344844194678889), 
    ({'case_sensitive': True}, {'repl_all': True, 'token_sep': ''.join(['_'])}, 'h', 129344844194678889), 
]
@pytest.mark.parametrize('t2,n2,v2,h2', _EQUALITY_TEST_TOKENIZERS)
@pytest.mark.parametrize('t1,n1,v1,h1', _EQUALITY_TEST_TOKENIZERS)
def test_example_tokenizer_eq_and_hash(t1, n1, v1, h1, t2, n2, v2, h2, print_hashes):
    """Equality and whatnot"""
    t1 = FakeTokenizer(**t1)
    t2 = FakeTokenizer(**t2)
    n1 = FakeNormalizer(t1, **n1)
    n2 = FakeNormalizer(t2, **n2)

    if print_hashes:
        print(__file__, v1, hash(n1), v2, hash(n2))
    else:  # Don't test while printing
        assert hash(n1) == h1
        assert hash(n2) == h2

    if v1 == v2:
        assert n1 == n2, "failed norm %s == %s" % (v1, v2)
        assert n2 == n1, "failed norm %s == %s" % (v1, v2)
        assert hash(n1) == hash(n2), "failed norm hash %s == %s" % (v1, v2)
        assert pickle.loads(pickle.dumps(n1)) == n2
        assert pickle.loads(pickle.dumps(n2)) == n1
    else:
        assert n1 != n2, "failed norm %s != %s, %s" % (v1, v2)
        assert n2 != n1, "failed norm %s != %s" % (v1, v2)
        assert hash(n1) != hash(n2), "failed norm hash %s != %s" % (v1, v2)
        assert pickle.loads(pickle.dumps(n1)) != n2
        assert pickle.loads(pickle.dumps(n2)) != n1
