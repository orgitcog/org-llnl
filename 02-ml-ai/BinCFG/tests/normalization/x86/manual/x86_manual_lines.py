import math
from bincfg import Tokens, TokenMismatchError
from bincfg.normalization import *
from bincfg.normalization.base_normalizer import DEFAULT_IMMEDIATE_THRESHOLD
from bincfg.normalization.norm_utils import *


# Name, class, kwargs
X86_TEST_OBJECTS = [
    ('x86_base_tokenizer', X86BaseTokenizer, {}),
    ('x86_base', X86BaseNormalizer, {}),
    ('x86_innereye', X86InnerEyeNormalizer, {}),
    ('x86_deepbindiff', X86DeepBinDiffNormalizer, {}),
    ('x86_safe', X86SafeNormalizer, {}),
    ('x86_deepsemantic', X86DeepSemanticNormalizer, {}),
    ('x86_hpc', X86HPCDataNormalizer, {}),
    ('x86_hpc_repl', X86HPCDataNormalizer, {'num_digits': 3, 'replace_strings': True}),
]


def _hpc_norm_nums(nums, num_digits=4):
    """Puts the numbers in hpc format (with num_digits=4), spaces inbetween them, starting with split_imm token if needed"""
    ret = ''
    prev_split = False
    for num in nums:
        s = str(num)
        if len(s) > num_digits or prev_split:
            ret += SPLIT_IMMEDIATE_TOKEN + ' '
            prev_split = True
        else:
            prev_split = False
        
        ret += ' '.join([s[i*num_digits: (i+1)*num_digits] for i in range(math.ceil(len(s) / num_digits))]) + ' '
    
    return ret.strip()


# A bunch of test inputs for tokenizers and normalizers. Each input is a dictionary with an 'input' key for the raw
#   string input, and multiple other keys, one for each class name for each tokenizer/normalizer we wish to test and
#   values being the expected outputs. Each normalizer output should be in the 'instruction-level' tokenization, even
#   though both instruction and op-level tokenizations will be tested. Normalizer/tokenizer outputs can optionally be 
#   exception classes, in which case it is assumed that test should raise that type (or a subclass of that type) of error
# Each value should also have a 'bad_assemly' key that is True if the input should be tokenized, but should fail the
#   correct assembly checks, and False if it should pass both
# Contant value strings can be inserted by using special keywords: {immval}, {func}, {self}, {innerfunc}, {externfunc},
#   {jmpdst}, {memexpr}, {reg}
X86_TEST_INPUTS = [


    ###############################
    # Actual assembly line inputs #
    ###############################


    {
        'input': '0x00402cc8: sub    rsp, 0x08',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00402cc8:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'sub'), (Tokens.SPACING, '    '), 
            (Tokens.REGISTER, 'rsp'), (Tokens.SPACING, ', '), 
            (Tokens.IMMEDIATE, '0x08'), 
            (Tokens.NEWLINE, '\n')
        ],
        'x86_base': ['sub rsp 8'],
        'x86_innereye': ['sub rsp {immval}'],
        'x86_deepbindiff': ['sub rsp {immval}'],
        'x86_safe': ['sub rsp 8'],
        'x86_deepsemantic': ['sub rsp {immval}'],
        'x86_hpc': ['sub rsp 8'],
        'x86_hpc_repl': ['sub rsp 8'],
    },

    {
        'input': '0x00402ccc: mov    rax, qword ds:[rip + 0x000000000025230d<absolute=0x0000000000654fe0>]',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00402ccc:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'mov'), (Tokens.SPACING, '    '), 
            (Tokens.REGISTER, 'rax'), (Tokens.SPACING, ', '), 
            (Tokens.MEMORY_SIZE, 'qword'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'ds'), (Tokens.COLON, ':'),
            (Tokens.OPEN_BRACKET, '['), (Tokens.REGISTER, 'rip'), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'), 
                (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x000000000025230d'), (Tokens.DISASSEMBLER_INFO, '<absolute=0x0000000000654fe0>'), 
            (Tokens.CLOSE_BRACKET, ']'), (Tokens.NEWLINE, '\n')
        ],
        'x86_base': ['mov rax qword ds: [ rip + 2433805 ]'],
        'x86_innereye': ['mov rax [ rip + {immval} ]'],
        'x86_deepbindiff': ['mov {reg}8 {memexpr}'],
        'x86_safe': ['mov rax ds: [ rip + {dispmem} ]'],
        'x86_deepsemantic': ['mov {reg}8 {memptr}8 ds: [ rip + {immval} ]'],
        'x86_hpc': ['mov rax qword ds: [ rip + {split_imm} 2433 805 ]'],
        'x86_hpc_repl': ['mov rax qword ds: [ rip + {split_imm} 243 380 5 ]'],
    },

    {
        'input': '0x00404373: mov    edi, 0x0043c066<"No %s section \'present\\n\\n">',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00404373:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'mov'), (Tokens.SPACING, '    '), 
            (Tokens.REGISTER, 'edi'), (Tokens.SPACING, ', '), 
            (Tokens.IMMEDIATE, '0x0043c066'), (Tokens.DISASSEMBLER_INFO, '<"No %s section \'present\\n\\n">'), 
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['mov edi 4440166 "No %s section \'present\\n\\n"'],
        'x86_innereye': ['mov edi {immval} {str}'],
        'x86_deepbindiff': ['mov {reg}4 {immval} "No %s section \'present\\n\\n"'],
        'x86_safe': ['mov edi {immval}'],
        'x86_deepsemantic': ['mov {reg}4 {immval} {str}'],
        'x86_hpc': ['mov edi {split_imm} 4440 166 "No %s section \'present\\n\\n"'],
        'x86_hpc_repl': ['mov edi {str}'],
    },

    {
        'input': '0x00404373: mov    edi, 0x0043c066<\'re-escape with "non-json\\n\\n\'>',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00404373:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'mov'), (Tokens.SPACING, '    '), 
            (Tokens.REGISTER, 'edi'), (Tokens.SPACING, ', '), 
            (Tokens.IMMEDIATE, '0x0043c066'), (Tokens.DISASSEMBLER_INFO, '<\'re-escape with "non-json\\n\\n\'>'), 
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['mov edi 4440166 "re-escape with \\"non-json\\n\\n"'],
        'x86_innereye': ['mov edi {immval} {str}'],
        'x86_deepbindiff': ['mov {reg}4 {immval} "re-escape with \\"non-json\\n\\n"'],
        'x86_safe': ['mov edi {immval}'],
        'x86_deepsemantic': ['mov {reg}4 {immval} {str}'],
        'x86_hpc': ['mov edi {split_imm} 4440 166 "re-escape with \\"non-json\\n\\n"'],
        'x86_hpc_repl': ['mov edi {str}'],
    },

    {
        'input': '0x00404373: mov    edi, 0x0043c066<{"insert": "\\\"test\\n\\n\\\""}>',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00404373:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'mov'), (Tokens.SPACING, '    '), 
            (Tokens.REGISTER, 'edi'), (Tokens.SPACING, ', '), 
            (Tokens.IMMEDIATE, '0x0043c066'), (Tokens.DISASSEMBLER_INFO, '<{"insert": "\\\"test\\n\\n\\\""}>'), 
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['mov edi 4440166 "test\\n\\n"'],
        'x86_innereye': ['mov edi {immval} {str}'],
        'x86_deepbindiff': ['mov {reg}4 {immval} "test\\n\\n"'],
        'x86_safe': ['mov edi {immval}'],
        'x86_deepsemantic': ['mov {reg}4 {immval} {str}'],
        'x86_hpc': ['mov edi {split_imm} 4440 166 "test\\n\\n"'],
        'x86_hpc_repl': ['mov edi {str}'],
    },

    {
        'input': '0x00404373: mov    edi, 0x0043c066<{"insert": "teSt\\n\\n", "insert_type": "string_literal"}>',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00404373:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'mov'), (Tokens.SPACING, '    '), 
            (Tokens.REGISTER, 'edi'), (Tokens.SPACING, ', '), 
            (Tokens.IMMEDIATE, '0x0043c066'), (Tokens.DISASSEMBLER_INFO, '<{"insert": "teSt\\n\\n", "insert_type": "string_literal"}>'), 
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['mov edi 4440166 "teSt\\n\\n"'],
        'x86_innereye': ['mov edi {immval} {str}'],
        'x86_deepbindiff': ['mov {reg}4 {immval} "teSt\\n\\n"'],
        'x86_safe': ['mov edi {immval}'],
        'x86_deepsemantic': ['mov {reg}4 {immval} {str}'],
        'x86_hpc': ['mov edi {split_imm} 4440 166 "teSt\\n\\n"'],
        'x86_hpc_repl': ['mov edi {str}'],
    },

    {
        'input': '0x004020f1: jmp    qword ds:[0x0000000000404520"\\"!@" + rax*0x08]',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x004020f1:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'jmp'), (Tokens.SPACING, '    '), (Tokens.MEMORY_SIZE, 'qword'),
            (Tokens.SPACING, ' '), (Tokens.REGISTER, 'ds'), (Tokens.COLON, ':'),
            (Tokens.OPEN_BRACKET, '['), (Tokens.IMMEDIATE, '0x0000000000404520'),
            (Tokens.STRING_LITERAL, '"\\"!@"'), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'),
            (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'), (Tokens.TIMES_SIGN, '*'),
            (Tokens.IMMEDIATE, '0x08'), (Tokens.CLOSE_BRACKET, ']'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['jmp qword ds: [ 4212000 "\\"!@" + rax * 8 ]'],
        'x86_innereye': ['jmp [ {immval} {str} + rax * {immval} ]'],
        'x86_deepbindiff': ['jmp {memexpr}'],
        'x86_safe': ['jmp ds: [ {immval} + rax * 8 ]'],
        'x86_deepsemantic': ['jmp {memptr}8 ds: [ {immval} {str} + {reg}8 * 8 ]'],
        'x86_hpc': ['jmp qword ds: [ {split_imm} 4212 000 "\\"!@" + rax * 8 ]'],
        'x86_hpc_repl': ['jmp qword ds: [ {str} + rax * 8 ]'],
    },

    {
        'input': '0x004020f1: jmp    qword ds:[0x0000000000404520<\'\\"!@\'> + rax*0x08]',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x004020f1:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'jmp'), (Tokens.SPACING, '    '), (Tokens.MEMORY_SIZE, 'qword'),
            (Tokens.SPACING, ' '), (Tokens.REGISTER, 'ds'), (Tokens.COLON, ':'),
            (Tokens.OPEN_BRACKET, '['), (Tokens.IMMEDIATE, '0x0000000000404520'),
            (Tokens.DISASSEMBLER_INFO, '<\'\\"!@\'>'), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'),
            (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'), (Tokens.TIMES_SIGN, '*'),
            (Tokens.IMMEDIATE, '0x08'), (Tokens.CLOSE_BRACKET, ']'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['jmp qword ds: [ 4212000 "\\"!@" + rax * 8 ]'],
        'x86_innereye': ['jmp [ {immval} {str} + rax * {immval} ]'],
        'x86_deepbindiff': ['jmp {memexpr}'],
        'x86_safe': ['jmp ds: [ {immval} + rax * 8 ]'],
        'x86_deepsemantic': ['jmp {memptr}8 ds: [ {immval} {str} + {reg}8 * 8 ]'],
        'x86_hpc': ['jmp qword ds: [ {split_imm} 4212 000 "\\"!@" + rax * 8 ]'],
        'x86_hpc_repl': ['jmp qword ds: [ {str} + rax * 8 ]'],
    },

    {
        'input': '0x004020f1: jmp    qword ds:[0x0000000000404520  \'\\"\\\' \\\'!@\' + rax*0x08]',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x004020f1:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'jmp'), (Tokens.SPACING, '    '), (Tokens.MEMORY_SIZE, 'qword'),
            (Tokens.SPACING, ' '), (Tokens.REGISTER, 'ds'), (Tokens.COLON, ':'),
            (Tokens.OPEN_BRACKET, '['), (Tokens.IMMEDIATE, '0x0000000000404520'), (Tokens.SPACING, '  '),
            (Tokens.STRING_LITERAL, '\'\\"\\\' \\\'!@\''), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'),
            (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rax'), (Tokens.TIMES_SIGN, '*'),
            (Tokens.IMMEDIATE, '0x08'), (Tokens.CLOSE_BRACKET, ']'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['jmp qword ds: [ 4212000 "\\"\' \'!@" + rax * 8 ]'],
        'x86_innereye': ['jmp [ {immval} {str} + rax * {immval} ]'],
        'x86_deepbindiff': ['jmp {memexpr}'],
        'x86_safe': ['jmp ds: [ {immval} + rax * 8 ]'],
        'x86_deepsemantic': ['jmp {memptr}8 ds: [ {immval} {str} + {reg}8 * 8 ]'],
        'x86_hpc': ['jmp qword ds: [ {split_imm} 4212 000 "\\"\' \'!@" + rax * 8 ]'],
        'x86_hpc_repl': ['jmp qword ds: [ {str} + rax * 8 ]'],
    },

    {
        'input': '0x00402cd3: test   rax, rax',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00402cd3:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'test'), (Tokens.SPACING, '   '), 
            (Tokens.REGISTER, 'rax'), (Tokens.SPACING, ', '), (Tokens.REGISTER, 'rax'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['test rax rax'],
        'x86_innereye': ['test rax rax'],
        'x86_deepbindiff': ['test {reg}8 {reg}8'],
        'x86_safe': ['test rax rax'],
        'x86_deepsemantic': ['test {reg}8 {reg}8'],
        'x86_hpc': ['test rax rax'],
        'x86_hpc_repl': ['test rax rax'],
    },

    {
        'input': '0x00402cd3: nop    rax, twoRd CS:[rip+eax*0x04+0x000032]',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00402cd3:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'nop'), (Tokens.SPACING, '    '), 
            (Tokens.REGISTER, 'rax'), (Tokens.SPACING, ', '), 
            (Tokens.MEMORY_SIZE, 'twoRd'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'CS'), (Tokens.COLON, ':'),
            (Tokens.OPEN_BRACKET, '['), (Tokens.REGISTER, 'rip'), (Tokens.PLUS_SIGN, '+'), (Tokens.REGISTER, 'eax'), 
                (Tokens.TIMES_SIGN, '*'), (Tokens.IMMEDIATE, '0x04'), (Tokens.PLUS_SIGN, '+'), (Tokens.IMMEDIATE, '0x000032'),
                (Tokens.CLOSE_BRACKET, ']'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['nop'],
        'x86_innereye': ['nop'],
        'x86_deepbindiff': ['nop'],
        'x86_safe': ['nop'],
        'x86_deepsemantic': ['nop'],
        'x86_hpc': ['nop'],
        'x86_hpc_repl': ['nop'],
    },

    {
        'input': '0x00402cd6: je     0x0000000000402cdd',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00402cd6:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'je'), (Tokens.SPACING, '     '), 
            (Tokens.IMMEDIATE, '0x0000000000402cdd'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['je 4205789'],
        'x86_innereye': ['je {immval}'],
        'x86_deepbindiff': ['je {immval}'],
        'x86_safe': ['je {immval}'],
        'x86_deepsemantic': ['je {jmpdst}'],
        'x86_hpc': ['je {split_imm} 4205 789'],
        'x86_hpc_repl': ['je {split_imm} 420 578 9'],
    },

    {
        'input': '0x00402cd8: call   0x0000000000403170',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00402cd8:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'call'), (Tokens.SPACING, '   '), 
            (Tokens.IMMEDIATE, '0x0000000000403170'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['call 4206960'],
        'x86_innereye': ['call {func}'],
        'x86_deepbindiff': ['call {immval}'],
        'x86_safe': ['call {immval}'],
        'x86_deepsemantic': ['call {func}'],
        'x86_hpc': ['call {split_imm} 4206 960'],
        'x86_hpc_repl': ['call {split_imm} 420 696 0'],
    },

    {
        'input': '0x00402cdd: add    rsp, 0x08',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00402cdd:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'add'), (Tokens.SPACING, '    '), 
            (Tokens.REGISTER, 'rsp'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x08'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['add rsp 8'],
        'x86_innereye': ['add rsp {immval}'],
        'x86_deepbindiff': ['add rsp {immval}'],
        'x86_safe': ['add rsp 8'],
        'x86_deepsemantic': ['add rsp {immval}'],
        'x86_hpc': ['add rsp 8'],
        'x86_hpc_repl': ['add rsp 8'],
    },

    {
        'input': '0x00402ce1: ret',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00402ce1:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'ret'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['ret'],
        'x86_innereye': ['ret'],
        'x86_deepbindiff': ['ret'],
        'x86_safe': ['ret'],
        'x86_deepsemantic': ['ret'],
        'x86_hpc': ['ret'],
        'x86_hpc_repl': ['ret'],
    },

    {
        'input': """0x00402cf0: puSh   QWORD ds:,[rip + 0x0000000000252312<absolute=0x0000000000655008>]'\\\\\\"!\\'©
\t\\t\n@' <"\\\\\\"!\'©\t\\t\n@\">""",
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00402cf0:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'puSh'), (Tokens.SPACING, '   '),
            (Tokens.MEMORY_SIZE, 'QWORD'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'ds'), (Tokens.COLON, ':'), (Tokens.SPACING, ','),
            (Tokens.OPEN_BRACKET, '['), (Tokens.REGISTER, 'rip'), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'), 
                (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x0000000000252312'), (Tokens.DISASSEMBLER_INFO, '<absolute=0x0000000000655008>'), 
            (Tokens.CLOSE_BRACKET, ']'), (Tokens.STRING_LITERAL, """'\\\\\\"!\\'©
\t\\t\n@'"""), (Tokens.SPACING, ' '), (Tokens.DISASSEMBLER_INFO, "<\"\\\\\\\"!'©\t\\t\n@\">"),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['push qword ds: [ rip + 2433810 ] "\\\\\\"!\'\\xc2\\xa9\\n\\t\\t\\n@" "\\\\\\"!\'\\xc2\\xa9\\t\\t\\n@"'],
        'x86_innereye': ['push [ rip + {immval} ] {str} {str}'],
        'x86_deepbindiff': ['push {memexpr} "\\\\\\"!\'\\xc2\\xa9\\n\\t\\t\\n@" "\\\\\\"!\'\\xc2\\xa9\\t\\t\\n@"'],
        'x86_safe': ['push ds: [ rip + {dispmem} ]'],
        'x86_deepsemantic': ['push {memptr}8 ds: [ rip + {immval} ] {str} {str}'],
        'x86_hpc': ['push qword ds: [ rip + {split_imm} 2433 810 ] "\\\\\\"!\'\\xc2\\xa9\\n\\t\\t\\n@" "\\\\\\"!\'\\xc2\\xa9\\t\\t\\n@"'],
        'x86_hpc_repl': ['push qword ds: [ rip + {split_imm} 243 381 0 ] {str} {str}'],
    },

    {
        'input': '0x00402cf6: jmp    qword ds:[RIP+0x0000000000252314<absolute=0x0000000000655010>]',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00402cf6:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'jmp'), (Tokens.SPACING, '    '),
            (Tokens.MEMORY_SIZE, 'qword'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'ds'), (Tokens.COLON, ':'),
            (Tokens.OPEN_BRACKET, '['), (Tokens.REGISTER, 'RIP'), (Tokens.PLUS_SIGN, '+'), 
                (Tokens.IMMEDIATE, '0x0000000000252314'), (Tokens.DISASSEMBLER_INFO, '<absolute=0x0000000000655010>'), 
            (Tokens.CLOSE_BRACKET, ']'), (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['jmp qword ds: [ rip + 2433812 ]'],
        'x86_innereye': ['jmp [ rip + {immval} ]'],
        'x86_deepbindiff': ['jmp {memexpr}'],
        'x86_safe': ['jmp ds: [ rip + {dispmem} ]'],
        'x86_deepsemantic': ['jmp {memptr}8 ds: [ rip + {immval} ]'],
        'x86_hpc': ['jmp qword ds: [ rip + {split_imm} 2433 812 ]'],
        'x86_hpc_repl': ['jmp qword ds: [ rip + {split_imm} 243 381 2 ]'],
    },

    {
        'input': '0x0040427f CMP    dword ds:[rip + 0x0000000000252f8a<absolute=0x0000000000657210>], 0x00',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x0040427f'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'CMP'), (Tokens.SPACING, '    '),
            (Tokens.MEMORY_SIZE, 'dword'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'ds'), (Tokens.COLON, ':'),
            (Tokens.OPEN_BRACKET, '['), (Tokens.REGISTER, 'rip'), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'), 
                (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x0000000000252f8a'), (Tokens.DISASSEMBLER_INFO, '<absolute=0x0000000000657210>'), 
            (Tokens.CLOSE_BRACKET, ']'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['cmp dword ds: [ rip + 2437002 ] 0'],
        'x86_innereye': ['cmp [ rip + {immval} ] {immval}'],
        'x86_deepbindiff': ['cmp {memexpr} {immval}'],
        'x86_safe': ['cmp ds: [ rip + {dispmem} ] 0'],
        'x86_deepsemantic': ['cmp {memptr}4 ds: [ rip + {immval} ] {immval}'],
        'x86_hpc': ['cmp dword ds: [ rip + {split_imm} 2437 002 ] 0'],
        'x86_hpc_repl': ['cmp dword ds: [ rip + {split_imm} 243 700 2 ] 0'],
    },

    {
        'input': '0x004042bf: mov    rsi, qword ds:[rip + 0x000000000025277a<absolute=0x0000000000656a40>]',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x004042bf:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'mov'), (Tokens.SPACING, '    '),
            (Tokens.REGISTER, 'rsi'), (Tokens.SPACING, ', '),
            (Tokens.MEMORY_SIZE, 'qword'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'ds'), (Tokens.COLON, ':'),
            (Tokens.OPEN_BRACKET, '['), (Tokens.REGISTER, 'rip'), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'), 
                (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x000000000025277a'), (Tokens.DISASSEMBLER_INFO, '<absolute=0x0000000000656a40>'), 
            (Tokens.CLOSE_BRACKET, ']'), (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['mov rsi qword ds: [ rip + 2434938 ]'],
        'x86_innereye': ['mov rsi [ rip + {immval} ]'],
        'x86_deepbindiff': ['mov {reg}8 {memexpr}'],
        'x86_safe': ['mov rsi ds: [ rip + {dispmem} ]'],
        'x86_deepsemantic': ['mov {reg}8 {memptr}8 ds: [ rip + {immval} ]'],
        'x86_hpc': ['mov rsi qword ds: [ rip + {split_imm} 2434 938 ]'],
        'x86_hpc_repl': ['mov rsi qword ds: [ rip + {split_imm} 243 493 8 ]'],
    },

    {
        'input': '0x00404486: test   byte ds: ,[rax+rax+,, 0x00656a80<(data)_sch_istable>], 0x04',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00404486:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'test'), (Tokens.SPACING, '   '),
            (Tokens.MEMORY_SIZE, 'byte'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'ds'), (Tokens.COLON, ':'), (Tokens.SPACING, ' ,'),
            (Tokens.OPEN_BRACKET, '['), (Tokens.REGISTER, 'rax'), (Tokens.PLUS_SIGN, '+'), 
                (Tokens.REGISTER, 'rax'), (Tokens.PLUS_SIGN, '+'), (Tokens.SPACING, ',, '),
                (Tokens.IMMEDIATE, '0x00656a80'), (Tokens.DISASSEMBLER_INFO, '<(data)_sch_istable>'), 
            (Tokens.CLOSE_BRACKET, ']'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x04'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['test byte ds: [ rax + rax + 6646400 ] 4'],
        'x86_innereye': ['test [ rax + rax + {immval} ] {immval}'],
        'x86_deepbindiff': ['test {memexpr} {immval}'],
        'x86_safe': ['test ds: [ rax + rax + {dispmem} ] 4'],
        'x86_deepsemantic': ['test {memptr}1 ds: [ {reg}8 + {reg}8 + {immval} ] {immval}'],
        'x86_hpc': ['test byte ds: [ rax + rax + {split_imm} 6646 400 ] 4'],
        'x86_hpc_repl': ['test byte ds: [ rax + rax + {split_imm} 664 640 0 ] 4'],
    },

    {
        'input': '0x004042ae: mov    qword PtR dS:[rax + 0x08], r15',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x004042ae:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'mov'), (Tokens.SPACING, '    '),
            (Tokens.MEMORY_SIZE, 'qword PtR'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'dS'), (Tokens.COLON, ':'),
            (Tokens.OPEN_BRACKET, '['), (Tokens.REGISTER, 'rax'), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'), 
                (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x08'), (Tokens.CLOSE_BRACKET, ']'), 
            (Tokens.SPACING, ', '), (Tokens.REGISTER, 'r15'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['mov qword ds: [ rax + 8 ] r15'],
        'x86_innereye': ['mov [ rax + {immval} ] r15'],
        'x86_deepbindiff': ['mov {memexpr} {reg}8'],
        'x86_safe': ['mov ds: [ rax + 8 ] r15'],
        'x86_deepsemantic': ['mov {memptr}8 ds: [ {reg}8 + {immval} ] {reg}8'],
        'x86_hpc': ['mov qword ds: [ rax + 8 ] r15'],
        'x86_hpc_repl': ['mov qword ds: [ rax + 8 ] r15'],
    },

    {
        'input': '0x004042ab mov    rax, qword ds:[rbx]',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x004042ab'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'mov'), (Tokens.SPACING, '    '),
            (Tokens.REGISTER, 'rax'), (Tokens.SPACING, ', '),
            (Tokens.MEMORY_SIZE, 'qword'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'ds'), (Tokens.COLON, ':'),
            (Tokens.OPEN_BRACKET, '['), (Tokens.REGISTER, 'rbx'), (Tokens.CLOSE_BRACKET, ']'), 
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['mov rax qword ds: [ rbx ]'],
        'x86_innereye': ['mov rax [ rbx ]'],
        'x86_deepbindiff': ['mov {reg}8 {memexpr}'],
        'x86_safe': ['mov rax ds: [ rbx ]'],
        'x86_deepsemantic': ['mov {reg}8 {memptr}8 ds: [ {reg}8 ]'],
        'x86_hpc': ['mov rax qword ds: [ rbx ]'],
        'x86_hpc_repl': ['mov rax qword ds: [ rbx ]'],
    },

    {
        'input': '0x00404449: repne.scasb',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x00404449:'), (Tokens.SPACING, ' '), 
            (Tokens.INSTRUCTION_PREFIX, 'repne'), (Tokens.SPACING, '.'), (Tokens.OPCODE, 'scasb'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['repne scasb'],
        'x86_innereye': ['repne scasb'],
        'x86_deepbindiff': ['repne scasb'],
        'x86_safe': ['repne scasb'],
        'x86_deepsemantic': ['repne scasb'],
        'x86_hpc': ['repne scasb'],
        'x86_hpc_repl': ['repne scasb'],
    },

    {
        'input': '0x0040429e: call   0x0000000000403360<(func)bfd_demangle>',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x0040429e:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'call'), (Tokens.SPACING, '   '),
            (Tokens.IMMEDIATE, '0x0000000000403360'), (Tokens.DISASSEMBLER_INFO, '<(func)bfd_demangle>'), 
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['call 4207456'],
        'x86_innereye': ['call {func}'],
        'x86_deepbindiff': ['call {immval}'],
        'x86_safe': ['call {immval}'],
        'x86_deepsemantic': ['call {func}'],
        'x86_hpc': ['call {split_imm} 4207 456'],
        'x86_hpc_repl': ['call {split_imm} 420 745 6'],
    },

    {
        'input': '0x004042b7: mov    ecx, 0x00000002',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x004042b7:'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'mov'), (Tokens.SPACING, '    '),
            (Tokens.REGISTER, 'ecx'), (Tokens.SPACING, ', '), (Tokens.IMMEDIATE, '0x00000002'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['mov ecx 2'],
        'x86_innereye': ['mov ecx {immval}'],
        'x86_deepbindiff': ['mov {reg}4 {immval}'],
        'x86_safe': ['mov ecx 2'],
        'x86_deepsemantic': ['mov {reg}4 {immval}'],
        'x86_hpc': ['mov ecx 2'],
        'x86_hpc_repl': ['mov ecx 2'],
    },

    {
        'input': '0x004042c6 mov,rDi,R12',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x004042c6'), (Tokens.SPACING, ' '), 
            (Tokens.OPCODE, 'mov'), (Tokens.SPACING, ','),
            (Tokens.REGISTER, 'rDi'), (Tokens.SPACING, ','), (Tokens.REGISTER, 'R12'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['mov rdi r12'],
        'x86_innereye': ['mov rdi r12'],
        'x86_deepbindiff': ['mov {reg}8 {reg}8'],
        'x86_safe': ['mov rdi r12'],
        'x86_deepsemantic': ['mov {reg}8 {reg}8'],
        'x86_hpc': ['mov rdi r12'],
        'x86_hpc_repl': ['mov rdi r12'],
    },


    ###############
    # Fake Inputs #
    ###############


    {  # Multiple prefixes in the correct order, along with a weird memory address, different immediates/registers, and COLON register address
        'input': 'lock add.repne.rep.repE.lock  v2xmmword ptr ss:[0x003 + r14*0o013 + rip*124 + mxcsr], gs:rip',
        'bad_assembly': True, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_PREFIX, 'lock'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'add'), (Tokens.SPACING, '.'), (Tokens.INSTRUCTION_PREFIX, 'repne'), 
                (Tokens.SPACING, '.'), (Tokens.INSTRUCTION_PREFIX, 'rep'), (Tokens.SPACING, '.'),
                (Tokens.INSTRUCTION_PREFIX, 'repE'), (Tokens.SPACING, '.'), (Tokens.INSTRUCTION_PREFIX, 'lock'),
            (Tokens.SPACING, '  '), (Tokens.MEMORY_SIZE, 'v2xmmword ptr'), 
                (Tokens.SPACING, ' '), (Tokens.REGISTER, 'ss'), (Tokens.COLON, ':'),
            (Tokens.OPEN_BRACKET, '['), (Tokens.IMMEDIATE, '0x003'), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'),
                (Tokens.SPACING, ' '), (Tokens.REGISTER, 'r14'), (Tokens.TIMES_SIGN, '*'), (Tokens.IMMEDIATE, '0o013'),
                (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'rip'), 
                (Tokens.TIMES_SIGN, '*'), (Tokens.IMMEDIATE, '124'), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'), 
                (Tokens.SPACING, ' '), (Tokens.REGISTER, 'mxcsr'), (Tokens.CLOSE_BRACKET, ']'),
            (Tokens.SPACING, ', '), (Tokens.REGISTER, 'gs'), (Tokens.COLON, ':'), (Tokens.REGISTER, 'rip'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['lock add repne rep repe lock v2xmmword ss: [ 3 + r14 * 11 + rip * 124 + mxcsr ] gs: rip'],
        'x86_innereye': ['lock add repne rep repe lock [ {immval} + r14 * {immval} + rip * {immval} + mxcsr ] rip'],
        'x86_deepbindiff': ['lock add repne rep repe lock {memexpr} rip'],
        'x86_safe': ['lock add repne rep repe lock ss: [ 3 + r14 * 11 + rip * 124 + mxcsr ] gs: rip'],
        'x86_deepsemantic': ['lock add repne rep repe lock {memptr}32 ss: [ {immval} + {reg}8 * 11 + rip * {immval} + mxcsr ] gs: rip'],
        'x86_hpc': ['lock add repne rep repe lock v2xmmword ss: [ 3 + r14 * 11 + rip * 124 + mxcsr ] gs: rip'],
        'x86_hpc_repl': ['lock add repne rep repe lock v2xmmword ss: [ 3 + r14 * 11 + rip * 124 + mxcsr ] gs: rip'],
    },

    {  # A real opcode with an underscore mixed in with instruction prefixes
        'input': 'lock.repne,vcmpneq_oqsd..,.,rep,,,,pt r15',
        'bad_assembly': True, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_PREFIX, 'lock'), (Tokens.SPACING, '.'), (Tokens.INSTRUCTION_PREFIX, 'repne'), (Tokens.SPACING, ','),
                (Tokens.OPCODE, 'vcmpneq_oqsd'), (Tokens.SPACING, '..,.,'), (Tokens.INSTRUCTION_PREFIX, 'rep'), (Tokens.SPACING, ',,,,'),
                (Tokens.BRANCH_PREDICTION, 'pt'),
            (Tokens.SPACING, ' '), (Tokens.REGISTER, 'r15'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['lock repne vcmpneq_oqsd rep pt r15'],
        'x86_innereye': ['lock repne vcmpneq_oqsd rep r15'],
        'x86_deepbindiff': ['lock repne vcmpneq_oqsd rep {reg}8'],
        'x86_safe': ['lock repne vcmpneq_oqsd rep r15'],
        'x86_deepsemantic': ['lock repne vcmpneq_oqsd rep {reg}8'],
        'x86_hpc': ['lock repne vcmpneq_oqsd rep pt r15'],
        'x86_hpc_repl': ['lock repne vcmpneq_oqsd rep pt r15'],
    },

    {  # Immediate thresholds and positive immediates in different bases
        'input': '0x3424 add %d %d %d 0x%x 0x%x 0x%x 0o%o 0o%o 0o%o %s %s %s' % ((DEFAULT_IMMEDIATE_THRESHOLD - 1, DEFAULT_IMMEDIATE_THRESHOLD, DEFAULT_IMMEDIATE_THRESHOLD + 1) * 3 + (bin(DEFAULT_IMMEDIATE_THRESHOLD - 1), bin(DEFAULT_IMMEDIATE_THRESHOLD), bin(DEFAULT_IMMEDIATE_THRESHOLD + 1))),
        'bad_assembly': True, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_ADDRESS, '0x3424'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '),
            (Tokens.IMMEDIATE, '%d' % (DEFAULT_IMMEDIATE_THRESHOLD - 1)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '%d' % (DEFAULT_IMMEDIATE_THRESHOLD)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '%d' % (DEFAULT_IMMEDIATE_THRESHOLD + 1)), (Tokens.SPACING, ' '),
            (Tokens.IMMEDIATE, '0x%x' % (DEFAULT_IMMEDIATE_THRESHOLD - 1)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x%x' % (DEFAULT_IMMEDIATE_THRESHOLD)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x%x' % (DEFAULT_IMMEDIATE_THRESHOLD + 1)), (Tokens.SPACING, ' '),
            (Tokens.IMMEDIATE, '0o%o' % (DEFAULT_IMMEDIATE_THRESHOLD - 1)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0o%o' % (DEFAULT_IMMEDIATE_THRESHOLD)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0o%o' % (DEFAULT_IMMEDIATE_THRESHOLD + 1)), (Tokens.SPACING, ' '),
            (Tokens.IMMEDIATE, bin(DEFAULT_IMMEDIATE_THRESHOLD - 1)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, bin(DEFAULT_IMMEDIATE_THRESHOLD)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, bin(DEFAULT_IMMEDIATE_THRESHOLD + 1)),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['add %d %d %d %d %d %d %d %d %d %d %d %d' % ((DEFAULT_IMMEDIATE_THRESHOLD - 1, DEFAULT_IMMEDIATE_THRESHOLD, DEFAULT_IMMEDIATE_THRESHOLD + 1) * 4)],
        'x86_innereye': ['add ' + ' '.join(['{immval}'] * 12)],
        'x86_deepbindiff': ['add ' + ' '.join(['{immval}'] * 12)],
        'x86_safe': ['add %d %d {immval} %d %d {immval} %d %d {immval} %d %d {immval}' % ((DEFAULT_IMMEDIATE_THRESHOLD - 1, DEFAULT_IMMEDIATE_THRESHOLD) * 4)],
        'x86_deepsemantic': ['add ' + ' '.join(['{immval}'] * 12)],
        'x86_hpc': ['add ' + _hpc_norm_nums((DEFAULT_IMMEDIATE_THRESHOLD - 1, DEFAULT_IMMEDIATE_THRESHOLD, DEFAULT_IMMEDIATE_THRESHOLD + 1) * 4)],
        'x86_hpc_repl': ['add ' + _hpc_norm_nums((DEFAULT_IMMEDIATE_THRESHOLD - 1, DEFAULT_IMMEDIATE_THRESHOLD, DEFAULT_IMMEDIATE_THRESHOLD + 1) * 4, num_digits=3)],
    },

    {  # Immediate thresholds and negative immediates in different bases
        'input': 'add -%d -%d -%d -0x%x -0x%x -0x%x -0o%o -0o%o -0o%o -%s -%s -%s' % ((DEFAULT_IMMEDIATE_THRESHOLD - 1, DEFAULT_IMMEDIATE_THRESHOLD, DEFAULT_IMMEDIATE_THRESHOLD + 1) * 3 + (bin(DEFAULT_IMMEDIATE_THRESHOLD - 1), bin(DEFAULT_IMMEDIATE_THRESHOLD), bin(DEFAULT_IMMEDIATE_THRESHOLD + 1))),
        'bad_assembly': True, 
        'x86_base_tokenizer': [
            (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '),
            (Tokens.IMMEDIATE, '-%d' % (DEFAULT_IMMEDIATE_THRESHOLD - 1)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '-%d' % (DEFAULT_IMMEDIATE_THRESHOLD)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '-%d' % (DEFAULT_IMMEDIATE_THRESHOLD + 1)), (Tokens.SPACING, ' '),
            (Tokens.IMMEDIATE, '-0x%x' % (DEFAULT_IMMEDIATE_THRESHOLD - 1)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '-0x%x' % (DEFAULT_IMMEDIATE_THRESHOLD)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '-0x%x' % (DEFAULT_IMMEDIATE_THRESHOLD + 1)), (Tokens.SPACING, ' '),
            (Tokens.IMMEDIATE, '-0o%o' % (DEFAULT_IMMEDIATE_THRESHOLD - 1)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '-0o%o' % (DEFAULT_IMMEDIATE_THRESHOLD)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '-0o%o' % (DEFAULT_IMMEDIATE_THRESHOLD + 1)), (Tokens.SPACING, ' '),
            (Tokens.IMMEDIATE, '-' + bin(DEFAULT_IMMEDIATE_THRESHOLD - 1)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '-' + bin(DEFAULT_IMMEDIATE_THRESHOLD)), (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '-' + bin(DEFAULT_IMMEDIATE_THRESHOLD + 1)),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['add -%d -%d -%d -%d -%d -%d -%d -%d -%d -%d -%d -%d' % ((DEFAULT_IMMEDIATE_THRESHOLD - 1, DEFAULT_IMMEDIATE_THRESHOLD, DEFAULT_IMMEDIATE_THRESHOLD + 1) * 4)],
        'x86_innereye': ['add ' + ' '.join(['-{immval}'] * 12)],
        'x86_deepbindiff': ['add ' + ' '.join(['{immval}'] * 12)],
        'x86_safe': ['add -%d -%d {immval} -%d -%d {immval} -%d -%d {immval} -%d -%d {immval}' % ((DEFAULT_IMMEDIATE_THRESHOLD - 1, DEFAULT_IMMEDIATE_THRESHOLD) * 4)],
        'x86_deepsemantic': ['add ' + ' '.join(['{immval}'] * 12)],
        'x86_hpc': ['add ' + _hpc_norm_nums((-DEFAULT_IMMEDIATE_THRESHOLD + 1, -DEFAULT_IMMEDIATE_THRESHOLD, -DEFAULT_IMMEDIATE_THRESHOLD - 1) * 4)],
        'x86_hpc_repl': ['add ' + _hpc_norm_nums((-DEFAULT_IMMEDIATE_THRESHOLD + 1, -DEFAULT_IMMEDIATE_THRESHOLD, -DEFAULT_IMMEDIATE_THRESHOLD - 1) * 4, num_digits=3)],
    },

    {  # Rose negative immediates insert minus signs properly (different depending on normalization)
        'input': 'add 0x003<-3> [rip +0x0012<-18>]',
        'bad_assembly': False,
        'x86_base_tokenizer': [
            (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), 
            (Tokens.IMMEDIATE, '0x003'), (Tokens.DISASSEMBLER_INFO, '<-3>'), (Tokens.SPACING, ' '),
            (Tokens.OPEN_BRACKET, '['), (Tokens.REGISTER, 'rip'), (Tokens.SPACING, ' '), (Tokens.PLUS_SIGN, '+'),
                (Tokens.IMMEDIATE, '0x0012'), (Tokens.DISASSEMBLER_INFO, '<-18>'), (Tokens.CLOSE_BRACKET, ']'),
            (Tokens.NEWLINE, '\n')
        ],
        'x86_base': ['add -3 [ rip + -18 ]'],
        'x86_innereye': ['add -{immval} [ rip + -{immval} ]'],
        'x86_deepbindiff': ['add {immval} {memexpr}'],
        'x86_safe': ['add -3 [ rip + -18 ]'],
        'x86_deepsemantic': ['add {immval} [ rip + {immval} ]'],
        'x86_hpc': ['add -3 [ rip + -18 ]'],
        'x86_hpc_repl': ['add -3 [ rip + -18 ]'],
    },

    {  # Multiple rose/ghidra newlines
        'input': 'add | sub r8 | mov [RIP]\n0x1234: ret',
        'bad_assembly': True, 
        'x86_base_tokenizer': [
            (Tokens.OPCODE, 'add'), (Tokens.SPACING, ' '), (Tokens.NEWLINE, '|'), (Tokens.SPACING, ' '),
            (Tokens.OPCODE, 'sub'), (Tokens.SPACING, ' '), (Tokens.REGISTER, 'r8'), (Tokens.SPACING, ' '), 
                (Tokens.NEWLINE, '|'), (Tokens.SPACING, ' '),
            (Tokens.OPCODE, 'mov'), (Tokens.SPACING, ' '), (Tokens.OPEN_BRACKET, '['), (Tokens.REGISTER, 'RIP'), 
                (Tokens.CLOSE_BRACKET, ']'), (Tokens.NEWLINE, '\n'),
            (Tokens.INSTRUCTION_ADDRESS, '0x1234:'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'ret'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['add', 'sub r8', 'mov [ rip ]', 'ret'],
        'x86_innereye': ['add', 'sub r8', 'mov [ rip ]', 'ret'],
        'x86_deepbindiff': ['add', 'sub {reg}8', 'mov {memexpr}', 'ret'],
        'x86_safe': ['add', 'sub r8', 'mov [ rip ]', 'ret'],
        'x86_deepsemantic': ['add', 'sub {reg}8', 'mov [ rip ]', 'ret'],
        'x86_hpc': ['add', 'sub r8', 'mov [ rip ]', 'ret'],
        'x86_hpc_repl': ['add', 'sub r8', 'mov [ rip ]', 'ret'],
    },

    {  # Jump with branch prediction (not taken)
        'input': 'lock jne.repne,Pn 0x002',
        'bad_assembly': True, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_PREFIX, 'lock'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'jne'), (Tokens.SPACING, '.'), 
                (Tokens.INSTRUCTION_PREFIX, 'repne'), (Tokens.SPACING, ','), (Tokens.BRANCH_PREDICTION, 'Pn'), 
            (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x002'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['lock jne repne pn 2'],
        'x86_innereye': ['lock jne repne {immval}'],
        'x86_deepbindiff': ['lock jne repne {immval}'],
        'x86_safe': ['lock jne repne 2'],
        'x86_deepsemantic': ['lock jne repne {immval}'],  # Jmpdst only applies to jump opcode immediately previous
        'x86_hpc': ['lock jne repne pn 2'],
        'x86_hpc_repl': ['lock jne repne pn 2'],
    },

    {  # Jump with branch prediction (taken)
        'input': 'lock jne.repne,PT 0x002',
        'bad_assembly': True, 
        'x86_base_tokenizer': [
            (Tokens.INSTRUCTION_PREFIX, 'lock'), (Tokens.SPACING, ' '), (Tokens.OPCODE, 'jne'), (Tokens.SPACING, '.'), 
                (Tokens.INSTRUCTION_PREFIX, 'repne'), (Tokens.SPACING, ','), (Tokens.BRANCH_PREDICTION, 'PT'), 
            (Tokens.SPACING, ' '), (Tokens.IMMEDIATE, '0x002'),
            (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['lock jne repne pt 2'],
        'x86_innereye': ['lock jne repne {immval}'],
        'x86_deepbindiff': ['lock jne repne {immval}'],
        'x86_safe': ['lock jne repne 2'],
        'x86_deepsemantic': ['lock jne repne {immval}'],  # Jmpdst only applies to jump opcode immediately previous
        'x86_hpc': ['lock jne repne pt 2'],
        'x86_hpc_repl': ['lock jne repne pt 2'],
    },

    {  # Opcode that starts with a register substring
        'input': 'cwde',
        'bad_assembly': False, 
        'x86_base_tokenizer': [
            (Tokens.OPCODE, 'cwde'), (Tokens.NEWLINE, '\n'),
        ],
        'x86_base': ['cwde'],
        'x86_innereye': ['cwde'],
        'x86_deepbindiff': ['cwde'],
        'x86_safe': ['cwde'],
        'x86_deepsemantic': ['cwde'],
        'x86_hpc': ['cwde'],
        'x86_hpc_repl': ['cwde'],
    },


    ################
    # Error Inputs #
    ################


    {  # Token mismatch (unknown character)
        'input': 'add ###',
        'bad_assembly': True, 
        'x86_base_tokenizer': TokenMismatchError,
        'x86_base': TokenMismatchError,
        'x86_innereye': TokenMismatchError,
        'x86_deepbindiff': TokenMismatchError,
        'x86_safe': TokenMismatchError,
        'x86_deepsemantic': TokenMismatchError,
        'x86_hpc': TokenMismatchError,
        'x86_hpc_repl': TokenMismatchError,
    },

    {  # Token mismatch (bad character in opcode)
        'input': 'op&code',
        'bad_assembly': True, 
        'x86_base_tokenizer': TokenMismatchError,
        'x86_base': TokenMismatchError,
        'x86_innereye': TokenMismatchError,
        'x86_deepbindiff': TokenMismatchError,
        'x86_safe': TokenMismatchError,
        'x86_deepsemantic': TokenMismatchError,
        'x86_hpc': TokenMismatchError,
        'x86_hpc_repl': TokenMismatchError,
    },

]

def _format_repl(string):
    _repls = [('{immval}', IMMEDIATE_VALUE_STR), ('{str}', STRING_LITERAL_STR), ('{func}', FUNCTION_CALL_STR), 
              ('{memexpr}', MEMORY_EXPRESSION_STR), ('{dispimm}', DISPLACEMENT_IMMEDIATE_STR), ('{reg}', GENERAL_REGISTER_STR), 
              ('{split_imm}', SPLIT_IMMEDIATE_TOKEN), ('{jmpdst}', JUMP_DESTINATION_STR), ('{dispmem}', IMMEDIATE_VALUE_STR),
              ('{memptr}', MEM_SIZE_TOKEN_STR)]
    for k, v in _repls:
        string = string.replace(k, v)
    return string
X86_TEST_INPUTS = [{k: _format_repl(v) if isinstance(v, str) else [_format_repl(o) for o in v] if isinstance(v, (list, tuple)) and isinstance(v[0], str) else v for k, v in d.items()} for d in X86_TEST_INPUTS]