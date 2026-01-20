"""Count lines of code

Requires:
    - pygments
    - pcpp: https://pypi.org/project/pcpp/
"""
from bincfg import get_module
from io import StringIO


# Make it so we don't need these libraries installed unless you wish to use these values
_FAILED_MODULES = []


if get_module('pygments', raise_err=False) is not None:
    from pygments.token import Token, Literal
    from pygments.lexers import get_lexer_by_name
    CPP_UNUSED_TYPES_EXACT = [Token.Keyword.Type,
                              Token.Operator,]
    CPP_UNUSED_TYPES_IN = [Literal.Number,
                           Literal.String,
                           Token.Name,
                           Token.Text,
                           Token.Comment,]
else:
    _FAILED_MODULES.append('pygments')


CPP_DONT_COUNT_KEYWORDS = [
    'alignas', 'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case', 'catch', 'char', 
    'char8_t', 'char16_t', 'char32_t', 'class', 'compl', 'concept', 'const', 'consteval', 'constexpr', 'constinit', 
    'const_cast', 'continue', 'co_await', 'co_return', 'co_yield', 'decltype', 'default', 'delete', 'do', 'double', 
    'dynamic_cast', 'else', 'enum', 'explicit', 'export', 'extern', 'false', 'float', 'for', 'friend', 'goto', 'if', 
    'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 
    'or_eq', 'reflexpr', 'register', 'reinterpret_cast', 'requires', 'return', 'short', 'signed', 'sizeof', 'static', 
    'static_assert', 'static_cast', 'struct', 'template', 'this', 'thread_local', 'throw', 'true', 'typedef', 'typeid', 
    'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile', 'wchar_t', 'xor', 'xor_eq', 'final', 
    'override', 'transaction_safe', 'transaction_safe_dynamic', 'import', 'module', '__int64', '__asm', '__restrict',
    '__inline', '__int32', 'throws', '_inline',
]

CPP_COUNT_KEYWORDS = [
    'atomic_cancel', 'atomic_commit', 'atomic_noexcept', 'case', 'default', 'if', 'private', 'protected', 'public', 
    'synchronized', 'switch', 'try', 'while',
]


if get_module('pcpp', raise_err=False) is not None:
    from pcpp import Preprocessor, OutputDirective, Action
    class CPPSingleFilePreprocessor(Preprocessor):
        """Class to expand some preprocessor directives in C++ code"""
        def __init__(self):
            super().__init__()
            self.line_directive = None
            
            # Passthru dunder magic macros
            self.undef('__DATE__')
            self.undef('__TIME__')
            self.expand_linemacro = False
            self.expand_filemacro = False
            self.expand_countermacro = False
        
        def expand_string(self, string):
            # Parse the input
            self.parse(string)
            
            # String to a StringIO() and return
            output = StringIO()
            self.write(output)
            return output.getvalue()
            
        def on_include_not_found(self, is_malformed, is_system_include, curdir, includepath):
            # Skip unfound includes, we are assuming all of them are like this
            raise OutputDirective(Action.IgnoreAndPassThrough)

        # Unknown macros/directives are passed through
        def on_unknown_macro_in_defined_expr(self, tok): return
        def on_unknown_macro_in_expr(self, ident): return
        def on_unknown_macro_function_in_expr(self, ident): return
        def on_directive_unknown(self, directive, toks, ifpassthru, precedingtoks): return

        # Stop it from printing errors to stderr (there was this increment of return code so i left it, idk what it does)
        def on_error(self,file,line,msg): self.return_code += 1
else:
    _FAILED_MODULES.append('pcpp')


def cpp_loc(string, expand_directives=True):
    """Counts the lines of code for C/C++ sources

    Requires the following packages to be installed:

        - pygments
        - pcpp: https://pypi.org/project/pcpp/
    
    The process is basically:

        - Expand preprocessor directives (currently, #includes are left as in since we are assuming `string` is the
          entire non-library file, and #defines/#if's within this file are expanded). This will only happen if
          `expand_directives` is True
        - Every remaining preprocessor directive counts as one line
        - All comments are removed
        - Empty semicolons are removed. An 'empty' semicolons is any semicolon that:

            1. Comes after some previous semicolon or immediately at the beginning of the file
            2. Contains only non-useful tokens (whitespace, comments, etc.)

            This will remove multiple semicolons in a row ';;;;' and count it as only one

        - Every remaining semicolon counts as one line
        - Every if/while statement counts as one line
        - Else statements don't count as a line
        - Function definitions count as a line
        - Every for statement removes one line since they have 2 semicolons, but should also count as one line
        - Some statements that don't have an associated semicolon also count as a line: 'public'
        - Keywords defining thread-used subroutines count as a line of code: 'synchronized', 'atomic_noexcept', etc.
        - The 'try' keyword counts as a line, but not the 'catch' part. Together they count as one line
        - All parts of a switch statement ('switch', 'case', 'default') count as one line each
    
    NOTE: some statements aren't outright counted but will be indirectly counted due to required semicolons. EG: 
    class and structure definitions, typedef's, using namespace, etc.

    Some notes/Caveats:
    
        - Currently, asm() code is just considered to be one line, no matter how many lines the assembly takes up
    
    Args:
        string (str): the string to count lines of code in
        expand_directives (bool): if True, then CPP directives will be expanded before counting lines of code
    """
    if len(_FAILED_MODULES) > 0:
        raise ImportError("Could not find required imports for loc computation: %s" % _FAILED_MODULES)
    
    if expand_directives:
        string = CPPSingleFilePreprocessor().expand_string(string)
    
    debug = False

    # Keeps track of the last 'real' token, that way, if any new semicolons appear right after, they are not counted
    new_tokens = []
    count = 0
    for token_type, token in get_lexer_by_name('c++').get_tokens(string):
        if debug: print(token_type, token)
        if not any(token_type in t for t in [Token.Comment]) and token.strip() != '':
            new_tokens.append(token)
        
        if token_type is Token.Comment.Preproc:
            if token == '#':
                count += 1
                if debug: print("Preprocessor", count)
            continue
        
        elif token_type is Token.Punctuation:
            if token == ';':
                if len(new_tokens) >= 2 and new_tokens[-2] != ';':
                    count += 1
                    if debug: print("Semicolon - real", count)
                continue
            elif token in "[](),.{}:":
                continue
        
        elif token_type in [Token.Keyword, Token.Keyword.Reserved]:
            if token in ['for']:  # For statements must remove 1 token
                count -= 1
                if debug: print("sub1 for for", count)
                continue
            elif token in CPP_COUNT_KEYWORDS:  # These don't have a ; after
                count += 1
                if debug: print("counted kekyword", count)
                continue
            elif not debug or token in CPP_DONT_COUNT_KEYWORDS:
                continue
        
        elif token_type is Token.Name.Function:
            count += 1
            if debug: print('function', count)
            continue
        
        elif token_type in CPP_UNUSED_TYPES_EXACT or token_type in CPP_UNUSED_TYPES_IN \
            or any(token_type in ut for ut in CPP_UNUSED_TYPES_IN) or token.strip() == '':
            continue

        elif token_type in Token.Error:
            continue
        
        raise ValueError("UNKNOWN: %s, %s\n%s" % (token_type, token, string))
    
    return count
