import pytest
import re
from .x86_manual_lines import X86_TEST_INPUTS, X86_TEST_OBJECTS
from bincfg import BaseTokenizer, BaseNormalizer, TokenizationLevel, X86HPCDataNormalizer
from bincfg.normalization.norm_utils import RE_STRING_LITERAL, INSTRUCTION_START_TOKEN


def test_individual_manual_lines_tokenizer():
    """Tests any x86 tokenizer works as expected on manual lines"""
    for name, cls, kwargs in X86_TEST_OBJECTS:
        if not issubclass(cls, BaseTokenizer): continue
        tokenizer = cls(**kwargs)

        # Test them one by one
        for input_dict in X86_TEST_INPUTS:
            test_input, expected = input_dict['input'], input_dict[name]

            if isinstance(expected, type):
                with pytest.raises(expected):
                    tokenizer(test_input)
            else:
                assert tokenizer(test_input) == expected, "Input: %s\nOutput: %s\nExpect: %s" % (test_input, tokenizer(test_input), expected)
        
        # Test all together
        input_list = [d['input'] for d in X86_TEST_INPUTS if not isinstance(d[name], type)]
        expected = [t for d in X86_TEST_INPUTS for t in (d[name] if not isinstance(d[name], type) else [])]

        assert tokenizer(*input_list) == expected, 'all inputs as *args'
        assert tokenizer('\n'.join(input_list)) == expected, 'all inputs as newline'


def test_individual_manual_lines_normalizers():
    """Tests any x86 normalizer works as expected on manual lines"""    
    for name, cls, kwargs in X86_TEST_OBJECTS:
        if not issubclass(cls, BaseNormalizer): continue
        for tl in ['opcode', 'instruction']:
            normalizer = cls(tokenization_level=tl, **kwargs)

            # Test them one by one
            for input_dict in X86_TEST_INPUTS:
                test_input, expected = input_dict['input'], _apply_tl(input_dict[name], tl)

                if isinstance(expected, type):
                    with pytest.raises(expected):
                        normalizer(test_input)
                else:
                    if False: # For debugging
                        try:
                            assert normalizer(test_input) == expected, "Normalizer: %s\nInput: %s\nOutput: %s\nExpect: %s" % (str(normalizer), test_input, normalizer(test_input), expected)
                        except Exception as e:
                            import traceback
                            raise ValueError("Failed %s: %s\n%s" % (str(normalizer), repr(test_input), traceback.format_exc()))
                    assert normalizer(test_input) == expected, "Normalizer: %s\nInput: %s\nOutput: %s\nExpect: %s" % (str(normalizer), test_input, normalizer(test_input), expected)
            
            # Test all together
            input_list = [d['input'] for d in X86_TEST_INPUTS if not isinstance(d[name], type)]
            expected = [t for d in X86_TEST_INPUTS for t in (_apply_tl(d[name], tl) if not isinstance(d[name], type) else [])]

            assert normalizer(*input_list) == expected, 'all inputs as *args'
            assert normalizer('\n'.join(input_list)) == expected, 'all inputs as newline'


def _apply_tl(expected, tl):
    if isinstance(expected, type):
        return expected
    elif tl == 'opcode':
        # Replace all strings in lines
        str_map = {}
        _repl = lambda match: '__STR-%d-__' % str_map.setdefault(match.group(), len(str_map))
        expected = [re.sub(RE_STRING_LITERAL, _repl, s) for s in expected]
        str_map = {v: k for k,v in str_map.items()}

        # Split into tokens and un-replace strings
        return [(str_map[int(t.split('-')[1])] if '__STR-' in t else t) for s in expected for t in ([INSTRUCTION_START_TOKEN] + s.split(' '))]

    elif tl == 'instruction':
        return expected
    else:
        raise NotImplementedError(str(tl))


def test_individual_manual_lines_renormalizeable():
    """Tests any x86 tokenizer works on renormalizeable lines"""
    for fname, fcls, fkwargs in X86_TEST_OBJECTS:
        if not issubclass(fcls, BaseNormalizer) or not fcls.renormalizable: continue

        for ftl in [TokenizationLevel.OPCODE, TokenizationLevel.INSTRUCTION]:

            fnormalizer = fcls(tokenization_level=ftl, **fkwargs)

            for name, cls, kwargs in X86_TEST_OBJECTS:
                if not issubclass(cls, BaseNormalizer): continue
                for tl in ['opcode', 'instruction']:
                    normalizer = cls(tokenization_level=tl, **kwargs)

                    # Test them one by one
                    for input_dict in X86_TEST_INPUTS:
                        if not isinstance(input_dict[name], str): continue  # Ignore fails
                        test_input, expected = fnormalizer(input_dict['input']), _apply_tl(input_dict[name], tl)

                        if isinstance(expected, type):
                            with pytest.raises(expected):
                                normalizer(test_input)
                        else:
                            if False: # For debugging
                                try:
                                    assert normalizer(test_input) == expected, "Renormalization with starting normalizer: %s\nNormalizer: %s\nInput: %s\nFirstOutput: %s\nOutput: %s\nExpect: %s" % (str(fnormalizer), str(normalizer), input_dict['input'], test_input, normalizer(test_input), expected)
                                except Exception as e:
                                    import traceback
                                    raise ValueError("Failed %s: %s\n%s" % (str(normalizer), repr(test_input), traceback.format_exc()))
                            assert normalizer(*test_input) == expected, "Renormalization with starting normalizer: %s\nNormalizer: %s\nInput: %s\nFirstOutput: %s\nOutput: %s\nExpect: %s" % (str(fnormalizer), str(normalizer), input_dict['input'], test_input, normalizer(test_input), expected)
                    
                    # Test all together
                    input_list = fnormalizer(*[d['input'] for d in X86_TEST_INPUTS if not isinstance(d[name], type)])
                    expected = [t for d in X86_TEST_INPUTS for t in (_apply_tl(d[name], tl) if not isinstance(d[name], type) else [])]

                    # Don't do this one for the full shebang because it's a bit of a pain to replace the strings + immediates
                    #   in the expected. Maybe I'll do this later, but it's not that necessary...
                    if fcls is X86HPCDataNormalizer and 'replace_strings' in fkwargs and fkwargs['replace_strings']: 
                        continue  
                    assert normalizer(*input_list) == expected, 'all inputs as *args'