import pytest
import pickle
import copy
import os
from bincfg import CFG, X86DeepSemanticNormalizer, Architectures, CFGEdge, EdgeType, CFGBasicBlock, CFGFunction, get_architecture
from .manual_cfgs import get_all_manual_cfg_functions


def test_build_cfg(print_hashes):
    """Building a blank CFG object"""
    cfg = CFG()

    if print_hashes:
        print(__file__+'-empty', hash(cfg))
    else:
        assert hash(cfg) == 2176661656791943841

    assert cfg.normalizer is None
    assert isinstance(cfg.metadata, dict) and cfg.metadata == {}
    assert isinstance(cfg.functions_dict, dict) and cfg.functions_dict == {}
    assert isinstance(cfg.blocks_dict, dict) and cfg.blocks_dict == {}
    assert cfg.functions == []
    assert cfg.blocks == []
    assert cfg.edges == []
    assert cfg.num_functions == 0
    assert cfg.num_blocks == 0
    assert cfg.num_asm_lines == 0
    assert cfg.num_edges == 0
    with pytest.raises(KeyError):
        cfg.architecture

    cfg = CFG(normalizer='x86deepsemantic', metadata={'a': 10})

    assert cfg.normalizer == X86DeepSemanticNormalizer()
    assert isinstance(cfg.metadata, dict) and cfg.metadata == {'a': 10}
    assert isinstance(cfg.functions_dict, dict) and cfg.functions_dict == {}
    assert isinstance(cfg.blocks_dict, dict) and cfg.blocks_dict == {}
    with pytest.raises(KeyError):
        cfg.architecture


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_cfgs_construction(cfg_func):
    """Tests all manual cfgs can be constructed, and basic types of attributes"""
    res = cfg_func(build_level='cfg')
    cfg: CFG = res['cfg']
    funcs, blocks, expected = res['functions'], res['blocks'], res['expected']

    # Check basics
    assert cfg.normalizer is None
    assert isinstance(cfg.metadata, dict) and cfg.metadata == {}
    assert isinstance(cfg.functions_dict, dict)
    assert isinstance(cfg.blocks_dict, dict)

    # Check builtins
    assert isinstance(hash(cfg), int)
    assert isinstance(str(cfg), str)
    assert isinstance(repr(cfg), str)


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_cfgs_func_block_attrs(cfg_func):
    """Tests all manual cfgs have correct `.function` and `.block` properties, and `functions_dict`/`blocks_dict`"""
    res = cfg_func(build_level='cfg')
    cfg: CFG = res['cfg']
    funcs, blocks, expected = res['functions'], res['blocks'], res['expected']
    # Check sorted function/block orders
    assert sorted([f.address for f in cfg.functions_dict.values()]) == expected['sorted_func_order']
    assert sorted(list(cfg.functions_dict.keys())) == expected['sorted_func_order']
    assert [f.address for f in cfg.functions] == expected['sorted_func_order']
    assert sorted([b.address for b in cfg.blocks_dict.values()]) == expected['sorted_block_order']
    assert sorted(list(cfg.blocks_dict.keys())) == expected['sorted_block_order']
    assert [b.address for b in cfg.blocks] == expected['sorted_block_order']
    assert len(cfg.functions_dict) == expected['num_functions']


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_cfgs_properties(cfg_func):
    """Tests all manual cfgs have working properties"""
    res = cfg_func(build_level='cfg')
    cfg: CFG = res['cfg']
    funcs, blocks, expected = res['functions'], res['blocks'], res['expected']

    # Check properties
    assert cfg.edges == [e for b in cfg.blocks for e in b.edges_out]
    assert cfg.num_functions == expected['num_functions']
    assert cfg.num_blocks == len(expected['sorted_block_order'])
    assert cfg.num_asm_lines == sum(expected['asm_counts'].values())
    assert cfg.num_edges == sum(len(blocks[addr].edges_out) for addr in expected['sorted_block_order'])
    assert cfg.asm_counts == expected['asm_counts']
    assert cfg.architecture == get_architecture(expected['architecture'])


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_cfgs_set_tokens_update_metadata(cfg_func):
    """Tests all manual cfgs have working set_tokens and update_metadata"""
    res = cfg_func(build_level='cfg')
    cfg: CFG = res['cfg']
    funcs, blocks, expected = res['functions'], res['blocks'], res['expected']

    # Check setting the tokens/updating metadata
    cfg.set_tokens({'a': 10})
    assert cfg.tokens == {'a': 10}

    for arch in Architectures:
        cfg.update_metadata({'architecture': arch.value[0]})
        assert cfg.architecture == arch


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_cfgs_cant_add_funcs(cfg_func):
    """Tests all manual cfgs cant add their functions with add_function again"""
    res = cfg_func(build_level='cfg')
    cfg: CFG = res['cfg']
    funcs, blocks, expected = res['functions'], res['blocks'], res['expected']

    # Can't add_function if already there
    for func in funcs.values():
        with pytest.raises(ValueError):
            cfg.add_function(func, override=False)


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_cfgs_get_funcs_and_blocks(cfg_func):
    """Tests all manual cfgs can get functions/blocks correctly"""
    res = cfg_func(build_level='cfg')
    cfg: CFG = res['cfg']
    funcs, blocks, expected = res['functions'], res['blocks'], res['expected']

    # Getting functions/basic blocks
    for func in funcs.values():
        assert cfg.get_function(func, raise_err=True) == func
        assert cfg.get_function(func.address, raise_err=True) == func
        assert cfg.get_function_by_name(func.name, raise_err=True) == func
    for block in blocks.values():
        assert cfg.get_block(block, raise_err=True) == block
        assert cfg.get_block(block.address, raise_err=True) == block
        assert cfg.get_block_containing_address(block, raise_err=True) == block
        for baddr in block.asm_memory_addresses:
            assert cfg.get_block_containing_address(baddr, raise_err=True) == block
    
    # DONT get these functions/blocks because they don't exist
    for addr in [21842781481277, 0, '0b01001010101010010101010101111011010', '0xFFFFFFFFFFFFFF']:
        with pytest.raises(ValueError):
            cfg.get_function(addr, raise_err=True)
        assert cfg.get_function(addr, raise_err=False) is None
        with pytest.raises(ValueError):
            cfg.get_block(addr, raise_err=True)
        assert cfg.get_block(addr, raise_err=False) is None
        with pytest.raises(ValueError):
            cfg.get_block_containing_address(addr, raise_err=True)
        assert cfg.get_block_containing_address(addr, raise_err=False) is None

    for addr in ['', 'idashfiasdjijfj', 'DONTGETTHISFUNC']:
        with pytest.raises(ValueError):
            cfg.get_function_by_name(addr, raise_err=True)
        assert cfg.get_function_by_name(addr, raise_err=False) is None


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_cfgs_inputs(cfg_func):
    """Tests that the inputs are converted into the expected CFG() when passed to the initializer"""
    res = cfg_func(build_level='cfg')
    cfg: CFG = res['cfg']

    for input in res['inputs']:
        new_cfg = CFG(input)
        assert cfg.update_metadata({'file_type': new_cfg.metadata['file_type']}) == new_cfg


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_cfgs_eq_and_hash(cfg_func, print_hashes):
    """Tests all manual cfgs are equal correctly with working strict hashes"""
    res = cfg_func(build_level='cfg')
    cfg: CFG = res['cfg']

    if print_hashes:
        print(__file__+'-manual_cfg-'+res['file'], hash(cfg))
    else:
        assert hash(cfg) == res['expected']['cfg_hash']

    all_cfgs = []
    hashes = set()

    assert cfg == cfg
    assert hash(cfg) == hash(cfg)
    assert hash(cfg) not in hashes
    all_cfgs.append(copy.deepcopy(cfg))
    hashes.add(hash(cfg))

    cfg.update_metadata({'arch': 'x86'})

    assert cfg == cfg
    assert hash(cfg) == hash(cfg)
    assert hash(cfg) not in hashes
    assert all(cfg != c for c in all_cfgs)
    all_cfgs.append(copy.deepcopy(cfg))
    hashes.add(hash(cfg))

    cfg.functions[0].metadata.update({'bananas': 10})

    assert cfg == cfg
    assert hash(cfg) == hash(cfg)
    assert hash(cfg) not in hashes
    assert all(cfg != c for c in all_cfgs)
    all_cfgs.append(copy.deepcopy(cfg))
    hashes.add(hash(cfg))

    cfg.normalizer = X86DeepSemanticNormalizer()

    assert cfg == cfg
    assert hash(cfg) == hash(cfg)
    assert hash(cfg) not in hashes, 'Normalizer'
    assert all(cfg != c for c in all_cfgs)
    all_cfgs.append(copy.deepcopy(cfg))
    hashes.add(hash(cfg))


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_cfgs_conversions(cfg_func):
    """Tests all manual cfgs can be converted to/from different formats and still be equal"""
    res = cfg_func(build_level='cfg')
    cfg: CFG = res['cfg']
    funcs, blocks, expected = res['functions'], res['blocks'], res['expected']

    check_expected(cfg, blocks, funcs, expected)

    # Reading in from file
    outpath = os.path.join(os.path.dirname(__file__), '_temp_test_cfg.txt')
    for input_str in res['inputs']:
        with open(outpath, 'w') as f:
            f.write(input_str)

        for read_type in ['read', 'readio', 'readlines', 'readstr']:
            if read_type == 'read':
                cfg2 = CFG(outpath)
            elif read_type == 'readio':
                with open(outpath, 'r') as f:
                    cfg2 = CFG(f)
            elif read_type == 'readlines':
                with open(outpath, 'r') as f:
                    cfg2 = CFG(f.readlines())
            elif read_type == 'readstr':
                with open(outpath, 'r') as f:
                    cfg2 = CFG(f.read())
            
            # Fix the metadata
            cfg2.metadata = cfg.metadata

            check_expected(cfg2, blocks, funcs, expected)
            assert cfg == cfg2
    os.remove(outpath)

    # Pickling
    cfg2 = pickle.loads(pickle.dumps(cfg))
    check_expected(cfg2, blocks, funcs, expected)
    assert cfg2 == cfg
    cfg2.blocks[2].edges_in.add(CFGEdge(cfg2.blocks[-1], cfg2.blocks[2], EdgeType.NORMAL))
    assert cfg2 != cfg

    # Deepcopy
    cfg2 = copy.deepcopy(cfg)
    check_expected(cfg2, blocks, funcs, expected)
    assert cfg2 == cfg
    new_block = CFGBasicBlock(address=1242134213)
    cfg2.functions[0].blocks.append(new_block)
    cfg2.blocks_dict[new_block.address] = new_block
    assert cfg2 != cfg

    # Networkx
    cfg2 = CFG.from_networkx(cfg.to_networkx())
    check_expected(cfg2, blocks, funcs, expected)
    assert cfg2 == cfg
    cfg2.functions[0].metadata.update({'b': 3})
    cfg2.blocks[3].metadata.update({'a': 10})
    assert cfg2 != cfg

    # Networkx constructor
    cfg2 = CFG(cfg.to_networkx())
    check_expected(cfg2, blocks, funcs, expected)
    assert cfg2 == cfg

    # CFG build code
    build_func_str = ("def _build_cfg():\n%s\nreturn __auto_cfg" % cfg.get_cfg_build_code()).replace('\n', '\n    ')
    exec(build_func_str)
    cfg2 = locals()['_build_cfg']()
    check_expected(cfg2, blocks, funcs, expected)
    assert cfg2 == cfg
    cfg2.functions_dict[123456789] = CFGFunction(address=123456789)
    assert cfg2 != cfg

    # Copy constructor
    assert CFG(cfg) == cfg


def check_expected(cfg: CFG, blocks, funcs, expected):
    """Checks all the ways down to make sure the cfg has the expected values (useful for post-copying)"""
    assert [b.address for b in cfg.blocks] == expected['sorted_block_order']
    assert [f.address for f in cfg.functions] == expected['sorted_func_order']
    assert cfg.blocks == [blocks[a] for a in expected['sorted_block_order']]
    assert cfg.functions == [funcs[a] for a in expected['sorted_func_order']]
    assert cfg.num_blocks == sum(expected['num_blocks'].values())
    assert cfg.num_functions == expected['num_functions'] == len(expected['sorted_func_order'])
    assert cfg.asm_counts == expected['asm_counts']
    assert cfg.metadata == {'architecture': get_architecture(expected['architecture']).value[0]} or cfg.metadata == {}
    assert cfg.architecture == get_architecture(expected['architecture'])
    assert isinstance(hash(cfg), int)
    assert isinstance(str(cfg), str)
    assert isinstance(repr(cfg), str)

    for f in cfg.functions:
        assert f.num_blocks == expected['num_blocks'][f.address]
        assert f.num_asm_lines == expected['num_asm_lines_per_function'][f.address]
        assert f.is_root_function == expected['is_root_function'][f.address]
        assert f.is_recursive == expected['is_recursive'][f.address]
        assert f.is_extern_function == expected['is_extern_function'][f.address]
        assert f.is_intern_function == expected['is_intern_function'][f.address]
        assert f.asm_counts == expected['asm_counts_per_function'][f.address]
        assert set(b.address for b in f.called_by) == expected['called_by'][f.address]

        assert f.function_entry_block.address == expected['function_entry_block'][f.address]
        assert f.blocks == [blocks[b.address] for b in f.blocks]
        assert f.metadata == {}
    
    for b in cfg.blocks:
        assert b.num_asm_lines == expected['num_asm_lines_per_block'][b.address]
        assert b.asm_counts == expected['asm_counts_per_block'][b.address]
        assert b.metadata == {}