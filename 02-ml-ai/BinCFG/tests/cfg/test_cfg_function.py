import pytest
from bincfg import CFGFunction, CFGBasicBlock
from .manual_cfgs import get_all_manual_cfg_functions


def check_builtins(func):
    """Checks builtins can be called and return expected value"""
    assert isinstance(str(func), str)
    assert isinstance(repr(func), str)
    assert isinstance(hash(func), int)


@pytest.mark.parametrize('args', [
    (None, None, 'apples', [], False, None),
    ('aaa', 13, '', None, True, {}),
    ('aaa', CFGBasicBlock(address=37), None, [1, 2, 3], True, {'a': 10, 'b': 'b'}),
    (object(), '0x14', None, (1, 2, 3), False, None),
    (object(), '0x14', None, set([1, 2, 3]), False, {}),  # Idk if set always converts to list the same or not...
])
def test_construct_cfg_func(args):
    """Can construct correctly"""
    parent_cfg, address, name, blocks, is_extern_func, metadata = args
    func = CFGFunction(parent_cfg=parent_cfg, address=address, name=name, blocks=blocks, is_extern_function=is_extern_func, metadata=metadata)

    assert func.parent_cfg is parent_cfg
    assert isinstance(func.address, int)
    assert func.address == (-1 if address is None else address if isinstance(address, int) else int(address, 0) if isinstance(address, str) else address.address)
    
    assert isinstance(func.name, str)
    if isinstance(name, str) and name != '':
        assert func.name == name
    
    assert isinstance(func.blocks, list)
    assert func.blocks == ([] if blocks is None else list(blocks))
    assert isinstance(func.is_extern_function, bool)
    assert func.is_extern_function == is_extern_func

    assert isinstance(func.metadata, dict)
    assert func.metadata == ({} if metadata is None else metadata)


def test_hash_values():
    """Tests that changing values changes the hash correctly"""
    func = CFGFunction()
    hashes = set()

    assert hash(func) not in hashes
    hashes.add(hash(func))

    func.metadata.update({'a': 10})

    assert hash(func) not in hashes
    hashes.add(hash(func))

    func.blocks.append(CFGBasicBlock())

    assert hash(func) not in hashes
    hashes.add(hash(func))

    func.name = 'aa'

    assert hash(func) not in hashes
    hashes.add(hash(func))

    func.address = 10

    assert hash(func) not in hashes
    hashes.add(hash(func))

    func.blocks[0].metadata.update({'b': 10})

    assert hash(func) not in hashes
    hashes.add(hash(func))


def test_empty_func_builtins():
    """Can still call builtins even with None values"""
    check_builtins(CFGFunction())

    func1 = CFGFunction(blocks=[CFGBasicBlock()])
    func2 = CFGFunction(blocks=[CFGBasicBlock()])
    assert func1 == func2
    assert hash(func1) == hash(func2)

    func2.blocks.append(CFGBasicBlock())
    assert func1 != func2
    assert hash(func1) != hash(func2)

    func3 = CFGFunction(blocks=[CFGBasicBlock(), CFGBasicBlock()])
    assert func2 == func3
    assert hash(func2) == hash(func3)


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_cfg_functions(cfg_func, print_hashes):
    """Tests the functions within the manual cfg building functions"""
    res = cfg_func(build_level='function')
    funcs, expected = res['functions'], res['expected']

    if print_hashes:
        print(__file__+'-'+res['file'], {f.address: hash(f) for f in funcs.values()})
    else:
        assert {f.address: hash(f) for f in funcs.values()} == expected['function_hashes']

    assert len(funcs) == len(expected['sorted_func_order']) == expected['num_functions']

    for func in funcs.values():
        check_builtins(func)

        assert func.num_blocks == expected['num_blocks'][func.address]
        assert func.num_asm_lines == expected['num_asm_lines_per_function'][func.address]

        assert func.is_root_function == expected['is_root_function'][func.address]
        assert func.is_recursive == expected['is_recursive'][func.address]
        assert func.is_extern_function == expected['is_extern_function'][func.address]
        assert func.is_intern_function == expected['is_intern_function'][func.address]

        assert func.function_entry_block.address == expected['function_entry_block'][func.address]
        assert func.function_entry_block == res['blocks'][expected['function_entry_block'][func.address]]

        assert set(b.address for b in func.called_by) == expected['called_by'][func.address]

        counts = dict(func.asm_counts)
        assert counts.keys() == expected['asm_counts_per_function'][func.address].keys()
        for k, v in counts.items():
            assert v == expected['asm_counts_per_function'][func.address][k]
