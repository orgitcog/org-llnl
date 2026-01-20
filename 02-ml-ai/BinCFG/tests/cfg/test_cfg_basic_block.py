import pytest
from bincfg import CFGBasicBlock, CFGEdge, EdgeType
from .manual_cfgs import get_all_manual_cfg_functions


def build_graph():
    """Make a complex-ish graph of some basic blocks to test with

    At least one node here has one of these properties:

        * function entry
        * function call
        * multi-function call
        * function jump
        * 0,1,>1 normal parents
        * 0,1,>1 normal children
    
    Graph structure (numbers are nodes [correspond to their memory address], numbers with a '#' are in a different function,
    '---' and '|' is normal edge, '===' is a double-normal-edge [one in each direction], '...' is function call edge, 
    ';;;' is a double-call-edge, '~>' next to a node means a normal self-loop):

                               ........  
                               .      .
                               \/     .
                     1 <=====> 0 ---> 2 .....> 3#
                                \    /         ^
                                 \/\/          |
                      ~>6 <;;;;;> 4 .........> 5
    """
    class TempCFGFunction:
        def __init__(self, address):
            self.address = address
            self.name = None
    parent_1, parent_2 = TempCFGFunction(0), TempCFGFunction(140)

    # Construct the basic blocks themselves, leave mostly empty, but set parent function
    bb = {i: CFGBasicBlock(address=i, parent_function=parent_2 if i in [3] else parent_1) for i in range(7)}

    # Construct the edges
    ei, eo = {}, {}
    ei[0] = set([CFGEdge(bb[2], bb[0], 'function_call'), CFGEdge(bb[1], bb[0], 'normal')])
    eo[0] = set([CFGEdge(bb[0], bb[1], 'normal'), CFGEdge(bb[0], bb[2], 'normal'), CFGEdge(bb[0], bb[4], 'normal')])

    ei[1] = set([CFGEdge(bb[0], bb[1], 'normal')])
    eo[1] = set([CFGEdge(bb[1], bb[0], 'normal')])

    ei[2] = set([CFGEdge(bb[0], bb[2], 'normal')])
    eo[2] = set([CFGEdge(bb[2], bb[4], 'normal'), CFGEdge(bb[2], bb[0], 'function_call'), CFGEdge(bb[2], bb[3], 'function_call')])

    ei[3] = set([CFGEdge(bb[5], bb[3], 'normal'), CFGEdge(bb[2], bb[3], 'function_call')])
    eo[3] = set()

    ei[4] = set([CFGEdge(bb[0], bb[4], 'normal'), CFGEdge(bb[2], bb[4], 'normal'), CFGEdge(bb[6], bb[4], 'function_call')])
    eo[4] = set([CFGEdge(bb[4], bb[5], 'function_call'), CFGEdge(bb[4], bb[6], 'function_call')])

    ei[5] = set([CFGEdge(bb[4], bb[5], 'function_call')])
    eo[5] = set([CFGEdge(bb[5], bb[3], 'normal')])

    ei[6] = set([CFGEdge(bb[4], bb[6], 'function_call'), CFGEdge(bb[6], bb[6], 'normal')])
    eo[6] = set([CFGEdge(bb[6], bb[4], 'function_call'), CFGEdge(bb[6], bb[6], 'normal')])

    for i in range(max(len(ei), len(eo))):
        bb[i].edges_in = ei[i]
        bb[i].edges_out = eo[i]

    return {'bb': bb, 'ei': ei, 'eo': eo, 'parent_1': parent_1, 'parent_2': parent_2}


def check_builtins(block):
    """Checks builtins can be called and return expected value"""
    assert isinstance(str(block), str)
    assert isinstance(repr(block), str)
    assert isinstance(hash(block), int)


# parent_function, address, edges_in, edges_out, asm_lines, asm_memory_addresses, metadata
@pytest.mark.parametrize('args', [
    [None, '1', set(), None, [], None, None],
    [None, None, None, set(), set(), set(), {}],
    ['aaa', 3, [], tuple([CFGEdge(CFGBasicBlock(), CFGBasicBlock(), 'normal')]), ['a', 'b', 'c'], [1, 2, 3], {'a': 10}],
    ['aaa', None, [], tuple([CFGEdge(CFGBasicBlock(), CFGBasicBlock(), 'normal')]), ['a', 'b', 'c'], [1, 2, 3], {'a': 10}],
])
def test_creation(args):
    """We can create basic blocks in a bunch of ways and it doesn't break"""
    bb = CFGBasicBlock(*args)

    assert isinstance(bb.parent_function, type(args[0]))
    assert isinstance(bb.address, int)
    assert isinstance(bb.edges_in, set)
    assert isinstance(bb.edges_out, set)
    assert isinstance(bb.asm_lines, list)
    assert isinstance(bb.asm_memory_addresses, list)
    assert isinstance(bb.metadata, dict)

    check_builtins(bb)


@pytest.mark.parametrize('direction', ['in', 'out'])
def test_edges_are_sets(direction):
    """Tests that we can pass different objects to edges_in and edges_out, and they will be sets of edges"""
    key = 'edges_' + direction

    blocks = {1: CFGBasicBlock(address='1'), 2: CFGBasicBlock(address='2'), 3: CFGBasicBlock(address='3')}
    edges = {1: CFGEdge(blocks[1], blocks[2], 'normal'), 2: CFGEdge(blocks[2], blocks[1], 'normal'), 3: CFGEdge(blocks[1], blocks[3], 'function_call')}
    
    bb = CFGBasicBlock(**{key: []})
    assert isinstance(getattr(bb, key), set) and len(getattr(bb, key)) == 0
    bb = CFGBasicBlock(**{key: [edges[1]]})
    assert isinstance(getattr(bb, key), set) and len(getattr(bb, key)) == 1
    bb = CFGBasicBlock(**{key: [edges[1], edges[2]]})
    assert isinstance(getattr(bb, key), set) and len(getattr(bb, key)) == 2
    bb = CFGBasicBlock(**{key: [edges[1], edges[1], edges[1]]})
    assert isinstance(getattr(bb, key), set) and len(getattr(bb, key)) == 1

    check_builtins(bb)


def test_init_address():
    """Address is handled correctly when passed to initialization"""
    addr_bb = CFGBasicBlock(address=1)

    bb = CFGBasicBlock(address=1, asm_memory_addresses=[2, 3, 4])
    assert bb.address == 1
    bb = CFGBasicBlock(address='1', asm_memory_addresses=[2, 3, 4])
    assert bb.address == 1
    bb = CFGBasicBlock(address=addr_bb, asm_memory_addresses=[2, 3, 4])
    assert bb.address == 1
    bb = CFGBasicBlock(address=None, asm_memory_addresses=[2, 3, 4])
    assert bb.address == 2
    bb = CFGBasicBlock(address=None)
    assert bb.address == -1

    check_builtins(bb)


def test_basic_properties():
    """Basic @property values work as expected"""
    edges_in = [CFGEdge(CFGBasicBlock(), CFGBasicBlock(), 'normal')]
    edges_out = [CFGEdge(CFGBasicBlock(), CFGBasicBlock(), 'normal'), CFGEdge(CFGBasicBlock(address=1), CFGBasicBlock(), 'normal')]
    asm_lines = ['1', '2', '3', '3']

    bb = CFGBasicBlock(edges_in=edges_in, edges_out=edges_out, asm_lines=asm_lines)

    assert bb.num_asm_lines == len(asm_lines)
    assert bb.num_edges == bb.num_edges_out == len(edges_out)
    assert bb.num_edges_in == len(edges_in)

    for k, v in bb.asm_counts.items():
        if k in ['3']:
            assert v == 2
        else:
            assert v == 1
    assert all(k in bb.asm_counts for k in set(asm_lines))

    check_builtins(bb)


def test_remove_edge():
    """Remove_edge()"""
    edges_in = [CFGEdge(CFGBasicBlock(), CFGBasicBlock(), 'normal')]
    edges_out = [CFGEdge(CFGBasicBlock(), CFGBasicBlock(address=5), 'normal'), CFGEdge(CFGBasicBlock(address=1), CFGBasicBlock(address=2), 'normal')]

    bb = CFGBasicBlock(edges_in=edges_in, edges_out=edges_out)

    assert len(bb.edges_in) == 1
    assert len(bb.edges_out) == 2

    with pytest.raises(TypeError):
        bb.remove_edge(None)
    with pytest.raises(ValueError):
        bb.remove_edge(CFGEdge(CFGBasicBlock(), CFGBasicBlock(), 'function_call'))

    bb.remove_edge(edges_out[-1])
    bb.remove_edge(CFGEdge(CFGBasicBlock(), CFGBasicBlock(), 'normal'))

    assert len(bb.edges_in) == 0
    assert len(bb.edges_out) == 1
    assert list(bb.edges_out)[0].from_block.address == -1

    check_builtins(bb)


def test_graph():
    """Test some graph/edge properties"""
    graph = build_graph()

    # Check properties
    for i, b in graph['bb'].items():
        assert b.is_function_entry == (i in [0]), i
        assert b.is_function_call == (i in [2, 4, 6]), i
        assert b.is_function_jump == (i in [5]), i
        assert b.is_multi_function_call == (i in [2, 4]), i
        assert b.all_edges == graph['ei'][i].union(graph['eo'][i]), i

        for edge in b.edges_in:
            assert edge in edge.from_block.edges_out, "in - %d - %s" % (i, edge)
        for edge in b.edges_out:
            assert edge in edge.to_block.edges_in, "out - %d - %s" % (i, edge)

        check_builtins(b)


def test_get_sorted_edges():
    """get_sorted_edges() function"""
    graph = build_graph()

    for i, b in graph['bb'].items():
        # Get lists/sets, check all return correct edge types
        func_out, func_in, norm_out, norm_in = b.get_sorted_edges(edge_types=['function_call', 'normal'], direction=['out', 'in'], as_sets=True)
        func_outl, func_inl, norm_outl, norm_inl = b.get_sorted_edges(edge_types=['function_call', 'normal'], direction=['out', 'in'])
        assert all(e.edge_type == EdgeType.FUNCTION_CALL for e in func_out), i
        assert all(e.edge_type == EdgeType.FUNCTION_CALL for e in func_in), i
        assert all(e.edge_type == EdgeType.NORMAL for e in norm_out), i
        assert all(e.edge_type == EdgeType.NORMAL for e in norm_in), i

        # Check getting list is sorted set
        assert func_outl == sorted(list(func_out), key=lambda edge: edge.to_block.address), i
        assert func_inl == sorted(list(func_in), key=lambda edge: edge.from_block.address), i
        assert norm_outl == sorted(list(norm_out), key=lambda edge: edge.to_block.address), i
        assert norm_inl == sorted(list(norm_in), key=lambda edge: edge.from_block.address), i

        # Check all the expected edges are present
        assert func_out == set(e for e in graph['eo'][i] if e.edge_type is EdgeType.FUNCTION_CALL), i
        assert func_in == set(e for e in graph['ei'][i] if e.edge_type is EdgeType.FUNCTION_CALL), "%d - %s" % (i, b.edges_in)
        assert norm_out == set(e for e in graph['eo'][i] if e.edge_type is EdgeType.NORMAL), i
        assert norm_in == set(e for e in graph['ei'][i] if e.edge_type is EdgeType.NORMAL), i

        # Check other calls get expected sets
        assert [func_in, func_out, norm_in, norm_out] == b.get_sorted_edges(edge_types=[EdgeType.FUNCTION_CALL, EdgeType.NORMAL], direction=['in', 'out'], as_sets=True), i
        assert [func_in, func_out] == b.get_sorted_edges(edge_types=EdgeType.FUNCTION_CALL, direction=None, as_sets=True), i
        assert [norm_in] == b.get_sorted_edges(edge_types='normal', direction='in', as_sets=True), i
        assert [norm_out] == b.get_sorted_edges(edge_types='normal', direction='out', as_sets=True), i


def test_has_edge():
    """has_edge() and calls()"""
    graph = build_graph()

    for i, b in graph['bb'].items():
        for edge_dir, edge_set, other_edge_set in zip(['out', 'in'], [b.edges_out, b.edges_in], [b.edges_in, b.edges_out]):
            for edge in edge_set:
                for af in ['0x{0:x}', '{0:d}', 'int', 'block']:
                    
                    # Format the to/from addresses
                    to_a, from_a = edge.to_block.address, edge.from_block.address
                    to_a = to_a if af == 'int' else CFGBasicBlock(address=to_a) if af == 'block' else af.format(to_a)
                    from_a = from_a if af == 'int' else CFGBasicBlock(address=from_a) if af == 'block' else af.format(from_a)

                    a1 = to_a if edge_dir == 'out' else from_a
                    a2 = from_a if edge_dir == 'out' else to_a

                    # Check the edge exists
                    assert b.has_edge(address=a1, edge_types=None, direction=None)
                    assert b.has_edge(address=a1, edge_types=edge.edge_type, direction=None)
                    assert b.has_edge(address=a1, edge_types=None, direction=edge_dir)
                    assert b.has_edge(address=a1, edge_types=[edge.edge_type], direction=edge_dir)
                    
                    # Check calls()
                    if edge.edge_type == EdgeType.FUNCTION_CALL and edge_dir == 'out':
                        assert b.calls(address=a1)

                    # If the reverse edge doesn't exist, check it doesn't exist
                    if CFGEdge(edge.to_block, edge.from_block, edge.edge_type) not in other_edge_set:
                        assert not b.has_edge(address=a2, edge_types=None, direction=None)
                        assert not b.has_edge(address=a2, edge_types=edge.edge_type, direction=None)
                        assert not b.has_edge(address=a2, edge_types=None, direction=edge_dir)
                        assert not b.has_edge(address=a2, edge_types=[edge.edge_type], direction=edge_dir)
                    
                    # Check some random values that should never match
                    assert not b.has_edge(address='1234214', edge_types=None, direction=None)
                    with pytest.raises(TypeError):
                        b.has_edge(address=None)
                    with pytest.raises(ValueError):
                        b.has_edge(address=-1)
                    with pytest.raises(TypeError):
                        b.has_edge(0, edge_types='aaa')
                    with pytest.raises(ValueError):
                        b.has_edge(0, direction='aaa')

_EQ_BLOCKS = [
    ([None, None, None, None, None, None], 'a', 2303513372440043212),
    ([-1, set(), set(), [], [], {}], 'a', 2303513372440043212),
    ([-1, set(), set(), [], [], {'vals': 10}], 'b', 517139234453982774),
    ([10, set(), set(), [], [], {}], 'c', 2058733285442219295),
    ([None, set(), set(), [1], [1], {}], 'd', 412735380168654700),
    ([1, set(), set(), [1], [1], {}], 'd', 412735380168654700),
    ([10, set([1, 2, 3]), set(['a', 'b', 'c']), [1, 2], [5, 7], {'meta': {'a': 10, 'b': 'b'}}], 'e', 1345849587625969234),
    ([10, set([1, 2, 3]), set(['a', 'b', 'c']), [1, 2], [5, 7], {'meta': {'a': 10, 'b': 'b'}}], 'e', 1345849587625969234),
    ([10, set([1, 2]), set(['a', 'b', 'c']), [1, 2], [5, 7], {'meta': {'a': 10, 'b': 'b'}}], 'f', 1917795210026091466),
    ([10, set([1, 2, 3]), set(['a', 'b', 'c']), [1, 2], [5, 7], {'meta': {'a': 10, 'b2': 'b'}}], 'g', 1288331464110070392),
]
@pytest.mark.parametrize('args2,v2,h2', _EQ_BLOCKS)
@pytest.mark.parametrize('args1,v1,h1', _EQ_BLOCKS)
def test_eq_and_hash(args1, v1, h1, args2, v2, h2, print_hashes):
    """Equality and hashing
    
    Check for: 'address' - int, 'edges_in' - set, 'edges_out' - set, 'asm_lines' - list, 'asm_memory_addresses' - list, 
    'metadata' - dict,
    """
    def _build_b(args):
        address, edges_in, edges_out, asm_lines, asm_memory_addresses, metadata = args
        b = CFGBasicBlock(address=address if address != -1 else None, edges_in=edges_in, edges_out=edges_out, asm_lines=asm_lines, 
                          asm_memory_addresses=asm_memory_addresses, metadata=metadata)
        b.address = address if address == -1 else b.address
        return b
    
    b1, b2 = _build_b(args1), _build_b(args2)

    if print_hashes:
        print(__file__+'-eq_and_hash', v1, hash(b1), v2, hash(b2))
    else:  # Don't test while printing
        assert hash(b1) == h1
        assert hash(b2) == h2

    check_builtins(b1)
    check_builtins(b2)

    if v1 == v2:
        assert b1 == b2
        assert b2 == b1
        assert hash(b1) == hash(b2)
    else:
        assert b1 != b2
        assert b2 != b1
        assert hash(b1) != hash(b2)


@pytest.mark.parametrize('cfg_func', get_all_manual_cfg_functions())
def test_manual_cfg_blocks(cfg_func, print_hashes):
    """Building the manual_cfg and checking its blocks"""
    res = cfg_func(build_level='block')
    blocks, expected = res['blocks'], res['expected']

    if print_hashes:
        print(__file__+'-manual_cfg-'+res['file'], {b.address: hash(b) for b in blocks.values()})
    else:
        assert {b.address: hash(b) for b in blocks.values()} == expected['block_hashes']

    assert len(blocks) == len(expected['sorted_block_order'])

    for b in blocks.values():
        check_builtins(b)
        assert b.num_asm_lines == expected['num_asm_lines_per_block'][b.address]

        counts = dict(b.asm_counts)
        assert counts.keys() == expected['asm_counts_per_block'][b.address].keys()
        for k, v in counts.items():
            assert v == expected['asm_counts_per_block'][b.address][k]
        
        for e in b.edges_out:
            assert CFGEdge(b, e.to_block, e.edge_type) in e.to_block.edges_in
        for e in b.edges_in:
            assert CFGEdge(e.from_block, b, e.edge_type) in e.from_block.edges_out
