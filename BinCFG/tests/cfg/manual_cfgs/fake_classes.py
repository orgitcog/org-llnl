"""A bunch of fake classes mirroring CFG* classes, but without anything happening to use for testing"""

class FakeCFG:
    """Fake version of a CFG"""
    def __init__(self):
        pass

    def get_block(self, addr):
        """Needed just for __str__ call on CFGFunctions, not really used elsewhere in testing"""
        assert isinstance(addr, int)
        return [b for b in self.blocks if b.address == addr][0]


class FakeCFGFunction:
    """Fake version of a CFGFunction"""
    def __init__(self, parent_cfg, address, name, is_extern_function, metadata):
        self.parent_cfg = parent_cfg
        self.address = address
        self.name = name
        self.is_extern_function = is_extern_function
        self.metadata = metadata
    
    def __str__(self):
        return "Func-%d" % self.address
    
    def __repr__(self):
        return str(self)


class FakeCFGBasicBlock:
    """Fake version of a CFGBasicBlock"""
    def __init__(self, parent_function, address, metadata, asm_memory_addresses, asm_lines):
        self.parent_function = parent_function
        self.address = address
        self.metadata = metadata
        self.asm_memory_addresses = asm_memory_addresses
        self.asm_lines = asm_lines
    
    def __str__(self):
        return "BB-%d" % self.address
    
    def __repr__(self):
        return str(self)


class FakeCFGEdge:
    """Fake version of a CFGEdge"""
    def __init__(self, from_block, to_block, edge_type):
        self.from_block = from_block
        self.to_block = to_block
        self.edge_type = edge_type


class FakeEdgeType:
    """Fake version of EdgeType enum"""
    NORMAL = 'normal'
    FUNCTION_CALL = 'function_call'