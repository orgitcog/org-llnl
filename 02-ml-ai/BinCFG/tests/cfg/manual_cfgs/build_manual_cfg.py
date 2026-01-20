"""Script to make the manual cfg's"""
from collections import Counter
from bincfg import EdgeType, MemCFG
from .fake_classes import FakeEdgeType
from ..test_normalize_cfg import ARCH_NORMS


FUNC_EDGE_TYPES = [FakeEdgeType.FUNCTION_CALL, EdgeType.FUNCTION_CALL]


def build_expected_dict(cfg_res, arch):
    """Makes the 'expected' dictionary values. Returns the string to copy/paste into the test file
    
    Args:
        cfg_res (dict): should be a dictionary containing the keys 'blocks' and 'functions', which should be list of 
            basic blocks and functions respectively. They don't have to be actual CFG* objects (and probably shouldn't 
            to reduce bugs), they just have to have similar attributes (like using the fake_classes)
        arch (str): the expected architecture of these blocks/functions
    """

    expected = {
        'sorted_func_order': [cfg_res['functions'][a].address for a in sorted([f.address for f in cfg_res['functions'].values()])],
        'sorted_block_order': [cfg_res['blocks'][a].address for a in sorted([b.address for b in cfg_res['blocks'].values()])],
        'architecture': arch,
        'num_blocks': {
            k: len(f.blocks) for k, f in cfg_res['functions'].items()
        },
        'num_asm_lines_per_block': {
            k: len(b.asm_lines) for k, b in cfg_res['blocks'].items()
        },
        'num_asm_lines_per_function': {
            k: sum(len(b.asm_lines) for b in f.blocks) for k, f in cfg_res['functions'].items()
        },
        'num_functions': len(cfg_res['functions']),
        'is_root_function': {
            k: all(e.edge_type not in FUNC_EDGE_TYPES or e.to_block.address not in set(b1.address for b1 in f.blocks) for b in cfg_res['blocks'].values() for e in b.edges_out) for k, f in cfg_res['functions'].items()
        },
        'is_recursive': {
            k: any(e.to_block.address in set(b.address for b in f.blocks) and e.edge_type in FUNC_EDGE_TYPES for b in f.blocks for e in b.edges_out) for k, f in cfg_res['functions'].items()
        },
        'is_extern_function': {
            k: f.is_extern_function for k, f in cfg_res['functions'].items()
        },
        'is_intern_function': {
            k: not f.is_extern_function for k, f in cfg_res['functions'].items()
        },
        'function_entry_block': {
            k: [b.address for b in f.blocks if b.address == f.address][0] for k, f in cfg_res['functions'].items()
        },
        'called_by': {
            k: set(b.address for b in cfg_res['blocks'].values() if any((e.to_block.address in set(b2.address for b2 in f.blocks) and e.edge_type in FUNC_EDGE_TYPES) for e in b.edges_out)) for k, f in cfg_res['functions'].items()
        },
        'function_hashes': {f.address: hash(f) for f in cfg_res['functions'].values()},
        'block_hashes': {b.address: hash(b) for b in cfg_res['blocks'].values()},
        'cfg_hash': hash(cfg_res['cfg']),
        'memcfg_hashes': {k: hash(MemCFG(cfg_res['cfg'], normalizer=n)) for k, n in [(k+'-'+tl, nc(tokenization_level=tl, **nk)) for k, (nc, nk) in ARCH_NORMS[cfg_res['cfg'].architecture].items() for tl in ['op', 'inst']]}, 
        'asm_counts_per_block': {
            k: dict(Counter(b.asm_lines)) for k, b in cfg_res['blocks'].items()
        },
        'asm_counts_per_function': {
            k: dict(Counter(l for b in f.blocks for l in b.asm_lines)) for k, f in cfg_res['functions'].items()
        },
        'asm_counts': dict(Counter(l for f in cfg_res['functions'].values() for b in f.blocks for l in b.asm_lines)),
    }

    def v_str(k, v):
        if k in ['asm_counts_per_function']:
            return '{\n\t\t%s\n\t}' % '\n\t\t'.join(['%s: %s,' % (repr(k), '{\n\t\t\t%s\n\t\t}' % '\n\t\t\t'.join(['%s: %s,' % (repr(k), repr(v2)) for k, v2 in v1.items()])) for k, v1 in v.items()])
        elif k in ['asm_counts_per_block', 'asm_counts']:
            return '{\n\t\t%s\n\t}' % '\n\t\t'.join(['%s: %s,' % (repr(k), repr(v)) for k, v in v.items()])
        return repr(v)
    print_str = 'expected = {\n\t%s\n}' % '\n\t'.join(['%s: %s,' % (repr(k), v_str(k, v)) for k, v in expected.items()])

    return print_str