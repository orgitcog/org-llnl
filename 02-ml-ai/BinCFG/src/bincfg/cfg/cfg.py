import numpy as np
import copy
import bincfg
import pickle
import re
from collections import Counter, namedtuple
from .cfg_parsers import parse_cfg_data
from .cfg_function import CFGFunction
from .cfg_edge import CFGEdge, EdgeType
from .cfg_basic_block import CFGBasicBlock
from ..utils import get_address, eq_obj, hash_obj, get_module
from ..utils.type_utils import *
from ..normalization import normalize_cfg_data, Architectures, get_architecture


# Extra bytes to pad the insertion of libraries into the CFG so we don't mess up other assembly instructions
_INSERTION_PADDING_BYTES: 'int' = 16


class CFG:
    """A Control Flow Graph (CFG) representation of a binary

    
    Parameters
    ----------
    data: `Optional[Union[str, TextIO, Sequence[str], SmdaReport]]`
        the data to use to make this CFG. Data type will be inferred based on the data passed:

            * string: either string with newline characters that will be split on all newlines and as a known disassembler
              format, or a string with no newline characters that will be treated as a filename.
            * Sequence of string: will be treated as already-read-in disassembler file split on newlines
            * open file object: will be read in using `.readlines`, then treated as disassembler input
            * SmdaReport: output from smda disassembly

    normalizer: `Optional[Union[str, NormalizerType]]`
        the normalizer to use to force-renormalize the incoming CFG, or None to not normalize
    metadata: `Optional[dict]`
        a dictionary of metadata to add to this CFG

        NOTE: passed dictionary will be shallow copied
    using_tokens: `Optional[Union[dict[str, int], AtomicTokenDict]]`
        optional token dictionary to use when initializing and normalizing. Only used if normalizer is not None
    """

    normalizer: 'Union[NormalizerType, None]'
    """The normalizer used to normalize assembly lines in this ``CFG``, or None if they have not been normalized"""

    metadata: 'dict'
    """Dictionary of metadata associated with this ``CFG``"""

    functions_dict: 'dict[int, CFGFunction]'
    """Dictionary mapping integer function addresses to their ``CFGFunction`` objects"""

    blocks_dict: 'dict[int, CFGBasicBlock]'
    """Dictionary mapping integer basic block addresses to their ``CFGBasicBlock`` objects"""

    def __init__(self, data: 'CFGInputDataType' = None, normalizer: 'Optional[Union[str, NormalizerType]]' = None, 
                 metadata: 'Optional[dict]' = None, using_tokens: 'Optional[TokenDictType]' = None):
        # These store functions/blocks while allowing for O(1) lookup by address
        self.functions_dict: 'dict[int, CFGFunction]' = {}
        self.blocks_dict: 'dict[int, CFGBasicBlock]' = {}

        self.normalizer: 'Union[NormalizerType, None]' = None
        self.metadata: 'dict' = {} if metadata is None else metadata.copy()

        # If data is not None, parse it
        if data is not None:
            parse_cfg_data(self, data)

        # Finally, normalize if needed
        if normalizer is not None:
            self.normalize(normalizer, using_tokens=using_tokens, inplace=True)
    
    def get_function(self, address: 'AddressLike', raise_err: 'bool' = True) -> 'Union[CFGFunction, None]':
        """Returns the function in this ``CFG`` with the given address

        Args:
            address (AddressLike): a string/integer memory address, or an addressable object (EG: CFGBasicBlock/CFGFunction)
            raise_err (bool): if True, will raise an error if the function with the given memory address was 
                not found, otherwise will return None

        Raises:
            ValueError: if the function with the given address could not be found

        Returns:
            Union[CFGFunction, None]: the function with the given address, or None if that function does not exist
        """
        address = get_address(address)
        if address not in self.functions_dict and raise_err:
            raise ValueError("Could not find function with address: (decimal) %d, (hex) 0x%x" % (address, address))
        return self.functions_dict.get(address, None)
    
    def get_function_by_name(self, name: 'str', raise_err: 'bool' = True) -> 'Union[CFGFunction, None]':
        """Returns the function in this ``CFG`` with the given name

        NOTE: if the name of the function is None, then the expected string name to this method would be:
        `"__UNNAMED_FUNC_%d" % func.address`

        Args:
            name (str): the name of the function to get
            raise_err (bool): if True, will raise an error if the function with the given memory address was 
                not found, otherwise will return None

        Raises:
            ValueError: if the function with the given address could not be found

        Returns:
            Union[CFGFunction, None]: the function with the given address, or None if that function does not exist
        """
        for func in self.functions_dict.values():
            if func.name == name:
                return func
        if raise_err:
            raise ValueError("Could not find function with name: %s" % repr(name))
        return None
    
    def get_block(self, address: 'AddressLike', raise_err: 'bool' = True) -> 'Union[CFGBasicBlock, None]':
        """Returns the basic block in this CFG with the given address

        Args:
            address (AddressLike): a string/integer memory address, or an addressable object (EG: CFGBasicBlock/CFGFunction)
            raise_err (bool): if True, will raise an error if the function with the given memory address was 
                not found, otherwise will return None

        Raises:
            ValueError: if the basic block with the given address could not be found

        Returns:
            Union[CFGBasicBlock, None]: the basic block with the given address
        """
        address = get_address(address)
        if address not in self.blocks_dict and raise_err:
            raise ValueError("Could not find basic block with address: (decimal) %d, (hex) %x" % (address, address))
        return self.blocks_dict.get(address, None)
    
    def get_block_containing_address(self, address: 'AddressLike', raise_err: 'bool' = True) -> 'Union[CFGBasicBlock, None]':
        """Returns the basic block in this CFG that contains the given address at the start of one of its instructions

        This will lazily compute an instruction lookup dictionary mapping addresses to the blocks that contain them

        NOTE: this will only return a block if the address is either equal to the block's address, or if it is exactly
        equal to one of the addresses for an assembly instruction in a block's `.asm_memory_addresses` list
        
        Args:
            address (AddressLike): a string/integer memory address, or an addressable object (EG: CFGBasicBlock/CFGFunction)
            raise_err (bool): if True, will raise an error if the function with the given memory address was 
                not found, otherwise will return None

        Raises:
            ValueError: if the basic block containing the given address could not be found

        Returns:
            Union[CFGBasicBlock, None]: the basic block that contains the given address
        """
        address = get_address(address)

        # Check if we have created an instruction lookup yet or not
        if address in self._inst_lookup:
            return self._inst_lookup[address]
        elif raise_err:
            raise ValueError("Could not find basic block containing the address: (decimal) %d, (hex) %x" % (address, address))
        else:
            return None
    
    @property
    def _inst_lookup(self) -> 'dict[int, CFGBasicBlock]':
        """Maps addresses to basic blocks containing those addresses. Will dynamically create dict if not present"""
        # Make the instruction address lookup if it doesn't already exist
        if not hasattr(self, '_inst_lookup_dict'):
            self._inst_lookup_dict = {}

            for block in self.blocks:
                for block_addr in (block.asm_memory_addresses + [block.address]):
                    self._inst_lookup_dict[block_addr] = block
        
        return self._inst_lookup_dict
    
    def add_function(self, *functions: 'CFGFunction', override: 'bool' = False) -> None:
        """Adds the given function(s) to this cfg. This should only be done once the given function(s) have been fully initialized

        This will do some housekeeping things such as:

            * setting the parent_cfg and parent_function attributes of functions and blocks respectively
            * adding missing edges to their associated edges_out and edges_in
            * converting edges from (None/address, None/address, edge_type) tuples into CFGEdge() objects
            * adding from_block and to_block in new edges if missing
            * functions with no address will have their address be that of the smallest addressed block in their blocks, if present

        Args:
            function (CFGFunction): arbitrary number of CFGFunction's to add
            override (bool): if False, an error will be raised if a function or basic block contains an address that
                already exists in this CFG. If True, then that error will not be raised and those functions/basic blocks
                will be overriden (which has unsupported behavior). Defaults to False.
        """
        for func in functions:
            # Check that the function has an address
            if func.address == -1 and len(func.blocks) > 0:
                func.address = min(b.address for b in func.blocks)

            # Check for bad function type, address being None, or function address already existing
            if not isinstance(func, CFGFunction):
                raise TypeError("Can only add function of type CFGFunction, not '%s'" % type(func).__name__)
            if func.address == -1:
                raise ValueError("Functions must have valid address when adding to CFG: %s" % func)
            if func.address in self.functions_dict:
                if not override:
                    raise ValueError("Function has address 0x%x which already exists in this CFG!" % func.address)
            
            func.parent_cfg = self
            self.functions_dict[func.address] = func

            for block in func.blocks:
                # Check for bad basic blocks
                if block.address is None:
                    raise ValueError("Block cannot have a None address when adding to CFG: %s" % block)
                if block.address in self._inst_lookup:
                    if not override:
                        raise ValueError("Basic block has address 0x%x which already exists in this CFG!" % block.address)
                
                block.parent_function = func
                self.blocks_dict[block.address] = block
        
            # Add all the instruction addresses if that has already been computed
            if hasattr(self, '_inst_lookup_dict'):
                for block in func.blocks:
                    for block_addr in (block.asm_memory_addresses + [block.address]):
                        self._inst_lookup[block_addr] = block
                
        # Check the edges out
        for block in self.blocks:
            block.edges_out = set((CFGEdge(block, e[1] if isinstance(e[1], CFGBasicBlock) else self.get_block(e[1]), e[2]) \
                                   if isinstance(e, tuple) else e) for e in block.edges_out)
            for edge in block.edges_out:
                edge.to_block.edges_in.add(edge)

        # Check the edges in
        for block in self.blocks:
            block.edges_in = set((CFGEdge(e[0] if isinstance(e[0], CFGBasicBlock) else self.get_block(e[0]), block, e[2]) \
                                  if isinstance(e, tuple) else e) for e in block.edges_in)
            for edge in block.edges_in:
                edge.from_block.edges_out.add(edge)

    def insert_library(self, cfg: 'CFG', function_mapping: 'dict[str, int]', offset: 'Optional[int]' = None):
        """WIP. Inserts the cfg of a shared library into this cfg

        This will modify the memory addresses of `cfg` (adding an appropriate offset), then add all of the functions and
        basic blocks from `cfg` into this cfg. Finally, external functions in this cfg that have implemented functions
        in the function_mapping will have normal edges added.

        NOTE: this assumes that no other libraries will be added later that depend on this one that is currently being
        added (otherwise, the external function edges might not be added properly). Make sure you add them in the
        correct order!
        
        Args:
            cfg (CFG): the cfg of the library to insert. It will be copied
            function_mappping (Dict[str, int]): dictionary mapping known exported function names to their addresses
                within `cfg`. While we can sometimes determine these mappings from function names in the new `cfg`,
                that is not always the case (EG: stripping function names from binaries, or compilers/linkers emitting
                aliases for the functions in `cfg`), hence why this parameter exists. If you don't wish to add in new
                normal edges, or if you wish to add them in manually, you can pass an empty dictionary
            offset (Optional[int]): if None, then the library will be inserted in the first available memory location.
                Otherwise this can be an integer memory address to insert the cfg at (this will raise an error if it
                can't fit there)
        """
        # Determine an acceptable offset. We can't just insert at the end or something since we may call this function
        #   multiple times, and binaries can do just about anything that may mess up hard-coded placements
        _min_max = lambda s: (min(s), max(s))

        # Find the size of `cfg` (just the needed memory locations, plus some padding)
        min_addr, max_addr = 2**64, 0
        for block in cfg.blocks:
            new_min, new_max = _min_max(block.asm_memory_addresses)
            min_addr = min(min_addr, new_min)
            max_addr = max(max_addr, new_max)
        cfg_size = max_addr - min_addr + _INSERTION_PADDING_BYTES * 2

        # Sort all min/max's of memory addresses for blocks in this cfg. Insert a 0 so we could insert in beginning
        addresses = np.sort([0] + [s for block in self.blocks for s in _min_max(block.asm_memory_addresses)] + [2 ** 32])

        # If the user didn't pass an offset, determine an appropriate one on our own
        if offset is None:

            # Compute all of the differences to get sizes (all negative or 0 since sorted), get every other one since we 
            #   couldn't place it inside a block
            diffs = np.diff(addresses)[::2]

            # Find the first spot in which we could place the new cfg, raise an error if we can't fit it. Get the original
            #   starting memory address of that location
            loc = np.argwhere(diffs >= cfg_size)
            if len(loc) == 0:
                raise ValueError("Could not find space to insert a library of size %d" % cfg_size)
            offset = addresses[loc[0][0] * 2] + _INSERTION_PADDING_BYTES
        
        # Otherwise, check that the offset the user passed works. It should in an available and large enough gap, and
        #   should be at least _INSERTION_PADDING_BYTES away from the nearest used memory address in this cfg
        else:
            idx = np.searchsorted(addresses, offset, side='left')

            # If the index is even, then it is within a block (note: addresses is always even length-ed, and the 
            #   searchsorted call will always return the index after the last used memory address). Otherwise if
            #   idx is within _INSERTION_PADDING_BYTES of the nearest block, then it is also bad
            # Another note: if the idx is 1, then it doesn't need the padding since it's already at the start of the memory,
            #   but it does need it after for possible instruction lengths
            if idx % 2 == 0 or idx >= len(addresses) or (1 <= idx \
                and (offset - addresses[idx - 1] < _INSERTION_PADDING_BYTES or addresses[idx] - offset - cfg_size < _INSERTION_PADDING_BYTES)):
                raise InvalidInsertionMemoryAddressError("Cannot insert library at address: 0x%x" % offset)

        # Insert all the new functions/basic blocks, adding offsets to the addresses
        edges = []
        for func in cfg.functions:
            new_func = CFGFunction(parent_cfg=self, address=func.address + offset, name=func.name, 
                                    is_extern_func=func.is_extern_function, blocks=None)
            self.functions_dict[new_func.address] = new_func

            for block in func.blocks:
                new_block = CFGBasicBlock(parent_function=new_func, address=block.address + offset, labels=block.labels,
                                            asm_lines=[(a + offset, l) for a, l in block.asm_lines])
                self.blocks_dict[new_block.address] = new_block
                new_func.blocks.append(block)

                # Keep track of the edges, they will be added later with references to the new block objects
                for edge_set in [block.edges_in, block.edges_out]:
                    for edge in edge_set:
                        edges.append((edge.from_block.address + offset, edge.to_block.address + offset, edge.edge_type))
        
        # Add in the edges for resolved external function symbols
        for func in self.functions:
            if func.symbol_name is not None and func.symbol_name in function_mapping:
                extern_func = self.get_function(function_mapping[func.symbol_name] + offset)

                # Assume the function has one block for now, we'll have to fix that later if that isn't true
                if len(func.blocks) != 1:
                    raise ValueError("Attempting to insert resolved symbolic normal edge to external function, but the "
                                        "external function had %d blocks! (expected 1)" % len(func.blocks))
                
                edges.append((func.blocks[0].address, extern_func.address, EdgeType.NORMAL))
        
        # Add in all of the edges
        for from_addr, to_addr, edge_type in edges:
            from_block = self.get_block(from_addr)
            to_block = self.get_block(to_addr)
            new_edge = CFGEdge(from_block, to_block, edge_type)

            from_block.edges_out.add(new_edge)
            to_block.edges_in.add(new_edge)
    
    @property
    def functions(self) -> 'list[CFGFunction]':
        """A list of functions in this CFG (in order of memory address)"""
        return [f[1] for f in sorted(self.functions_dict.items(), key=lambda x: x[0])]
    
    @property
    def blocks(self) -> 'list[CFGBasicBlock]':
        """A list of basic blocks in this CFG (in order of memory address)"""
        return [b[1] for b in sorted(self.blocks_dict.items(), key=lambda x: x[0])]
    
    @property
    def num_blocks(self) -> 'int':
        """The number of basic blocks in this cfg"""
        return len(self.blocks_dict)
    
    @property
    def num_functions(self) -> 'int':
        """The number of functions in this cfg"""
        return len(self.functions_dict)

    @property
    def num_edges(self) -> 'int':
        """The number of edges in this cfg"""
        return sum(b.num_edges for b in self.blocks_dict.values())

    @property
    def num_asm_lines(self) -> 'int':
        """The number of asm lines across all blocks in this cfg"""
        return sum(b.num_asm_lines for b in self.blocks_dict.values())

    @property
    def asm_counts(self) -> 'Mapping[str, int]':
        """A collections.Counter() of all unique assembly lines and their counts in this cfg"""
        return sum((f.asm_counts for f in self.functions_dict.values()), Counter())
    
    @property
    def edges(self) -> 'list[CFGEdge]':
        """A list of all outgoing ``CFGEdge``'s in this ``CFG``"""
        return [e for b in self.blocks for e in b.edges_out]
    
    @property
    def architecture(self) -> 'Architectures':
        """Returns the architecture being used. Currently a WIP
        
        Checks for an 'arch' or 'architecture' key in the metadata and returns it if it is known. Can currently return:
        'java', 'x86'
        """
        for k in ['arch', 'architecture']:
            if k in self.metadata:
                arch = self.metadata[k]
                break
        else:
            auto_detect_assembly_language(self)
            if 'architecture' in self.metadata:
                arch = self.metadata['architecture']
            else:
                raise KeyError("Could not find 'arch' or 'architecture' key in metadata, and failed to autodetect")
        
        return get_architecture(arch)
    
    def update_metadata(self, other: 'dict') -> 'CFG':
        """Updates this CFG's metadata dictionary with the given dictionary, and returns self"""
        self.metadata.update(other)
        return self
    
    def set_tokens(self, tokens: 'TokenDictType') -> 'CFG':
        """Sets this CFG's tokens to the given tokens, and returns self"""
        self.tokens = tokens
        return self

    def to_adjacency_matrix(self, type: 'str' = 'np', sparse: 'bool' = False) -> 'Union[np.ndarray, torch.Tensor]':
        """Returns an adjacency matrix representation of this cfg's graph connections

        Currently is slow because I just convert to a MemCFG, then call that object's to_adjacency_matrix(). I should
        probably speed this up at some point...

        Connections will be directed and have values:

            - 0: No edge
            - 1: Normal edge
            - 2: Function call edge

        See :func:`~bincfg.memcfg.to_adjacency_matrix` for more details

        Args:
            type (str, optional): the type of matrix to return. Defaults to 'np'. Can be:

                - 'np'/'numpy' for a numpy ndarray (dtype: np.int32)
                - 'torch'/'pytorch' for a pytorch tensor (type: LongTensor)
            
            sparse (bool, optional): whether or not the return value should be a sparse matrix. Defaults to False. Has 
                different behaviors based on type:

                - numpy array: returns a 2-tuple of sparse COO representation (indices, values). 
                    NOTE: if you want sparse CSR format, you already have it with self.graph_c and self.graph_r
                - pytorch tensor: returns a pytorch sparse COO tensor. 
                    NOTE: not using sparse CSR format for now since it seems to have less documentation/supportedness. 

        Returns:
            Union[np.ndarray, torch.Tensor]: an adjacency matrix representation of this ``CFG``
        """
        return bincfg.MemCFG(self, normalizer='base' if self.normalizer is None else None).to_adjacency_matrix(type=type, sparse=sparse)
    
    def normalize(self, normalizer: 'Union[str, NormalizerType]', using_tokens: 'Optional[TokenDictType]' = None, 
                  inplace: 'bool' = True, force_renormalize: 'bool' = False) -> 'CFG':
        """Normalizes this cfg.

        Args:
            normalizer (Union[str, NormalizerType]): the normalizer to use. Can be a ``Normalizer`` object, or a 
                string of a built-in normalizer to use
            using_tokens (Optional[TokenDictType]): token dictionary to use when normalizing, or None to normalize from scratch
            inplace (bool): whether or not to normalize inplace
            force_renormalize (bool): by default, this method will only normalize this cfg only if the passed 
                `normalizer` is != `self.normalizer`. However if `force_renormalize=True`, then this will be renormalized
                even if it has been previously normalized with the same normalizer

        Returns:
            CFG: this ``CFG`` normalized
        """
        return normalize_cfg_data(self, normalizer=normalizer, using_tokens=using_tokens, inplace=inplace, 
                                  force_renormalize=force_renormalize)

    def to_networkx(self) -> 'networkx.MultiDiGraph':
        """Converts this CFG to a networkx DiGraph() object
        
        Requires that networkx be installed.

        Creates a new MultiDiGraph() and adds as attributes to that graph:

            - 'normalizer': string name of normalizer, or None if it had none
            - 'metadata': a dictionary of metadata
            - 'functions': a dictionary mapping integer function addresses to named tuples containing its data with the
               structure ('name': `Union[str, None]`, 'is_extern_function': `bool`, 'blocks': `Tuple[int, ...]`, 'metadata': `dict`).

                * The 'name' element (first element) is a string name of the function, or None if it doesn't have a name
                * The 'is_extern_function' element (second element) is True if this function is an extern function, False otherwise.
                  An extern function is one that is located in an external library intended to be found at runtime, and
                  that doesn't have its code here in the CFG, only a small function meant to jump to the external function
                  when loaded at runtime
                * The 'blocks' element (third element) is an arbitrary-length tuple of integers, each integer being the
                  memory address (equivalently, the block_id) of a basic block that is a part of that function. Each
                  basic block is only part of a single function, and each function should have at least one basic block
                * The 'metadata' element (fourth element) is a dictionary of metadata associated with that function.
                  May be empty.
        
        NOTE: we use a multidigraph because edges are directed (in order of control flow), and it is theoretically
        possible (and occurs in some data) to have a node that calls another node, then has a normal edge back out
        to it. This has occured in some libc setup code
        
        Then, each basic block will be added to the graph as nodes. Their id in the graph will be their integer address.
        Each block will have the following attributes:

            - 'asm_lines' (Tuple[str]): tuple of string assembly lines
            - 'asm_memory_addresses (Tuple[int]): tuple of integer assembly line memory addresses, one for each line
              in order. Unless, if these addresses are not present, then this will be an empty tuple
            - 'metadata' (dict): dictionary (possibly empty) of metadata associated with this basic block
        
        Finally, all edges will be added (directed based on control flow direction), and with the attributes:

            - 'edge_type' (str): the edge type, will be 'normal' for normal edges and 'function_call' for function call edges

        """
        # Done like this so I have IDE autocomplete while making sure the package is installed
        _netx = get_module('networkx', raise_err=True)
        import networkx

        # Add all of the functions to a dictionary to set as an attribute on the graph
        functions = {func.address: _NetXTuple(func.name, func._is_extern_function, tuple(b.address for b in func.blocks), func.metadata.copy())
                     for func in self.functions_dict.values()}

        ret = networkx.MultiDiGraph(normalizer=copy.deepcopy(self.normalizer), functions=functions, metadata=self.metadata.copy())
        
        # Add all of the blocks to the graph
        for block in self.blocks_dict.values():
            ret.add_node(block.address, metadata=block.metadata.copy(), asm_memory_addresses=tuple(block.asm_memory_addresses),
                         asm_lines=tuple(block.asm_lines))
        
        # Finally, add all the edges
        for edge in self.edges:
            ret.add_edge(edge.from_block.address, edge.to_block.address, key=edge.edge_type.name.lower())
        
        return ret
    
    @classmethod
    def from_networkx(cls, graph: 'networkx.MultiDiGraph', cfg: 'Optional[CFG]'=None) -> 'CFG':
        """Converts a networkx graph to a CFG

        Expects the graph to have the exact same structure as is shown in CFG().to_networkx()

        Args:
            graph (networkx.MultiDiGraph): the networkx graph
            cfg (Optional[CFG]): can be None to create/return a new CFG object, or an already
                created and empty CFG() object to put data into that one
        """
        ret = CFG() if cfg is None else cfg
        ret.normalizer = graph.graph['normalizer']
        ret.metadata = {} if graph.graph['metadata'] is None else graph.graph['metadata']

        ret.add_function(*[
            CFGFunction(address=addr, name=name, is_extern_function=ef, metadata=meta, blocks=[
                CFGBasicBlock(
                    address=block_addr,
                    edges_out=[(None, a, et) for _, a, et in graph.edges(block_addr, keys=True)],
                    asm_lines=graph.nodes[block_addr]['asm_lines'],
                    asm_memory_addresses=graph.nodes[block_addr]['asm_memory_addresses'],
                    metadata=graph.nodes[block_addr]['metadata']
                )
                for block_addr in blocks
            ])
            for addr, (name, ef, blocks, meta) in graph.graph['functions'].items()
        ])

        return ret
    
    def copy(self) -> 'CFG':
        return pickle.loads(pickle.dumps(self))
    
    def __getstate__(self) -> 'dict':
        """State for pickling"""
        state = {k: v for k, v in self.__dict__.items() if k not in ['functions_dict', 'blocks_dict', '_inst_lookup_dict']}
        state['functions'] = tuple(f._get_pickle_state() for f in self.functions_dict.values())
        return state
    
    def __setstate__(self, state: 'dict'):
        """State for unpickling"""
        for k, v in state.items():
            if k == 'functions':
                continue
            setattr(self, k, v)
        
        self.functions_dict = {func_addr: CFGFunction(parent_cfg=self)._set_pickle_state([func_addr,] + rest) for func_addr, *rest in state['functions']}
        self.blocks_dict = {b.address: b for f in self.functions_dict.values() for b in f.blocks}
        
        # Recreate all the edges
        edges = set(e for b in self.blocks_dict.values() for e in (b._temp_edges_in + b._temp_edges_out))
        for from_addr, to_addr, edge_type in edges:
            edge = CFGEdge(self.get_block(from_addr), self.get_block(to_addr), edge_type)
            edge.from_block.edges_out.add(edge)
            edge.to_block.edges_in.add(edge)
        
        # Delete the _temp_edges attributes for all blocks
        for block in self.blocks_dict.values():
            if hasattr(block, '_temp_edges_in'):
                del block._temp_edges_in
            if hasattr(block, '_temp_edges_out'):
                del block._temp_edges_out
    
    def __eq__(self, other: 'Any') -> 'bool':
        return isinstance(other, CFG) and all(eq_obj(self, other, selector=s) for s in ['normalizer', 'functions_dict', 'metadata'])
    
    def __hash__(self) -> 'int':
        return hash_obj([self.functions_dict, self.metadata, self.normalizer], return_int=True)

    def __str__(self) -> 'str':
        norm_str = 'no normalizer' if self.normalizer is None else ('normalizer: ' + repr(str(self.normalizer)))
        return "CFG with %s and %d functions, %d basic blocks, %d edges, and %d lines of assembly\nMetadata: %s" \
            % (norm_str, len(self.functions_dict), self.num_blocks, self.num_edges, self.num_asm_lines, self.metadata)

    def __repr__(self) -> 'str':
        return str(self)
    
    def get_cfg_build_code(self) -> 'str':
        """Returns python code that will build the given cfg. Used for testing.

        This will return the plain code itself to build, with no initial tabs.

        Args:
            cfg (CFG): the cfg
        
        Returns:
            str: string of python code to build the cfg
        """
        all_functions = "\n    ".join([("%d: CFGFunction(parent_cfg=__auto_cfg, address=%d, name=%s, is_extern_function=%s, metadata=%s)," % 
            (f.address, f.address, repr(f.name), f.is_extern_function, repr(f.metadata))) for f in self.functions])
        
        all_blocks = "\n    ".join([("%s: CFGBasicBlock(parent_function=__auto_functions[%d], address=%d, asm_memory_addresses=%s, metadata=%s, asm_lines=[\n        %s\n    ])," % 
                (b.address, b.parent_function.address, b.address, b.asm_memory_addresses, repr(b.metadata), '\n        '.join([repr(l) + ',' for l in b.asm_lines])
            )) for b in self.blocks])
        
        all_edges = "\n\n".join([("__auto_blocks[%d].edges_out = set([\n    %s\n])" % (b.address, 
            "\n    ".join([("CFGEdge(from_block=__auto_blocks[%d], to_block=__auto_blocks[%d], edge_type=EdgeType.%s)," % (edge.from_block.address, edge.to_block.address, edge.edge_type.name)) for edge in b.edges_out])
        )) for b in self.blocks])

        add_blocks = '\n\n'.join([("__auto_functions[%d].blocks = [\n    %s\n]" % (f.address,
            "\n    ".join([("__auto_blocks[%d]," % b.address) for b in f.blocks])
        )) for f in self.functions])

        return _CFG_BUILD_CODE_STR % (self.num_functions, self.num_blocks, self.num_edges, self.num_asm_lines, all_functions,
            all_blocks, all_edges, add_blocks)
    

# Dictionary mapping architectures to known matches that uniquely determine architecture (at least, for known supported architectures)
_DETECT_ARCH_START_DELIM = r'[^><"\']*(?<![a-z0-9])'
_DETECT_ARCH_END_DELIM = r'(?:[^a-z0-9].*|$)'
DETECT_ARCHITECTURE_RES = {
    # Basically, just any common keyword that isn't in Java for now
    Architectures.X86: [
        r'{start}(?:add|mov|test|xor){end}'.format(start=_DETECT_ARCH_START_DELIM, end=_DETECT_ARCH_END_DELIM),  
    ],

    # All of the 'invoke' commands for calls. Java should always invoke <init> at some point...
    Architectures.JAVA: [
        r'{start}invoke(?:virtual|interface|special|static|dynamic){end}'.format(start=_DETECT_ARCH_START_DELIM, end=_DETECT_ARCH_END_DELIM),  
    ],
}
DETECT_ARCHITECTURE_RES = {k: [re.compile(x) for x in v] for k, v in DETECT_ARCHITECTURE_RES.items()}


def auto_detect_assembly_language(cfg: 'CFG') -> 'None':
    """Attempts to detect the assembly language used in the given CFG, settings its 'architecture' key in the metadata if successful
    
    Will attempt to find known substrings in any block that indicate a specific language. Assumes the full CFG is all the
    same language

    Args:
        cfg (CFG): the cfg to detect language on
    """
    for block in cfg.blocks:
        for arch, matches in DETECT_ARCHITECTURE_RES.items():
            for match in matches:
                if any(match.fullmatch(l.lower()) for l in block.asm_lines):
                    cfg.metadata['architecture'] = arch.value[0]
                    return


class InvalidInsertionMemoryAddressError(Exception):
    pass


# NamedTuple used for conversion to networkx graph
_NetXTuple = namedtuple('CFGFunctionDataTuple', 'name is_extern_function blocks metadata')

        
_CFG_BUILD_CODE_STR: 'str' = """
##################
# AUTO-GENERATED #
##################

# Create the cfg object. This cfg has %d functions, %d basic blocks, %d edges, and %d lines of assembly.
__auto_cfg = CFG()

# Building all functions. Dictionary maps integer address to CFGFunction() object
__auto_functions = {
    %s
}

# Building basic blocks. Dictionary maps integer address to CFGBasicBlock() object
__auto_blocks = {
    %s
}

# Building all edges
%s

# Adding basic blocks to their associated functions
%s

# Adding functions to the cfg
__auto_cfg.add_function(*__auto_functions.values())

######################
# END AUTO-GENERATED #
######################
"""
