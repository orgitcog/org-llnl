import numpy as np
import pickle
import inspect
from enum import Enum
from ..normalization import normalize_cfg_data, Architectures
from .cfg import CFG, CFGBasicBlock, CFGFunction, EdgeType
from ..utils import get_smallest_np_dtype, scatter_nd_numpy, hash_obj, eq_obj, get_module, AtomicTokenDict
from ..utils.type_utils import *


if IN_PYTHON_TYPING_VERSION:
    BlockInfoBitDtype = np.int8  # Int type being used to store block info flags
    BlockInfoBitMaskFunctionType = Callable[[CFGBasicBlock], Literal[0, 1, True, False]]  # Dtype of functions that are used to compute bit flags


class MemCFG:
    """A CFG that is more memory/speed efficient.
    
    Keeps only the bare minimum information needed from a CFG. Stores edge connections in a CSR-like format.

    Parameters
    ----------
    cfg: `CFG`
        a CFG object. Can be a normalized or un-normalized. If un-normalized, then it will be normalized using the 
        `normalizer` parameter.
    normalizer: `Optional[Union[str, Normalizer]]`
        the normalizer to use to normalize the incoming CFG (or None if it is already normalized). If the incoming CFG 
        object has already been normalized, and `normalizer` is not None, then this will attempt to normalize the CFG 
        again with this normalizer
    keep_memory_addresses: `bool`
        if True, then memory addresses will also be kept. Otherwise they will be removed to save space
    inplace: `bool`
        if True and cfg needs to be normalized, it will be normalized inplace
    using_tokens: `Union[Dict[str, int], AtomicTokenDict]`
        if not None, then a dictionary mapping token strings to integer values. Any tokens in cfg but not in using_tokens
        will be added. Can also be an AtomicTokenDict for atomic updates to tokens
    force_renormalize: `bool`
        by default, this method will only normalize cfg's whose .normalizer != to the passed normalizer. However if 
        `force_renormalize=True`, then all cfg's will be renormalized even if they have been previously normalized with 
        the same normalizer.
    """

    normalizer: 'NormalizerType'
    """The normalizer used to normalize input before converting to ``MemCFG``
    
    Can be shared with a ``MemCFGDataset`` object if this ``MemCFG`` is a part of one
    """

    tokens: 'dict[str, int]'
    """Dictionary mapping token strings to integer values used in this ``MemCFG``
    
    Can be shared with a ``MemCFGDataset`` object if this ``MemCFG`` is a part of one.

    Can also be an AtomicTokenDict object for atomic token updates
    """

    function_name_to_idx: 'dict[str, int]'
    """Dictionary mapping string function names to their integer ids used in this ``MemCFG``"""

    asm_lines: 'np.ndarray'
    """Assembly line information
    
    A contiguous 1-d numpy array of shape (num_asm_lines,) of integer assembly line tokens. Dtype is the smallest 
    unsigned dtype needed to store the largest token value in this ``MemCFG``

    To get the assembly lines for some block index `block_idx`, you must get the assembly line indices from ``block_asm_idx``,
    and use those to slice the assembly lines:

    >>> block_idx = 7
    >>> memcfg.asm_lines[memcfg.block_asm_idx[block_idx]:memcfg.block_asm_idx[block_idx + 1]]

    Also see :func:`~bincfg.MemCFG.get_block_asm_lines`
    """

    asm_memory_addresses: 'Union[None, np.ndarray]'
    """Memory addresses for all of the assembly lines
    
    Only saved if `keep_memory_addresses=True` when constructing the ``MemCFG``. This will be a 1-d signed integer numpy
    array, where a value of -1 means the memory address for that corresponding line was not present in the basic block
    """

    block_asm_idx: 'np.ndarray'
    """Indices in ``asm_lines`` that correspond to the assembly lines for each basic block in this ``MemCFG``
    
    A 1-d numpy array of shape (num_blocks + 1,). Dtype is the smallest unsigned dtype needed to store the value 
    `num_asm_lines`. Assembly tokens for a block at index `i` would have a start index of `block_asm_idx[i]` and an end
    index of `block_asm_idx[i + 1]` in ``asm_lines``.
    """

    block_func_idx: 'np.ndarray'
    """Integer ids for the function that each basic block belongs to
    
    A 1-d numpy array of shape (num_blocks,) where each element is a function id for the block at that index. The id
    can be found in ``function_name_to_idx``. Dtype is the smallest unsigned dtype needed to store the value `num_functions`

    Also see :func:`~bincfg.MemCFG.get_block_function_idx` and :func:`~bincfg.MemCFG.get_block_function_name`
    """

    block_flags: 'np.ndarray'
    """Integer of bit flags for each basic block
    
    A 1-d numpy array of shape (num_blocks,) where each element is an integer of bit flags. See ``BlockInfoBitMask``
    for more info. Dtype is the smallest unsigned dtype with enough bits to store all flags in ``BlockInfoBitMask``

    Also see :func:`~bincfg.MemCFG.get_block_flags`
    """

    block_memory_addresses: 'Union[np.ndarray, None]'
    """Integer memory addresses of basic blocks.

    Only saved if `keep_memory_addresses=True` when constructing the ``MemCFG``. This will be a 1-d unsigned integer numpy
    array containing the memory addresses
    """

    block_asm_mem_addr_idx: 'Union[np.ndarray, None]'
    """Indices in ``block_memory_addresses`` that correspond to the assembly line memory addresses for basic blocks
    
    A 1-d numpy array of shape (num_blocks + 1,). Dtype is the smallest unsigned dtype needed to store the number of
    assembly line memory addresses. Memory addresses for a block at index `i` would have a start index of 
    `block_asm_mem_addr_idx[i]` and an end index of `block_asm_mem_addr_idx[i + 1]` in ``block_memory_addresses``.
    Only saved if `keep_memory_addresses=True` when constructing the ``MemCFG``.
    """

    metadata: 'dict'
    """Dictionary of metadata associated with this MemCFG"""

    function_metadata: 'list[Union[int, dict]]'
    """Metadata for functions
    
    A list of run length compressed metadata at the function level. We only compress metadata dictionaries that are empty.
    Elements are in the same order as the function indices in `block_func_idx`. Elements are either dictionaries (for the
    metadata of that current function), or integers indicating we should skip that many functions as they all have
    no metadata.
    """

    block_metadata: 'list[Union[int, dict]]'
    """Metadata for blocks
    
    A list of run length compressed metadata at the basic block level. We only compress metadata dictionaries that are empty.
    Elements are in the same order as the block indices in `block_asm_idx`. Elements are either dictionaries (for the
    metadata of that current block), or integers indicating we should skip that many blocks as they all have
    no metadata.
    """

    graph_c: 'np.ndarray'
    """Array containing all of the outgoing edges for each block in order
    
    1-D numpy array of shape (num_edges,). Dtype will be the smallest unsigned dtype required to store the value
    `num_blocks + 1`. Each element is a block index to which that edge connects. Edges will be in the order they appear in 
    each block's ``edges_out`` attribute, for each block in order of their block_idx. 

    Also see :func:`~bincfg.MemCFG.get_edges_out`

    NOTE: this also contains information on which types of edges they are inherently. If the block is NOT a function call
    (stored as bit flag in the block_info array), then all edges for that block are normal edges. If it IS a function
    call, then there are 3 cases:

        1. it has one outgoing edge: that edge is always a function call
        2. it has two outgoing edges, one function call, one normal: the first edge is the function call edge, the second
           is a normal edge
        3. it has >2 outgoing edges, or 2 function call edges: the edges will be listed first by function call edges, 
           then by normal edges, with a separator inbetween. The separator will have the max unsigned int value for 
           graph_c's dtype. This is why we use the dtype that can store `num_blocks + 1`, since we need this extra value
           just in case. Whatever exactly it means for a basic block to have >2 outgoing edges while being a function 
           call is left up to the user. Possibly due to call operators with non-explicit operands (eg: register memory 
           locations)?
    """

    graph_r: 'np.ndarray'
    """Array containing information on the number of outgoing edges for each block
    
    1-D numpy array of shape (num_edges + 1,). Dtype will be the smallest unsigned dtype required to store the value
    `num_edges`. This array is a cumulative sum of the number of edges for each basic block. One could get all of the 
    outgoing edges for a block using:

    >>> start_idx = memcfg.graph_r[block_idx]
    >>> end_idx = memcfg.graph_r[block_idx + 1]
    >>> edges = memcfg.graph_c[start_idx:end_idx]

    Also see :func:`~bincfg.MemCFG.get_edges_out`
    """

    class BlockInfoBitMask(Enum):
        """An Enum for block info bit masks
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        Each value is a tuple of the bit mask for that boolean, and a function to call with the block that returns a
        boolean True if that bit should be set, False otherwise. If True, then that bit will be '1' in that block's 
        block_flags int.
        """

        IS_FUNCTION_CALL: 'Tuple[int, BlockInfoBitMaskFunctionType]' = (1 << 0, lambda block: block.is_function_call)
        """Bit set if this block is a function call. See :py:func:`~bincfg.cfg_basic_block.is_function_call`"""
        IS_FUNCTION_ENTRY: 'Tuple[int, BlockInfoBitMaskFunctionType]' = (1 << 1, lambda block: block.is_function_entry)
        """Bit set if this block is a function entry. See :py:func:`~bincfg.cfg_basic_block.is_function_entry`"""
        IS_IN_EXTERN_FUNCTION: 'Tuple[int, BlockInfoBitMaskFunctionType]' = (1 << 2, lambda block: block.parent_function.is_extern_function)
        """Bit set if this block is within an external function. See :py:func:`~bincfg.cfg_function.is_extern_function`"""
        IS_FUNCTION_JUMP: 'Tuple[int, BlockInfoBitMaskFunctionType]' = (1 << 3, lambda block: block.is_function_jump)
        """Bit set if this block is a function jump. IE: this block has a jump instruction that resolves to a basic block
        in a separate function. See :py:func:`~bincfg.cfg_basic_block.is_function_jump`"""
        IS_MULTI_FUNCTION_CALL: 'Tuple[int, BlockInfoBitMaskFunctionType]' = (1 << 4, lambda block: 0)
        """Bit set if this block is a multi-function call. IE: this block has either two or more function call edges out,
        or one function call and two or more normal edges out. See :py:func:`~bincfg.cfg_basic_block.is_multi_function_call`
        
        Currently not setting the block here in _block_flags_int(), but instead in MemCFG initialization in order to save
        time (we don't have to compute get_sorted_edges() multiple times)
        """

    @staticmethod
    def _block_flags_int(block: 'CFGBasicBlock') -> 'int':
        """Gets all of the block information and stores it as a integer of bit-set flags

        Args:
            block (CFGBasicBlock): the CFGBasicBlock to get the info int from

        Returns:
            int: the integer of bit-set flags
        """
        block_flags = 0
        for bm in MemCFG.BlockInfoBitMask:
            block_flags |= bm.value[0] * bm.value[1](block)  # Times will convert boolean to 1 or 0
        return block_flags

    def __init__(self, cfg: 'CFG', normalizer: 'Optional[Union[str, NormalizerType]]' = None, keep_memory_addresses: 'bool' = False,
                 inplace: 'bool' = False, using_tokens: 'Optional[Union[dict, AtomicTokenDict]]' = None, force_renormalize: 'bool' = False):
        # Make sure input is a cfg
        if not isinstance(cfg, CFG):
            raise TypeError("Can only build a MemCFG out of a CFG object, not '%s'" % type(cfg).__name__)

        # Normalize the CFG if needed
        if normalizer is not None or cfg.normalizer is None:
            # Make sure there is some normalizer to use
            if normalizer is None:
                raise ValueError("Must pass a normalizer if `cfg` is unnormalized!")
            cfg = normalize_cfg_data(cfg, normalizer=normalizer, inplace=inplace, force_renormalize=force_renormalize)
        
        # Keep cfg's normalization if possible
        self.normalizer = cfg.normalizer

        # Figure out what the tokens should be, updating the token dict if we find new ones
        self.tokens = {} if using_tokens is None else using_tokens

        if isinstance(self.tokens, AtomicTokenDict):
            self.tokens.addtokens(*{l for block in cfg.blocks for l in block.asm_lines})
        else:
            for block in cfg.blocks:
                for l in block.asm_lines:
                    self.tokens.setdefault(l, len(self.tokens))
                    
        max_token = len(self.tokens)
        max_asm_addr = default_max((a for b in cfg.blocks for a in b.asm_memory_addresses), 0)
        max_block_addr = default_max((b.address for b in cfg.blocks), 0)

        # Make mappings from function names to indices. Make sure there aren't duplicates (most likely only going to
        #   occur in functions with no name)
        self.function_name_to_idx = {}
        function_addr_to_idx = {}
        self.function_metadata = []
        for i, f in enumerate(cfg.functions):
            if f.name in self.function_name_to_idx:
                func_name = f.name
                fn_idx = 0
                while func_name in self.function_name_to_idx:
                    func_name = f.name + '_%d' % fn_idx
                    fn_idx += 1
            else:
                func_name = f.name
            self.function_name_to_idx[func_name] = i
            function_addr_to_idx[f.address] = i
            self.function_metadata.append(f.metadata)

        # Make the data arrays
        self.asm_lines = np.empty([cfg.num_asm_lines], dtype=get_smallest_np_dtype(max_token))
        self.block_asm_idx = np.empty([cfg.num_blocks + 1], dtype=get_smallest_np_dtype(cfg.num_asm_lines))
        self.block_func_idx = np.empty([cfg.num_blocks], dtype=get_smallest_np_dtype(len(cfg.functions_dict)))
        self.block_flags = np.empty([cfg.num_blocks], dtype=get_smallest_np_dtype(len(MemCFG.BlockInfoBitMask)))
        graph_c = []
        self.graph_r = np.empty([cfg.num_blocks + 1])
        self.block_labels = {}  # Mapping of block indices to block labels
        self.block_metadata = []

        # Set the initial graph_r start and final block_asm_idx
        self.graph_r[0] = 0
        self.block_asm_idx[-1] = len(self.asm_lines)

        if keep_memory_addresses:
            self.block_memory_addresses = np.empty([cfg.num_blocks], dtype=get_smallest_np_dtype(max_block_addr, signed=True))
            self.asm_memory_addresses = np.empty(sum(len(b.asm_memory_addresses) for b in cfg.blocks), dtype=get_smallest_np_dtype(max_asm_addr, signed=True))
            self.block_asm_mem_addr_idx = np.empty([cfg.num_blocks + 1], dtype=get_smallest_np_dtype(cfg.num_asm_lines))
            self.block_asm_mem_addr_idx[-1] = len(self.asm_memory_addresses)
        else:
            self.asm_memory_addresses, self.block_asm_mem_addr_idx, self.block_memory_addresses = None, None, None

        # Copy the metadata from the cfg
        self.metadata = cfg.metadata.copy()

        # Create temporary mapping of CFGBasicBlock.address to integer index of block (sorted by address for determinism)
        block_addr_to_idx = {block.address: i for i, block in enumerate(cfg.blocks)}

        # Convert the data at each block to better memory one
        asm_line_idx, asm_mem_idx = 0, 0
        for block_idx, block in enumerate(cfg.blocks):

            # Get the assembly lines and memory addresses, and store the length of the assembly lines
            asm_line_end = asm_line_idx + block.num_asm_lines
            self.asm_lines[asm_line_idx: asm_line_end] = [self.tokens[l] for l in block.asm_lines]
            self.block_asm_idx[block_idx] = asm_line_idx
            asm_line_idx = asm_line_end

            # Add in the memory addresses if using
            if self.asm_memory_addresses is not None:
                self.block_memory_addresses[block_idx] = block.address
                asm_mem_end = asm_mem_idx + len(block.asm_memory_addresses)
                self.asm_memory_addresses[asm_mem_idx: asm_mem_end] = block.asm_memory_addresses
                self.block_asm_mem_addr_idx[block_idx] = asm_mem_idx
                asm_mem_idx = asm_mem_end

            # Get the block's function name
            self.block_func_idx[block_idx] = function_addr_to_idx[block.parent_function.address]

            # Get all of the edges associated with this block in order: normal edges, function call edges
            edge_lists = block.get_sorted_edges(direction='out', as_sets=False)

            # Add in the function call edges
            graph_c += [block_addr_to_idx[edge.to_block.address] for edge in edge_lists[1]]

            # Check for either >1 function call edges, or >2 edges while being a function call.
            # We add a -1, which will be rolled over to the unsigned int max value for graph_c array
            added_minus_1 = 0
            if len(edge_lists[1]) >= 2 or (len(edge_lists[1]) == 1 and len(edge_lists[0]) > 1):
                graph_c.append(-1)
                added_minus_1 = 1
            
            # Add in the normal edges
            graph_c += [block_addr_to_idx[edge.to_block.address] for edge in edge_lists[0]]

            # Update the new graph_r, taking into account whether or not we inserted an extra -1 to split function call/normal edges
            self.graph_r[block_idx + 1] = sum(len(l) for l in edge_lists) + added_minus_1
            
            # Get the block information flags for this block.
            # Set the flag for is_multi_function_call here so we don't have to call get_sorted_edges more than once
            self.block_flags[block_idx] = MemCFG._block_flags_int(block) | \
                (MemCFG.BlockInfoBitMask.IS_MULTI_FUNCTION_CALL.value[0] * added_minus_1)
            
            # Get the block metadata and memory address
            self.block_metadata.append(block.metadata)
        
        # Conver graph_c and graph_r to numpy arrays
        graphc_dtype = get_smallest_np_dtype(len(cfg.blocks_dict) + 1, signed=False)
        self.max_graph_c_int = np.iinfo(graphc_dtype).max
        self.graph_c = np.array([(v if v != -1 else self.max_graph_c_int) for v in graph_c], dtype=graphc_dtype)
        self.graph_r = np.cumsum(self.graph_r, dtype=get_smallest_np_dtype(len(self.graph_c)))
        
        # Run length encode the metadata
        self.function_metadata = _rl_encode_metadata(self.function_metadata)
        self.block_metadata = _rl_encode_metadata(self.block_metadata)
    
    def get_block_info(self, block_idx):
        """Returns all the info associated with the given block index as a dictionary

        Args:
            block_idx (int): integer block index

        Returns:
            dict: the block info dictionary with keys/values:

                * 'asm_lines' (np.ndarray): 1-d numpy array of unsigned integer assembly line tokens in this block
                * 'asm_memory_addresses' (np.ndarray): 1-d numpy array of signed integer memory addresses
                  for the assembly lines in this block. Values will be -1 if the memory addresses do not exist
                * 'edges_out' (np.ndarray): 1-d numpy array of unsigned integer block indices for all
                  of the edges out from this block
                * 'edge_types' (np.ndarray): 1-d numpy array of uint8 values for the edge types associated
                  with all of the edges out. These are the values of objects in the EdgeType enum. Currently: EdgeType.NORMAL == 1,
                  EdgeType.FUNCTION_CALL == 2
                * 'function_index' (int): the integer function index of the function this block resides in
                * 'is_function_call' (bool): true if this block is a function call block (has at least one outgoing function call edge)
                * 'is_function_entry' (bool): true if this block is a function entry block (has the same memory address
                  as its parent function)
                * 'is_extern_function' (bool): true if this block is within an external function (parent_function.is_extern_function is True)
                * 'is_function_jump' (bool): true if this block is a function jump block (has a 'normal' edge to a block
                  that is within another function)
                * 'is_multi_function_call' (bool): true if this block is a multi-function call block (has 2 or more outgoing
                  function call edges. IE: a call table)
                * 'metadata' (dict): dictionary of metadata associated with this block
        """
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')

        ret = (self.get_block_asm_lines(block_idx),) + (self.get_block_asm_memory_addresses(block_idx),) + self.get_block_edges_out(block_idx, ret_edge_types=True) \
            + (self.get_block_function_idx(block_idx),) + self.get_block_flags(block_idx) + (self.get_block_metadata(block_idx),)
        
        ret = {k: v for k, v in zip(['asm_lines', 'asm_memory_addresses', 'edges_out', 'edge_types', 'function_index', 'is_function_call', 
                'is_function_entry', 'is_extern_function', 'is_function_jump', 'is_multi_function_call', 'metadata'], ret)}
        
        return ret

    def get_block_asm_lines(self, block_idx: 'int') -> 'np.ndarray':
        """Get the asm lines associated with this block index

        Args:
            block_idx (int): integer block index

        Returns:
            np.ndarray: a 1-d numpy array of unsigned integer assembly tokens
        """
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')
        return self.asm_lines[self.block_asm_idx[block_idx]:self.block_asm_idx[block_idx+1]]

    def get_block_asm_memory_addresses(self, block_idx: 'int') -> 'np.ndarray':
        """Get the asm memory addresses associated with this block index

        Values are -1 if the memory address did not exist in that block

        Args:
            block_idx (int): integer block index

        Returns:
            np.ndarray: a 1-d numpy array of signed integer assembly tokens
        """
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')
        return self.asm_memory_addresses[self.block_asm_mem_addr_idx[block_idx]:self.block_asm_mem_addr_idx[block_idx+1]]
    
    def get_block_edges_out(self, block_idx: 'int', ret_edge_types: 'bool' = False) -> \
        'Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]':
        """Get numpy array of block indices for all edges out associated with the given block index

        Args:
            block_idx (int): integer block index
            ret_edge_types (bool): if True, will also return a numpy array (1-d, dtype np.uint8) containing the edge
                type values for each edge with values:

                    - 1: normal edge
                    - 2: function call edge

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray: 
                either a 1-d numpy array of unsigned integer block indices for all edges out associated with the given 
                block index, or if `ret_edge_types=True`, then a tuple of (block_edge_inds, edge_types) where the `edge_types`
                is a 1-d numpy array of uint8 edge types with the same shape as block_edge_inds that designates the types
                of the edges. Edge types will be the values of those in the `EdgeType` enum.
        """
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')

        # Get all of the edges
        ret = self.graph_c[self.graph_r[block_idx]:self.graph_r[block_idx+1]]

        # Check if we are returning the edge types as well
        if ret_edge_types:

            # Check if this block is a function call
            if self.is_block_function_call(block_idx):

                # Check to see if this block is a multi-function call, in which case we have to split up the array
                if self.is_block_multi_function_call(block_idx):
                    
                    # Get the split index (index in ret that is equal to unsigned dtype max). It should have split_idx
                    #   function call edges, and len(ret) - split_idx - 1 normal edges (to account for the fact that
                    #   the split_idx itself is also stored in the array)
                    split_idx = np.argwhere(ret == self.max_graph_c_int)[0][0]
                    return ret[ret != self.max_graph_c_int], \
                        np.array([EdgeType.FUNCTION_CALL.value] * split_idx + [EdgeType.NORMAL.value] * (len(ret) - split_idx - 1))
                
                # Otherwise it is not a multi-function call. We can simply return ret, and edge types are either 
                #   [function_call] or [function_call, normal]
                else:
                    return ret, np.array(([EdgeType.FUNCTION_CALL.value] if len(ret) == 1 else \
                        [EdgeType.FUNCTION_CALL.value, EdgeType.NORMAL.value]), dtype=np.uint8)
            
            # It's not a function call, so we can just return ret and all edge types must be normal edges
            else:
                return ret, np.full(ret.shape, EdgeType.NORMAL.value, dtype=np.uint8)

        # Otherwise we are not returning the edge types, just return all values in ret that are not the splitting value
        # We only need to remove the splitting value if there are more than two edges, AND it is a function call. Otherwise
        #   we are doing a useless numpy comparison and memory-grabbing every call which takes a non-neglible amount of time
        return ret[ret != self.max_graph_c_int]

    def get_block_function_idx(self, block_idx: 'int') -> 'int':
        """Get the function index for the given block index

        Args:
            block_idx (int): integer block index

        Returns:
            int: the integer function index for the given block index
        """
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')
        return self.block_func_idx[block_idx]
    
    def get_block_function_name(self, block_idx: 'int') -> 'str':
        """Get the function name for the given block index
        
        Functions without names will start with '__unnamed_func__'

        Args:
            block_idx (int): integer block index

        Returns:
            str: the function name for the given block index
        """
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')

        func_idx = self.get_block_function_idx(block_idx)

        # Make an inverse mapping now that we know we are calling this function
        if not hasattr(self, 'function_idx_to_name'):
            self.function_idx_to_name = {v: k for k, v in self.function_name_to_idx.items()}

        return self.function_idx_to_name[func_idx]
    
    def get_block_memory_address(self, block_idx: 'int') -> 'int':
        """Returns the memory address for the given block, if present, -1 if not present
        
        Args:
            block_idx (int): integer block index
        
        Returns:
            int: the memory address
        """
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')
        return -1 if self.block_memory_addresses is None else self.block_memory_addresses[block_idx]
    
    def get_block_flags(self, block_idx: 'int') -> 'Tuple[bool, bool, bool, bool, bool, bool]':
        """Get all block flags for the given block index

        Args:
            block_idx (int): integer block index

        Returns:
            Tuple[bool, bool, bool, bool, bool, bool]: (is_block_function_call, is_block_function_entry, 
                is_block_extern_function, is_block_function_jump, is_block_multi_function_call)
        """
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')

        return self.is_block_function_call(block_idx), self.is_block_function_entry(block_idx), \
            self.is_block_extern_function(block_idx), self.is_block_function_jump(block_idx), \
            self.is_block_multi_function_call(block_idx)
    
    def get_function_block_inds(self, func_idx: 'int') -> 'list[int]':
        """Returns all of the block indices that are within the given function
        
        Args:
            func_idx (int): the integer function index
        
        Returns:
            list[int]: list of integer block indices that are within the given function
        """
        assert_valid_idx(func_idx, self.num_functions, 'functions')

        if func_idx not in self.function_name_to_idx.values():
            raise ValueError("Unknown function index: %d" % func_idx)
        return [i for i in range(self.num_blocks) if self.block_func_idx[i] == func_idx]

    def get_function_metadata(self, func_idx: 'Union[int, None]') -> 'Union[dict, list[dict]]':
        """Returns the metadata associated with that function index
        
        Args:
            func_idx (Union[int, None]): the integer function index of the metadata to get, or None to get the full
                list of metadata
        
        Returns:
            Union[dict, list[dict]]: dictionary of metadata associated with the given function index
        """
        if isinstance(func_idx, INTEGER_TYPES):
            assert_valid_idx(func_idx, self.num_functions, 'functions')
        return _decode_rl_metadata(self.function_metadata, func_idx)

    def get_block_metadata(self, block_idx: 'Union[int, None]') -> 'Union[dict, list[dict]]':
        """Returns the metadata associated with that function index
        
        Args:
            block_idx (Union[int, None]): the integer block index of the metadata to get, or None to get the full
                list of metadata
        
        Returns:
            Union[dict, list[dict]]: dictionary of metadata associated with the given block index
        """
        if isinstance(block_idx, INTEGER_TYPES):
            assert_valid_idx(block_idx, self.num_blocks, 'blocks')
        return _decode_rl_metadata(self.block_metadata, block_idx)
    
    def is_block_function_call(self, block_idx: 'int') -> 'bool':
        """True if this block is a function call, False otherwise"""
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')
        return (self.block_flags[block_idx] & MemCFG.BlockInfoBitMask.IS_FUNCTION_CALL.value[0]) > 0
    
    def is_block_function_entry(self, block_idx: 'int') -> 'bool':
        """True if this block is a function entry, False otherwise"""
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')
        return (self.block_flags[block_idx] & MemCFG.BlockInfoBitMask.IS_FUNCTION_ENTRY.value[0]) > 0
    
    def is_block_extern_function(self, block_idx: 'int') -> 'bool':
        """True if this block is in an external function, False otherwise"""
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')
        return (self.block_flags[block_idx] & MemCFG.BlockInfoBitMask.IS_IN_EXTERN_FUNCTION.value[0]) > 0
    
    def is_block_function_jump(self, block_idx: 'int') -> 'bool':
        """True if this block is a function jump, False otherwise"""
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')
        return (self.block_flags[block_idx] & MemCFG.BlockInfoBitMask.IS_FUNCTION_JUMP.value[0]) > 0
    
    def is_block_multi_function_call(self, block_idx: 'int') -> 'bool':
        """True if this block is a multi-function call, False otherwise"""
        assert_valid_idx(block_idx, self.num_blocks, 'blocks')
        return (self.block_flags[block_idx] & MemCFG.BlockInfoBitMask.IS_MULTI_FUNCTION_CALL.value[0]) > 0
    
    @property
    def num_blocks(self) -> 'int':
        """The number of blocks in this ``MemCFG``"""
        return len(self.block_func_idx)
    
    @property
    def num_edges(self) -> 'int':
        """The number of edges in this ``MemCFG``"""
        return len(self.graph_c[self.graph_c != self.max_graph_c_int])
    
    @property
    def num_asm_lines(self) -> 'int':
        """The number of assembly lines in this ``MemCFG``"""
        return len(self.asm_lines)
    
    @property
    def num_functions(self) -> 'int':
        """The number of functions in this ``MemCFG``"""
        return len(self.function_name_to_idx)
    
    @property
    def inv_tokens(self) -> 'dict[int, str]':
        """Returns the inverse of `self.tokens`: dictionary mapping token integers to their original strings"""
        return {v:k for k, v in self.tokens.items()}
    
    def update_metadata(self, other: 'dict') -> 'Self':
        """Updates this MemCFG's metadata dictionary with the given dictionary, and returns self"""
        self.metadata.update(other)
        return self
    
    def set_tokens(self, tokens: 'Union[dict, AtomicTokenDict]') -> 'Self':
        """Sets this MemCFG's tokens to the given tokens, and returns self"""
        self.tokens = tokens
        return self
    
    def normalize(self, normalizer: 'Optional[Union[str, NormalizerType]]' = None, using_tokens: 'Union[dict, AtomicTokenDict]' = None, 
                  inplace: 'bool' = True, force_renormalize: 'bool' = False) -> 'MemCFG':
        """Normalizes this memcfg in-place.

        Args:
            normalizer (Optional[Union[str, NormalizerType]]): the normalizer to use. Can be a ``Normalizer`` object, or a 
                string, or None to use the default BaseNormalizer(). Defaults to None.
            using_tokens (Union[dict, AtomicTokenDict]): tokens to use when normalizing
            inplace (bool): whether or not to normalize inplace. Defaults to True.
            force_renormalize (bool): by default, this method will only normalize this cfg if the passed 
                `normalizer` is != `self.normalizer`. However if `force_renormalize=True`, then this will be renormalized
                even if it has been previously normalized with the same normalizer. Defaults to False.

        Returns:
            MemCFG: this ``MemCFG`` normalized
        """
        return normalize_cfg_data(self, normalizer=normalizer, using_tokens=using_tokens, inplace=inplace, 
                                  force_renormalize=force_renormalize)
    
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
            raise ValueError("Could not find 'arch' or 'architecture' key in metadata")
        
        if arch in ['x86']:
            return Architectures.X86
        elif arch in ['java', 'java-bytecode']:
            return Architectures.JAVA
        else:
            raise ValueError("Unknown architecture value: %s" % repr(arch))
    
    def get_edge_values(self) -> 'np.ndarray':
        """Returns the edge type values
        
        Returns a 1-d numpy array of length self.num_edges and dtype np.int32 containing an integer type for each
        edge depending on if it is a normal or function call edge. Edges are directed and have values from `EdgeType` enum.
        Values:

            - 1: 'normal' edges
            - 2: 'function call' edges
        
        NOTE: this returns as type np.int32 since pytorch can be finicky about what dtypes it wants

        Returns:
            np.ndarray: a 1-d numpy array of length self.num_edges and dtype np.int32 containing integer edge types
        """
        return np.array([v for b in range(self.num_blocks) for v in self.get_block_edges_out(b, ret_edge_types=True)[1]], dtype=np.int32)
    
    def get_coo_indices(self) -> 'np.ndarray':
        """Returns the COO indices for this MemCFG

        Returns a 2-d numpy array of shape (num_edges, 2) of dtype np.int32. Each row is an edge, column 0 is the 'row' 
        indexer, and column 1 is the 'column' indexer. EG:
        
        .. highlight:: python
        .. code-block:: python
        
            original = np.array([
                [0, 1],
                [1, 1]
            ])

            coo_indices = np.array([
                [0, 1],
                [1, 0],
                [1, 1]
            ])
        
        NOTE: this returns as type np.int32 since pytorch can be finicky about what dtypes it wants
        NOTE: pytorch sparse_coo_tensor's indicies are the transpose of the array this method returns

        Returns:
            np.ndarray: a 2-d numpy array of shape (num_edges, 2) of dtype np.int32 containing COO indices
        """
        inds = np.empty([self.num_edges, 2], dtype=np.int32)
        start = 0
        for bi in range(self.num_blocks):
            edges_out = self.get_block_edges_out(bi, ret_edge_types=False)
            end = start + len(edges_out)
            inds[start:end, 0] = bi
            inds[start:end, 1] = edges_out
            start = end
        return inds
    
    def to_adjacency_matrix(self, type: 'Literal["np", "numpy", "torch"]' = 'np', sparse: 'bool' = False) ->\
        'Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]':
        """Returns an adjacency matrix representation of this memcfg's graph connections

        Edges are directed and have values from `EdgeType` enum. Values:

            - 1: 'normal' edges
            - 2: 'function call' edges

        Args:
            type (Literal["np", "numpy", "torch"]): the type of matrix to return. Defaults to 'np'. Can be:

                - 'np'/'numpy' for a numpy ndarray (dtype: np.int32)
                - 'torch'/'pytorch' for a pytorch tensor (type: LongTensor)
            
            sparse (bool): whether or not the return value should be a sparse matrix. Defaults to False. Has 
                different behaviors based on type:

                - numpy array: returns a 2-tuple of sparse COO representation (indices, values).
                    NOTE: the indices are the transpose of those from `get_coo_indices()`
                    NOTE: if you want sparse CSR format, you already have it with self.graph_c and self.graph_r
                - pytorch tensor: returns a pytorch sparse COO tensor. 
                    NOTE: not using sparse CSR format for now since it seems to have less documentation/supportedness. 

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray: 
                an adjacency matrix representation of this ``MemCFG``
        """
        type = type.lower()

        # Return adj mat as intended type
        if type in ['np', 'numpy']:
            if sparse:
                return (self.get_coo_indices().T, self.get_edge_values())
            else:
                ret = np.zeros((self.num_blocks, self.num_blocks))
                return scatter_nd_numpy(ret, self.get_coo_indices(), self.get_edge_values())

        elif type in ['torch', 'pytorch']:
            torch = get_module('torch', err_message="Cannot find module `torch` required to return pytorch tensors!")
            sparse_coo = torch.sparse_coo_tensor(indices=self.get_coo_indices().T, values=self.get_edge_values(), size=(self.num_blocks, self.num_blocks)).coalesce()
            return sparse_coo if sparse else sparse_coo.to_dense()

        else:
            raise ValueError("Unknown adjacency matrix type: '%s'" % type)
    
    def to_cfg(self) -> 'CFG':
        """Converts this MemCFG back into a CFG
        
        NOTE: if `keep_memory_addresses=False` when constucting this MemCFG, then memory addresses will not be present
        and basic blocks will be given a memory address that is just their index in the block list
        """
        cfg = CFG(metadata=self.metadata, normalizer=self.normalizer)

        # Build the blocks
        edge_types = [None] * (max(et.value for et in EdgeType) + 1)
        for et in EdgeType:
            edge_types[et.value] = et.name.lower()

        block_metadata = self.get_block_metadata(None)
        block_addrs = self.block_memory_addresses if self.block_memory_addresses is not None else np.arange(self.num_blocks)

        blocks = [CFGBasicBlock(
            address = block_addrs[block_idx],
            edges_out = [(None, block_addrs[addr], edge_types[et]) for addr, et in zip(*self.get_block_edges_out(block_idx, ret_edge_types=True))], 
            asm_lines = [self.inv_tokens[t] for t in self.asm_lines[self.block_asm_idx[block_idx]:self.block_asm_idx[block_idx+1]]],
            asm_memory_addresses = self.asm_memory_addresses[self.block_asm_mem_addr_idx[block_idx]:self.block_asm_mem_addr_idx[block_idx+1]]\
                if self.asm_memory_addresses is not None else None,
            metadata = block_metadata[block_idx],
        ) for block_idx in range(self.num_blocks)]

        # Build the functions
        func_metadata = self.get_function_metadata(None)
        funcs = [CFGFunction(metadata=func_metadata[func_idx]) for func_idx in range(self.num_functions)]
        for block_idx in range(self.num_blocks):
            func_idx = self.get_block_function_idx(block_idx)

            funcs[func_idx].blocks.append(blocks[block_idx])
            funcs[func_idx].name = self.get_block_function_name(block_idx)
            if self.is_block_extern_function(block_idx):
                funcs[func_idx]._is_extern_function = True
            if self.is_block_function_entry(block_idx):
                funcs[func_idx].address = blocks[block_idx].address
        
        # Add the functions to the cfg and return
        cfg.add_function(*funcs)
        return cfg
    
    def save(self, path: 'str') -> 'None':
        """Saves this MemCFG to the given path"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    def dumps(self) -> 'str':
        """Returns this object pickled with pickle.dumps()"""
        return pickle.dumps(self)
    
    def drop_tokens(self) -> 'Self':
        """Sets the tokens in this normalizer to None. Make sure you only do this if tokens are saved elsewhere! Returns self"""
        self.tokens = None
        return self
    
    @classmethod
    def load(cls, path: 'str') -> 'MemCFG':
        """Loads a MemCFG from the given path"""
        with open(path, 'rb') as f:
            return pickle.load(f)
        
    def __str__(self) -> 'str':
        return "MemCFG with normalizer: %s and %d functions, %d blocks, %d assembly lines, and %d edges. Metadata:\n%s" % \
            (repr(str(self.normalizer)), self.num_functions, self.num_blocks, self.num_asm_lines, self.num_edges, self.metadata)
    
    def __repr__(self) -> 'str':
        return self.__str__()

    def __getstate__(self) -> 'dict':
        """State for pickling"""
        ret = self.__dict__.copy()
        if 'function_idx_to_name' in ret:
            del ret['function_idx_to_name']
        return ret
    
    def __setstate__(self, state: 'dict') -> 'None':
        """Set state for pickling"""
        for k, v in state.items():
            setattr(self, k, v)

        if not hasattr(self, 'max_graph_c_int'):
            self.max_graph_c_int = np.iinfo(self.graph_c.dtype).max
    
    def __eq__(self, other) -> 'bool':
        return isinstance(other, MemCFG) and all(eq_obj(self, other, selector=s) for s in [
            'normalizer', 'tokens', 'function_name_to_idx', 'asm_lines', 'asm_memory_addresses', 'block_asm_idx', 'block_func_idx', 
            'block_flags',  'metadata', 'function_metadata', 'block_metadata', 'graph_c', 'graph_r', 'block_labels'
        ])
    
    def __hash__(self) -> 'hash':
        return hash_obj([
            self.normalizer, self.tokens, self.function_name_to_idx, self.asm_lines, self.asm_memory_addresses, 
            self.block_asm_idx, self.block_func_idx, self.block_flags, self.metadata, self.function_metadata, 
            self.block_metadata, self.graph_c, self.graph_r, self.block_labels,
        ], return_int=True)
    

def _rl_encode_metadata(meta_list: 'list[dict]') -> 'list[Union[int, dict]]':
    """Run length encodes metadata. Only encodes empty dictionaries for now"""
    ret = []
    for d in meta_list:
        if d is None or len(d) == 0:
            # If we need to insert a new integer for rl encoding
            if len(ret) == 0 or isinstance(ret[-1], dict):
                ret.append(1)
            else:
                ret[-1] += 1
        else:
            ret.append(d)
    return ret


def _decode_rl_metadata(meta_list: 'list[Union[int, dict]]', idx: 'Optional[int]' = None):
    """Decodes run length encoded metadata lists
    
    Assumes 0 < `idx` < number of functions

    Args:
        meta_list (list[Union[int, dict]]): the metadata compressed list
        idx (Optional[int]): if None, then this will return the full uncompressed list. Otherwise this will return the
            element at this index
    """
    # If we are just returning the full list of metadata
    if idx is None:
        ret = []
        for e in meta_list:
            if isinstance(e, int):
                ret += [{} for _ in range(e)]
            else:
                ret.append(e)
        return ret

    assert 0 <= idx < len(meta_list) + sum((e - 1) for e in meta_list if isinstance(e, int))

    # Otherwise, we can stop early if possible
    curr_idx = 0
    ml_idx = 0
    while True:
        if curr_idx == idx:
            return {} if isinstance(meta_list[ml_idx], int) else meta_list[ml_idx]
        elif isinstance(meta_list[ml_idx], int):
            # If our index is in this range, return empty dict
            if curr_idx <= idx < curr_idx + meta_list[ml_idx]:
                return {}
            curr_idx += meta_list[ml_idx]
        else:
            curr_idx += 1
        ml_idx += 1

    # We should not be able to reach here
    assert False


def assert_valid_idx(idx: 'int', max_val: 'int', objects_str: 'str') -> 'None':
    """Asserts that the idx passed is >= 0 and < max_val. `objects_str` is the type of object for error message (IE: 'blocks', 'functions', etc.)"""
    func_name = repr(inspect.getframeinfo(inspect.currentframe().f_back)[2])
    if idx < 0:
        raise ValueError("Index passed to function %s must be non-negative, got: %d" % (func_name, idx))
    elif idx >= max_val:
        raise ValueError("Index passed to function %s must be < the maximum number of %s (%d), got: %d" % (func_name, objects_str, max_val, idx))


def default_max(iterable, default=None):
    """Returns the max value in `iterable`, or `default` value if len(iterable) == 0"""
    try:
        return max(iterable)
    except ValueError:
        return default
    