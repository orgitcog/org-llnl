import warnings
import bincfg
from collections import Counter
from .cfg_basic_block import CFGBasicBlock
from ..utils import eq_obj, hash_obj, get_address
from ..utils.type_utils import *


if IN_PYTHON_TYPING_VERSION:
    CFGFunctionPickledState = Tuple[int, str, Tuple[CFGBasicBlock, ...], bool]
    """The pickled state of a function"""


class CFGFunction:
    """A single function in a ``CFG``

    Can be initialized empty, or by passing kwarg values.

    NOTE: these objects should not be pickled/copy.deepcopy()-ed by themselves, only as a part of a cfg

    Parameters
    ----------
    parent_cfg: `Optional[bincfg.CFG]`
        the parent ``CFG`` object to which this ``CFGFunction`` belongs
    address: `Optional[AddressLike]`
        the memory address of this function. If not present, then the address will be set to -1
    name: `Optional[str]`
        the string name of this function. If not present, or if the name passed is the empty string, this function is 
        given a default name '__UNNAMMED_FUNC_X' where 'X' is the memory address
    blocks: `Optional[Iterable[CFGBasicBlock]]`
        if None, will be initialized to an empty list, otherwise an iterable of ``CFGBasicBlock`` objects that are within 
        this function
    is_extern_function: `bool`
        if True, then this function is an external function (a dynamically loaded function)
    metadata: `Optional[dict]`
        optional dictionary of metadata to associate with this function
    """

    parent_cfg: 'Union[bincfg.CFG, None]'
    """the parent ``CFG`` object to which this ``CFGFunction`` belongs, or None if it hasn't been initialized yet"""

    address: 'int'
    """the integer memory address of this function. Will be -1 if not initialized yet"""

    name: 'str'
    """the string name of this function. Will be given a default name based on its memory address if not present"""

    blocks: 'list[CFGBasicBlock]'
    """list of all basic blocks in this function"""

    metadata: 'dict'
    """Dictionary of metadata associated with this function"""

    def __init__(self, parent_cfg: 'Optional[bincfg.CFG]' = None, address: 'Optional[AddressLike]' = None, name: 'Optional[str]' = None, 
                 blocks: 'Optional[Iterable[CFGBasicBlock]]' = None, is_extern_function: bool = False, metadata: 'Optional[dict]' = None):
        self.parent_cfg = parent_cfg
        self.address = -1 if address is None else get_address(address)
        self.name = name if name is not None and name != '' else ("__UNNAMED_FUNC_%d" % self.address)
        self.blocks = [] if blocks is None else list(blocks)
        self.metadata = {} if metadata is None else metadata
        self._is_extern_function: bool = is_extern_function
    
    @property
    def num_blocks(self) -> 'int':
        """The number of basic blocks in this function"""
        return len(self.blocks)
    
    @property
    def num_asm_lines(self) -> 'int':
        """The total number of assembly lines across all blocks in this function"""
        return sum(b.num_asm_lines for b in self.blocks)
    
    @property
    def asm_counts(self) -> 'Mapping[str, int]':
        """A ``collections.Counter`` of all unique assembly lines and their counts in this function"""
        return sum((b.asm_counts for b in self.blocks), Counter())
    
    @property
    def is_root_function(self) -> 'bool':
        """True if this function is not called by any other functions, False otherwise"""
        return len(self.called_by) == 0
    
    @property
    def is_recursive(self) -> 'bool':
        """True if this function calls itself at some point
        
        Specifically, if at least one ``CFGBasicBlock`` in this ``CFGFunction.blocks`` list has an `edges_out` function
        call address that is equal to this ``CFGFunction``'s address
        """
        return self.address is not None and any(block.calls(self.address) for block in self.blocks)
    
    @property
    def is_extern_function(self) -> 'bool':
        """True if this function is an external function, False otherwise"""
        return self._is_extern_function
    
    @property
    def is_intern_function(self) -> 'bool':
        """True if this function is an internal function, False otherwise"""
        return not self._is_extern_function
    
    @property
    def function_entry_block(self) -> 'CFGBasicBlock':
        """The ``CFGBasicBlock`` that is the function entry block
        
        Specifically, returns the first ``CFGBasicBlock`` found that has the same address as this function (there ~should~
        only be one as each basic block ~should~ have a unique memory address)
        """
        for block in self.blocks:
            if block.address == self.address:
                return block
        raise ValueError("Function %s does not have an entry block!" % (repr(self.name) if self.name is not None else 'with no name'))
    
    @property
    def called_by(self) -> 'list[CFGBasicBlock]':
        """A list of ``CFGBasicBlock``'s that call this function
        
        Specifically, the list of all ``CFGBasicBlock`` objects in this function's `.parent_cfg` CFG object that call 
        this function. If this ``CFGFunction`` has no parent, then the empty list will be returned.
        
        NOTE: this is computed dynamically each call (as ``CFG`` objects are mutable), so it may be useful to compute it
        once per function and save it if needed
        """
        if self.parent_cfg is None:
            return list()

        return [block for block in self.parent_cfg.blocks if block.calls(self)]
    
    def __str__(self) -> 'str':
        un_funcs = set([self.parent_cfg.get_block(b.address).parent_function.address for b in self.called_by])
        include_self = " (including self)" if self.is_recursive else ""
        extra_str = ", called by %d basic blocks across %d functions%s" % (len(self.called_by), len(un_funcs), include_self)
        addr_str = ("0x%08x" % self.address) if self.address is not None else "NO_ADDRESS"

        func_name_str = ("'%s'" % self.name) if self.name is not None else "with no name"
        num_asm_lines = sum([len(b.asm_lines) for b in self.blocks])
        return "CFGFunction %s %s at %s with %d blocks, %d assembly lines%s with metadata: %s" \
            % ('externfunc' if self.is_extern_function else 'innerfunc', func_name_str, addr_str, len(self.blocks), 
                num_asm_lines, extra_str, self.metadata)
    
    def __repr__(self) -> 'str':
        return str(self)
    
    def __eq__(self, other) -> 'bool':
        if not isinstance(other, CFGFunction) or len(self.blocks) != len(other.blocks) or \
            not all(eq_obj(self, other, selector=s) for s in ['address', 'name', 'is_extern_function', 'metadata']):
            return False
        
        # Check all the blocks are the same using a dictionary since sets don't work for some reason...
        d_self, d_other = {}, {}
        for blocks, count_dict in [(self.blocks, d_self), (other.blocks, d_other)]:
            for b in blocks:
                _hash = hash(b)
                if _hash not in count_dict:
                    count_dict[_hash] = (1, b)
                else:
                    assert b == count_dict[_hash][1]
                    count_dict[_hash] = (count_dict[_hash][0] + 1, b)
        
        return eq_obj(d_self, d_other)
    
    def __hash__(self) -> 'int':
        return hash_obj([self.name, self.address, self.blocks, self._is_extern_function, self.metadata], return_int=True)

    def __getstate__(self) -> 'CFGFunctionPickledState':
        """Print a warning about pickling singleton function objects"""
        warnings.warn("Attempting to pickle a singleton function object! This will mess up edges unless you know what you're doing!")
        return self._get_pickle_state()
    
    def __setstate__(self, state: 'CFGFunctionPickledState') -> None:
        """Print a warning about pickling singleton function objects"""
        warnings.warn("Attempting to unpickle a singleton function object! This will mess up edges unless you know what you're doing!")
        self._set_pickle_state(state)
    
    def _get_pickle_state(self) -> 'CFGFunctionPickledState':
        """Returns info of this CFGFunction as a tuple"""
        return (self.address, self.name, tuple(b._get_pickle_state() for b in self.blocks), self.is_extern_function)

    def _set_pickle_state(self, state: 'CFGFunctionPickledState'):
        """Sets state from _get_pickle_state"""
        self.address, self.name, blocks, self._is_extern_function = state
        self.blocks = [CFGBasicBlock(parent_function=self)._set_pickle_state(b) for b in blocks]
        return self
