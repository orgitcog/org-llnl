"""
Classes/Methods involving edges in a ``CFG`` object
"""

import re
import bincfg
from ..utils import eq_obj
from enum import Enum
from ..utils.type_utils import *


# First value is None since the values start at 1
EDGE_TYPE_RES = [None, re.compile(r'(?:normal|plain|jump|branch)(?:[ -_]?edge)?'), re.compile(r'(?:call|fc|function|func)(?:[ -_]?call)?(?:[ -_]?edge)?')]


# Different edge types. Used when parsing a graphviz dot file
class EdgeType(Enum):
    """Enum for different edge types for ``CFGBasicBlock`` objects."""

    NORMAL = 1
    """a normal edge as a result of some branching/jumping instruction, or plain continuation to a next block
    
    (IE: an edge of control flow that does not involve calling a function)
    """

    FUNCTION_CALL = 2
    """an edge going from a basic block to another basic block in another function (or the same function). 
    
    The outgoing edge should always connect to a function entry block (IE: that block's ``.is_function_entry`` would be 
    True).
    """


def get_edge_type(edge_type: 'Union[EdgeType, str]') -> EdgeType:
    """Returns the edge type (instance of EdgeTypes enum class)

    Args:
        edge_type (Union[EdgeType, str]): can be either an EdgeTypes object, or a string. String values include:
            - 'normal': a EdgeTypes.NORMAL edge
            - 'function_call': a EdgeTypes.FUNCTION_CALL edge

    Raises:
        ValueError: for an unknown EdgeType string
        TypeError: for a bad `edge_type` type

    Returns:
        EdgeType: the given `edge_type` as a class from the EdgeType enum
    """
    if isinstance(edge_type, EdgeType):
        return edge_type
    elif isinstance(edge_type, int):
        for et in EdgeType:
            if et.value == edge_type:
                return et
        raise ValueError("Unknown EdgeType integer value: %d" % edge_type)
    elif isinstance(edge_type, str):
        lower_edge_type = edge_type.lower()
        for et in EdgeType:
            if EDGE_TYPE_RES[et.value].fullmatch(lower_edge_type):
                return et
        raise ValueError("Unknown EdgeType string: %s" % repr(edge_type))
    else:
        raise TypeError("Cannot get an EdgeType from object of type '%s'" % type(edge_type).__name__)


class CFGEdge:
    """A single immutable edge in a ``CFG`` object
    
    Parameters
    ----------
    from_block: `CFGBasicBlock`
        'from' ``CFGBasicBlock`` object
    to_block: `CFGBasicBlock`
        'to' ``CFGBasicBlock`` object
    edge_type: `Union[EdgeType, str]`
        the edge type. can be either an ``EdgeTypes`` object, or a string. String values include:
            - 'normal': a ``EdgeTypes.NORMAL`` edge
            - 'function_call': a ``EdgeTypes.FUNCTION_CALL`` edge
    """

    from_block: 'bincfg.CFGBasicBlock'
    """The from block of this directed edge"""
    to_block: 'bincfg.CFGBasicBlock'
    """The to block of this directed edge"""
    edge_type: 'EdgeType'
    """The type of this edge"""

    def __init__(self, from_block: 'bincfg.CFGBasicBlock', to_block: 'bincfg.CFGBasicBlock', edge_type: 'Union[EdgeType, str]'):
        if not isinstance(from_block, bincfg.CFGBasicBlock) or not isinstance(to_block, bincfg.CFGBasicBlock):
            raise TypeError("Both `from_block` and `to_block` inputs must be CFGBasicBlock's. Got: %s and %s"
                % (repr(type(from_block).__name__), repr(type(to_block).__name__)))
        
        self.from_block = from_block
        self.to_block = to_block
        self.edge_type = get_edge_type(edge_type)
        self._initialized: bool = True
    
    def __setattr__(self, name: 'str', value: 'Any') -> 'None':
        if hasattr(self, '_initialized') and self._initialized:
            raise TypeError("Cannot re-initialize immutable CFGEdge!")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: 'str') -> 'NoReturn':
        raise TypeError("Cannot delete attributes on immutable CFGEdge!")
    
    @property
    def is_normal_edge(self) -> 'bool':
        """True if this is a 'normal' edge type, False otherwise"""
        return self.edge_type is EdgeType.NORMAL

    @property
    def is_function_call_edge(self) -> 'bool':
        """True if this is a 'function_call' edge type, False otherwise"""
        return self.edge_type is EdgeType.FUNCTION_CALL
    
    @property
    def is_branch(self) -> 'bool':
        """True if this edge is one of a branching instruction, False otherwise
        
        Specifically, returns True if this edge's `from_block` has exactly two outgoing edges, both of which are 'normal'
        edges. Sometimes, it is possible for blocks to have more than two 'normal' edges out (IE: jump tables), and 
        those are NOT considered branches and this method would return False
        """
        return len(self.from_block.edges_out) == 2 and all(e.edge_type is EdgeType.NORMAL for e in self.from_block.edges_out)
    
    def __str__(self) -> 'str':
        from_addr = ('0x%08x' % self.from_block.address if self.from_block.address is not None else 'NO_ADDRESS')
        to_addr = ('0x%08x' % self.to_block.address if self.to_block.address is not None else 'NO_ADDRESS')
        return "CFGEdge of type 'EdgeType.%s' from basic block at address %s to basic block at address %s" % \
            (self.edge_type.name, from_addr, to_addr)
    
    def __repr__(self) -> 'str':
        return self.__str__()
    
    def __eq__(self, other: 'Any') -> 'bool':
        """Equality for edges. Checks the to/from basic block addresses, and edge type"""
        return isinstance(other, CFGEdge) and \
            (self.from_block.address, self.to_block.address, self.edge_type.value) == (other.from_block.address, other.to_block.address, other.edge_type.value)
    
    def __hash__(self) -> 'int':
        return bincfg.hash_obj([self.from_block.address, self.to_block.address, self.edge_type.value], return_int=True)
    
    def __lt__(self, other: 'CFGEdge') -> 'bool':
        return (self.from_block.address, self.to_block.address, self.edge_type.value) < (other.from_block.address, other.to_block.address, other.edge_type.value)
    def __gt__(self, other: 'CFGEdge') -> 'bool':
        return (self.from_block.address, self.to_block.address, self.edge_type.value) > (other.from_block.address, other.to_block.address, other.edge_type.value)
    def __le__(self, other: 'CFGEdge') -> 'bool':
        return (self.from_block.address, self.to_block.address, self.edge_type.value) <= (other.from_block.address, other.to_block.address, other.edge_type.value)
    def __ge__(self, other: 'CFGEdge') -> 'bool':
        return (self.from_block.address, self.to_block.address, self.edge_type.value) >= (other.from_block.address, other.to_block.address, other.edge_type.value)
