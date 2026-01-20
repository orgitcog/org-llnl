import warnings
import bincfg
from collections import Counter
from .cfg_edge import get_edge_type, EdgeType, CFGEdge
from ..utils import get_address, eq_obj, hash_obj
from ..utils.type_utils import *


if IN_PYTHON_TYPING_VERSION:
    CFGBasicBlockPickledState = Tuple[int, Tuple[Tuple[int, int, EdgeType], ...], Tuple[Tuple[int, int, EdgeType], ...], list[str], list[int], dict]
    """The pickled state of a CFGBasicBlock"""


class CFGBasicBlock:
    """A single basic block in a ``CFG``.

    Can be initialized empty, or with attributes. Assumes its memory address is always unique within a cfg.

    NOTE: these objects should not be pickled/copy.deepcopy()-ed by themselves, only as a part of a cfg

    Parameters
    ----------
    parent_function: `Optional[CFGFunction]`
        the ``CFGFunction`` this basic block belongs to
    address: `Optional[Union[int, str, Addressable]]`
        the memory address of this ``CFGBasicBlock``. Should be unique to the ``CFG`` that contains it. If None, but
        `asm_memory_addresses` is passed, this will be set to the first value in `asm_memory_addresses`
    edges_in: `Optional[Iterable[CFGEdge]]` 
        an iterable of incoming CFGEdge objects
    edges_out: `Optional[Iterable[CFGEdge]]`
        an iterable of outgoing CFGEdge objects
    asm_lines: `Optional[Iterable[str]]`
        an iterable of string assembly lines present at this basic block
    asm_memory_addresses: `Optional[Iterable[Union[str, int, Addressable]]]`
        an iterable of string or integer memory addresses, one for each assembly line (will be converted into integer 
        memory addresses). If this was passed, but `address` was not, then `address` will be set to the first value in
        `asm_memory_addresses`
    metadata: `Optional[Dict]`
        optional dictionary of metadata to associate with this basic block
    """

    parent_function: 'Union[bincfg.CFGFunction, None]'
    """The parent function of this basic block. Will be None if not set yet"""
    address: 'int'
    """The integer memory address of this basic block. Will be -1 if not set yet"""
    edges_in: 'set[CFGEdge]'
    """The set of incomming ``CFGEdge``'s to this basic block"""
    edges_out: 'set[CFGEdge]'
    """The set of outgoing ``CFGEdge``'s from this basic block"""
    asm_lines: 'list[str]'
    """List of string assembly lines at this basic block"""
    asm_memory_addresses: 'list[int]'
    """List of integer memory addresses for all assembly lines at this basic block. Will be empty list if not set yet"""
    metadata: 'dict'
    """Dictionary of extra metadata to associate with this basic block"""


    def __init__(self, parent_function: 'Optional[bincfg.CFGFunction]' = None, address: 'Optional[AddressLike]' = None, 
                 edges_in: 'Optional[Iterable[CFGEdge]]' = None, edges_out: 'Optional[Iterable[CFGEdge]]' = None, 
                 asm_lines: 'Optional[Iterable[str]]' = None, asm_memory_addresses: 'Optional[Iterable[AddressLike]]' = None, 
                 metadata: 'Optional[dict]' = None):
        # Get the memory addresses all figured out
        self.asm_memory_addresses = [] if asm_memory_addresses is None else [get_address(addr) for addr in asm_memory_addresses]
        self.asm_memory_addresses: 'list[int]' = [] if all(a < 0 for a in self.asm_memory_addresses) else self.asm_memory_addresses

        if address is None and len(self.asm_memory_addresses) > 0:
            self.address = self.asm_memory_addresses[0]
        else:
            self.address = get_address(address) if address is not None else -1

        # Set the rest of the params
        self.parent_function = parent_function
        self.edges_in = set() if edges_in is None else set(edges_in)
        self.edges_out = set() if edges_out is None else set(edges_out)
        self.asm_lines = [] if asm_lines is None else list(asm_lines)
        self.metadata = {} if metadata is None else metadata
    
    @property
    def num_edges(self) -> 'int':
        """The number of edges out in this basic block"""
        return self.num_edges_out
    
    @property
    def num_edges_out(self) -> 'int':
        """The number of outgoing edges in this basic block"""
        return len(self.edges_out)

    @property
    def num_edges_in(self) -> 'int':
        """The number of incoming edges in this basic block"""
        return len(self.edges_in)
    
    @property
    def num_asm_lines(self) -> 'int':
        """The number of assembly lines/tokens in this basic block"""
        return len(self.asm_lines)
    
    @property
    def asm_counts(self) -> 'Mapping[str, int]':
        """A ``collections.Counter`` of all unique assembly lines/tokens and their counts in this basic block"""
        return Counter(l for l in self.asm_lines)
    
    @property
    def is_function_entry(self) -> 'bool':
        """True if this block is a function entry block, False otherwise
        
        Specifically, returns True if this block's address matches its parent function's address. If this block has
        no parent, False is returned.
        """
        return self.parent_function is not None and self.address == self.parent_function.address
    
    @property
    def is_function_call(self) -> 'bool':
        """True if this block is a function call, False otherwise
        
        Checks if this block has one or more outgoing function call edges
        """
        return any(e.edge_type is EdgeType.FUNCTION_CALL for e in self.edges_out)

    @property
    def is_function_jump(self) -> 'bool':
        """True if this block is a function jump, False otherwise
        
        Checks if this block has a 'jump' instruction to a basic block in a different function. Specifically, checks if
        this block has an outgoing EdgeType.NORMAL edge to a basic block who's parent_function has an address different
        than this basic block's parent_function's address.
        """
        return self.parent_function is not None and \
            any((e.edge_type is EdgeType.NORMAL and e.to_block.parent_function is not None 
                 and e.to_block.parent_function.address != self.parent_function.address) for e in self.edges_out)
    
    @property
    def is_multi_function_call(self) -> 'bool':
        """True if this block is a multi-function call, False otherwise
        
        IE: this block has either two or more function call edges out
        """
        return len(self.get_sorted_edges(edge_types='function_call', direction='out')[0]) >= 2
    
    @property
    def all_edges(self) -> 'set[CFGEdge]':
        """Returns a set of all edges in this basic block"""
        return self.edges_in.union(self.edges_out)
    
    def remove_edge(self, edge: 'CFGEdge') -> None:
        """Removes the given edge from this block's edges (both incoming and outgoing)
        
        Args:
            edge (CFGEdge): the CFGEdge to remove
        
        Raises:
            ValueError: if the edge doesn't exist in the incomming/outgoing edges
        """
        if not isinstance(edge, CFGEdge):
            raise TypeError("edge must be a CFGEdge, not %s" % repr(type(edge).__name__))
        if edge not in self.edges_in and edge not in self.edges_out:
            raise ValueError("%s does not exist in this block's (%s) edges in or out!" % 
                             (edge, ("0x%x" % self.address) if self.address is not None else "NoAddr"))
        
        if edge in self.edges_in:
            self.edges_in.remove(edge)
        if edge in self.edges_out:
            self.edges_out.remove(edge)

    def has_edge(self, address: 'AddressLike', edge_types: 'Optional[Union[str, EdgeType, Iterable[Union[str, EdgeType]]]]' = None, 
                 direction: "Optional[Literal['in', 'out']]" = None) -> 'bool':
        """Checks if this block has an edge from/to the given address

        Args:
            address (AddressLike): a string/integer memory address, or an addressable object (EG: ``CFGBasicBlock``/``CFGFunction``). 
            edge_types (Optional[Union[str, EdgeType, Iterable[Union[str, EdgeType]]]]): either an edge type or an
                iterable of edge types. Only edges with one of these types will be considered. If None, then all edge 
                types will be considered
            direction (Optional[Literal['in', 'out']]): the direction to check (strings 'in' or 'out), or None to check both

        Returns:
            bool: True if this block has an edge from/to the given address, False otherwise
        """
        addr, edge_types, directions = get_address(address), _get_edge_types(edge_types, as_set=True), self._get_directions(direction)

        # Check for that edge
        for edge_set in directions:
            for edge in edge_set:
                # Check both that it is an allowable edge_type, and that the edge from/to address matches the given address
                if edge.edge_type in edge_types and \
                    ((edge_set is self.edges_in and edge.from_block.address == addr) or (edge_set is self.edges_out and edge.to_block.address == addr)):
                    return True

        return False
    
    def calls(self, address: 'AddressLike'):
        """Checks if this block calls the given address

        IE: checks if this block has an outgoing `function_call` edge to the given address

        Args:
            address (AddressLike): a string/integer memory address, or an addressable object (EG: ``CFGBasicBlock``/``CFGFunction``)

        Returns:
            bool: True if this block calls the given address, False otherwise
        """
        return self.has_edge(address=address, edge_types='function_call', direction='to')
    
    def get_sorted_edges(self, edge_types: 'Optional[Union[str, EdgeType, Iterable[Union[str, EdgeType]]]]' = None, 
                         direction: 'Optional[Union[Literal["out", "in"], Iterable[Literal["out", "in"]]]]' = None, 
                         as_sets: bool = False) -> 'Union[Tuple[list[CFGEdge], ...], Tuple[set[CFGEdge], ...]]':
        """Returns a tuple of sorted lists of edges (sorted by address of the "other" block) of each type/direction in this block
        
        Will return edge lists ordered first by edge type (their order of appearance in the cfg_edge.EdgeType enum),
        then by direction ('in', then 'out'). Unless, if `edge_types` is passed, then only those edge types will be
        returned and the edge lists will be returned in the order of the edge types in `edge_types`, then by direction
        ('in', then 'out').

        For example, with `edge_types=None` and `direction=None`, this would return the 4-tuple of:
        (normal_edges_in, normal_edges_out, function_call_edges_in, function_call_edges_out)
        Where each element is a list of CFGEdge objects.

        Args:
            edge_types (Optional[Union[str, EdgeType, Iterable[Union[str, EdgeType]]]]): either an edge type or an
                iterable of edge types. Only edges with one of these types will be returned. If not None, then the edge 
                lists will be returned sorted based on the order of the edge types listed here, then by direction
            direction (Optional[Union[Literal["out", "in"], Iterable[Literal["out", "in"]]]): the direction to get. Can 
                be the strings 'in' or 'out', or None to get both
            as_sets (bool): if True, then this will return unordered sets of edges instead of sorted lists. This may
                save a ~tiny~ bit of time in the long run, but will hinder deterministic behavior of this method.

        Returns:
            Union[Tuple[List[CFGEdge], ...], Tuple[Set[CFGEdge], ...]]: a tuple of lists/sets of CFGEdge's
        """
        edge_types, directions = _get_edge_types(edge_types, as_set=False), self._get_directions(direction)

        # Iterate through all edges, adding them to their appropriate sets
        ret_sets = [set() for d in directions for et in edge_types]
        for i, edge_set in enumerate(directions):
            for edge in edge_set:
                if edge.edge_type in edge_types:
                    ret_sets[edge_types.index(edge.edge_type) * len(directions) + i].add(edge)
        
        # Check if we need to sort the edges. If so, sort based on the address of the "other" block
        return ret_sets if as_sets else [list(sorted(s, key=lambda edge: 
                (edge.from_block.address if directions[i % len(directions)] is self.edges_in else edge.to_block.address))) 
            for i, s in enumerate(ret_sets)]

    def _get_directions(self, direction: 'Literal["in", "out", "to", "from"]') -> 'list[list[CFGEdge]]':
        """Gets the directions

        Args:
            direction (Union[str, None]): the direction to get (strings 'in'/'from' or 'to'/'out'), or None to get both 
                (in order ['in', 'out'])

        Raises:
            ValueError: on an unknown direction

        Returns:
            List[List[CFGEdge]]: a list of edges in/out based on direction
        """
        if direction is None:
            return [self.edges_in, self.edges_out]
        elif isinstance(direction, str):
            if direction not in ['from', 'to', 'in', 'out']:
                raise ValueError("Unknown `direction`: %s" % repr(direction))
            return [self.edges_in] if direction in ['from', 'in'] else [self.edges_out]
        else:
            return [self._get_directions(v)[0] for v in direction]
        
    def __str__(self) -> 'str':
        valid_pf = isinstance(self.parent_function, bincfg.CFGFunction)
        asm = '\n'.join([("\t0x%s: %s" % ('%08x' % addr, line)) for addr, line in zip(self.asm_memory_addresses, self.asm_lines)]) \
            if len(self.asm_memory_addresses) > 0 else '\n'.join([("\t%s" % line) for line in self.asm_lines])
        func_name = '' if not valid_pf or self.parent_function.name is None else self.parent_function.name

        func_str = ('in function \"%s\"' % func_name) if valid_pf else 'with NO PARENT'
        addr_str = ('0x%08x' % self.address) if self.address is not None else 'NO_ADDRESS'

        ret = "CFGBasicBlock %s at %s with %d edges out, %d edges in, with %d lines of assembly:\n%s\nEdges Out: %s\nEdges In: %s\n" \
            % (func_str, addr_str, self.num_edges_out, self.num_edges_in, len(self.asm_lines), asm, self.edges_out, self.edges_in)
        
        return (ret + "Metadata: %s" % self.metadata) if len(self.metadata) > 0 else ret
    
    def __repr__(self) -> 'str':
        return str(self)
    
    def __eq__(self, other: 'Any') -> 'bool':
        return isinstance(other, CFGBasicBlock) and all(eq_obj(self, other, selector=s) for s in \
            ['address', 'edges_in', 'edges_out', 'metadata', 'asm_memory_addresses', 'asm_lines'])
    
    def __hash__(self) -> 'int':
        return hash_obj([self.address, self.edges_in, self.edges_out, self.metadata, self.asm_lines, self.asm_memory_addresses], return_int=True)
    
    def __getstate__(self) -> 'CFGBasicBlockPickledState':
        """Print a warning about pickling singleton basic block objects"""
        warnings.warn("Attempting to pickle a singleton basic block object! This will mess up edges unless you know what you're doing!")
        return self._get_pickle_state()
    
    def __setstate__(self, state: 'CFGBasicBlockPickledState') -> 'None':
        """Print a warning about pickling singleton basic block objects"""
        warnings.warn("Attempting to unpickle a singleton basic block object! This will mess up edges unless you know what you're doing!")
        self._set_pickle_state(state)
    
    def _get_pickle_state(self) -> 'CFGBasicBlockPickledState':
        """Returns info of this CFGBasicBlock as a tuple"""
        edges_in = tuple((e.from_block.address, e.to_block.address, e.edge_type) for e in self.edges_in)
        edges_out = tuple((e.from_block.address, e.to_block.address, e.edge_type) for e in self.edges_out)
        return (self.address, edges_in, edges_out, self.asm_lines, self.asm_memory_addresses, self.metadata)
    
    def _set_pickle_state(self, state: 'CFGBasicBlockPickledState') -> 'CFGBasicBlock':
        """Set the pickled state, looking at parent function.parent_cfg for info for building edges"""
        self.address, self._temp_edges_in, self._temp_edges_out, self.asm_lines, self.asm_memory_addresses, self.metadata = state
        return self


def _get_edge_types(edge_types: 'Optional[Union[str, EdgeType, Iterable[Union[str, EdgeType]]]]' = None, as_set: bool = False):
    """Gets the edge types passed by the user
    
    Args:
        edge_types (Optional[Union[str, EdgeType, Iterable[Union[str, EdgeType]]]]): either an edge type or an 
            iterable of edge types. Only edges with one of these types will be considered. Defaults to None.
        as_set (bool): if True, will return the result as a set instead. Defaults to False.

    Raises:
        TypeError: on an unknown `edge_types`

    Returns:
        Union[List[EdgeType], Set[EdgeType]]: a list/set of EdgeType objects
    """
    ret_type = set if as_set else list

    if edge_types is None:
        return ret_type(e for e in EdgeType)
    else:
        # Attempt a single EdgeTypeLike first
        try:
            return ret_type([get_edge_type(edge_types)])
        except (ValueError, TypeError):
            try:
                return ret_type(get_edge_type(et) for et in edge_types)
            except Exception:
                raise TypeError("Could not get acceptable edge types from `edge_types` of type %s" % type(edge_types).__name__)
