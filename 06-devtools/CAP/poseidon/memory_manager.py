"""
Class to help with loading things into memory and making sure they don't overwrite eachother
"""

from triton import ARCH


class MemoryManager:
    """A class to handle loading binaries (and their libraries) into memory correctly
    
    Parameters
    ----------
        arch (triton.ARCH): the triton architecture being used
    """

    executor = None
    """The TritonExecutor"""

    allocated_blocks = None
    """List of tuples of memory areas. Each tuple is a [start_addr, end_addr), start inclusive, end exclusive
    
    This will always be sorted.
    """

    def __init__(self, executor):
        self.executor = executor
        self.allocated_blocks = []
        self.max_addr = 2 ** 32 if self.executor.binary_info['arch'] in [ARCH.X86, ARCH.ARM32] else 2 ** 64  # Exclusive endpoint!
    
    @property
    def available_blocks(self):
        """Returns a list of [start_addr, end_addr) tuples of available memory blocks"""
        if len(self.allocated_blocks) == 0:
            return [(0, self.max_addr)]
        
        ret = [(0, self.allocated_blocks[0][0])] if self.allocated_blocks[0][0] > 0 else []
        for i in range(len(self.allocated_blocks) - 1):
            app = (self.allocated_blocks[i][1], self.allocated_blocks[i + 1][0])
            if app[1] - app[0] > 0:
                ret.append(app)
        return ret + [(self.allocated_blocks[-1][1], self.max_addr)]
    
    def _insert_block(self, address, size):
        """Inserts the given starting address/size into the current list of memory blocks in order"""
        if len(self.allocated_blocks) == 0 or address >= self.allocated_blocks[-1][1]:
            self.allocated_blocks.append((address, address + size))
        else:
            for i, (start, end) in enumerate(self.allocated_blocks):
                if address < start:
                    break
            self.allocated_blocks.insert(i, (address, address + size))
    
    def _conflicting(self, address, size):
        """Returns True if the given address/size conflicts with other allocated memory (BAD), False otherwise (GOOD)"""
        address_end = address + size

        # Check for overflowing max memory size
        if address_end >= self.max_addr:
            return True

        # Check for conflictions with previously allocated memory
        for start_addr, end_addr in self.allocated_blocks:
            if (address >= start_addr and address < end_addr) or (address_end >= start_addr and address_end < end_addr):
                return True
        
        return False
    
    def _memory_aligned(self, address, alignment):
        """Returns True if the given memory address fits the given alignment, False otherwise"""
        return address % alignment == 0
    
    def _ensure_aligned(self, address, alignment):
        """Returns the given address if it matches the alignment, or the nearest address larger that does if it doesnt match"""
        return address if self._memory_aligned(address, alignment) else (address + alignment - (address % alignment))
    
    def _smallest_fitting_gap(self, size):
        """Returns the starting memory address of the smallest fitting gap, returning None if not possible"""
        blocks = [(start, end - start) for start, end in self.available_blocks if end - start >= size]

        if len(blocks) == 0:
            return None
        
        return min(blocks, key=lambda x: x[1])[0]
    
    def malloc(self, size, address=None, force_addr=False, alignment=1):
        """Allocates memory
        
        Args:
            size (int): the size in bytes to allocate. Must be > 0
            address (Optional[int]): a suggested address to load it into. If this breaks things, it will be loaded into
                another address (unless force_addr=True, in which case an error will be raised)
            force_addr (bool): if True, address is not None, and the act of allocating memory at the given address would
                conflict with other memory, then an error will be raised
            alignment (int): integer memory address alignment
        
        Returns:
            int: the memory address of the allocated memory
        """
        if size <= 0:
            raise ValueError("size must be > 0, got: %d" % size)
        if address is not None and address < 0:
            raise ValueError("address, if not None, must be >= 0, got: %d" % address)
        
        # If we have a forced memory address
        if address is not None and force_addr:
            malloc_addr = address

            # Check if the forced address causes problems
            if not self._memory_aligned(malloc_addr, alignment):
                raise ValueError("Forced memory address 0x%016x does not fit with alignment 0x%x" % (malloc_addr, alignment))
            if self._conflicting(malloc_addr, size):
                raise ValueError("Forced memory address 0x%016x is too large, or conflicts with other previously malloc-ed memory!" 
                    % malloc_addr)

        # If there is a hint address, check to see if it works
        elif address is not None:
            malloc_addr = self._ensure_aligned(address, alignment)
            malloc_addr = self._smallest_fitting_gap(size) if self._conflicting(malloc_addr, size) else malloc_addr
        
        # Otherwise, pick the smallest address that would fit
        else:
            malloc_addr = self._smallest_fitting_gap(size)
        
        # Check if the automatic allocation failed
        if malloc_addr is None:
            raise ValueError("Could not allocate %d bytes in memory!" % size)
        
        # Add this new allocated block
        self._insert_block(malloc_addr, size)
        
        return malloc_addr

    def calloc(self, size, fill_val=0, address=None, force_addr=False, alignment=1):
        """Allocates memory, filling all bytes with the specified value (defaults to 0)
        
        Args:
            size (int): the size in bytes to allocate. Must be > 0
            fill_val (int): the value to fill bytes with. Must be in range [0, 255]. Defaults to 0
            address (Optional[int]): a suggested address to load it into. If this breaks things, it will be loaded into
                another address (unless force_addr=True, in which case an error will be raised)
            force_addr (bool): if True, address is not None, and the act of allocating memory at the given address would
                conflict with other memory, then an error will be raised
            alignment (int): integer memory address alignment
        
        Returns:
            int: the memory address of the allocated memory
        """
        if fill_val < 0 or fill_val > 255:
            raise ValueError("fill_val must be in range [0, 255], got: %d" % fill_val)

        addr = self.malloc(size, address=address, force_addr=force_addr, alignment=alignment)
        self.executor.context.setConcreteMemoryAreaValue(addr, bytes([fill_val] * size))
        return addr

    def free(self, addr):
        """Frees the previously allocated memory block. Addr should be the start address of a block
        
        Args:
            addr (int): the start address of a block of memory to free
        """
        for i, (start, end) in enumerate(self.allocated_blocks):
            if start == addr:
                self.allocated_blocks.pop(i)
                return
        
        raise ValueError("Could not find allocated block starting at memory address %x to free" % addr)
