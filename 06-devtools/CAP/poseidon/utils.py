"""Utility functions"""

import numpy as np
import logging
import struct
import sys
from bincfg import CFG


# Maximum number of bits to check for in get_set_bit_vals()
GET_SET_BIT_LIMIT = 64


def get_set_bit_vals(val):
    """Returns a set of all values of set bits up to 64 bits (IE: if the 10th bit is set, then 2^10 would be in this list"""
    return {2 ** i for i in range(GET_SET_BIT_LIMIT) if val & (2 ** i) > 0}


def _enc_str(b, encoding):
    return b.encode(encoding) if encoding is not None else b


READ_STR_BUF_SIZE = 100
def read_string(be, mem_addr, count=None, encoding='utf-8'):
    """Reads up to count bytes of a null-terminated string starting at the given memory address
    
    If encoding is None, then a bytes() object is returned. Otherwise that object is encoded using the given encoding
    """
    # If we know a maximum to read
    if count is not None:
        b = be.context.getConcreteMemoryAreaValue(mem_addr, count)
        try:
            null_idx = b.index(0)
            return _enc_str(b[:null_idx], encoding)
        except:
            return _enc_str(b, encoding)
    
    # Otherwise, keep reading until we get a 0
    b = bytes()
    while True:
        b += be.context.getConcreteMemoryAreaValue(mem_addr, READ_STR_BUF_SIZE)
        mem_addr += READ_STR_BUF_SIZE

        try:
            null_idx = b.index(0)
            return _enc_str(b[:null_idx], encoding)
        except:
            continue


def read_bytes(be, mem_addr, count):
    """Reads count bytes from the given memory address"""
    return be.context.getConcreteMemoryAreaValue(mem_addr, count)


def random_strings(max_num, max_size) -> list[bytes]:
    """Generate random cmd line args"""
    return [bytes(np.random.randint(1, 256, size=np.random.randint(0, max_size), dtype=np.uint8)) + bytes([0]) \
        for _ in range(np.random.randint(0, max_num + 1))]


def load_segments(segments, context, memory_mangager, force_addr=False):
    """Loads the given segments into memory. Returns the starting memory address it was loaded into

    Loades all segments sequentially, consuming all memory from the start to ending addresses.
    
    Args:
        segments (List[Tuple[int, int, bytes]]): segments from get_binary_segments()
        context (TritonContext): the triton context to load into
        memory_manager (MemoryManager): the memory manager to use
        force_addr (bool): if True, then the segments will be forced to be at their given addresses
    
    Returns:
        int: the starting memory address the segments were loaded into
    """
    logging.info("Loading %d segments into memory..." % len(segments))

    start_addr = min([s[0] for s in segments])
    end_addr = max([s[0] + s[1] for s in segments])

    # We shouldn't need to zero-pad the content as we are allocating the memory, and I assume Triton starts off memory
    #   values with 0?
    offset = memory_mangager.malloc(end_addr - start_addr, address=start_addr, force_addr=force_addr)
    logging.info("Loading %d bytes of content into address 0x%016x" % (end_addr - start_addr, offset))
    for addr, _, content in segments:
        context.setConcreteMemoryAreaValue(addr + offset, content)
    
    return offset


def get_float_bits(val, size):
    """Converts val (a float) to its IEEE 754 floating point representation as an integer
    
    Args:
        val (float): the value to convert
        size (Literal[4, 8]): the size in bytes to use. Can be 4 or 8 bytes
    
    Returns:
        int: integer representation of the bits for the float val
    """
    if size not in [4, 8]:
        raise ValueError("size must be either 4 or 8")
    return int.from_bytes(struct.pack('f' if size == 4 else 'd', val), sys.byteorder)


def get_float_from_bits(val, size):
    """Converts val (an IEEE-754 floating point representation in integer form) to a float"""
    if size not in [4, 8]:
        raise ValueError("size must be either 4 or 8")
    return struct.unpack('f' if size == 4 else 'd', val.to_bytes(size, sys.byteorder))[0]


def twos_complement(val, size):
    """Returns val if val => 0, or the 2's complement in the given size in bytes if val < 0 (returns as a positive integer)"""
    if val >= 0:
        return val
    
    bits = size * 8 - 1

    if size < - (2 ** bits):
        raise ValueError("Attempting to store too large a negative number in too small a size: %d in %d bytes" % (val, size))
    
    return int.from_bytes(val.to_bytes(size, 'big', signed=True), byteorder='big')


def load_binary_cfg(binary_info):
    """Attempts to find an appropriately load the CFG associated with the given binary info"""
    if binary_info['cfg_path'] is None:
        raise ValueError("Could not find the CFG for binary %s, needed for trace! %s" % (repr(binary_info['name']),
            "It should be stored in a file next to the binary with the same name, but ending in one of the "
            "acceptable file formats: %s" % ['.pkl', '.cfg', '.txt']))
    return CFG.load(binary_info['cfg_path']) if binary_info['cfg_path'].endswith('.pkl') else CFG(binary_info['cfg_path'])
