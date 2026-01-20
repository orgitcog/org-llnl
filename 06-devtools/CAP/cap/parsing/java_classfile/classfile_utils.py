"""Utilities for parsing java classfiles"""

import struct


class ByteReader:
    """Class to help with reading bytes from a bytes() object"""
    def __init__(self, bytes_obj):
        self._bytes = bytes_obj
        self.idx = 0
        self.eof = False
    
    def read(self, num=1):
        """Reads the specified number of bytes moving the seek head forward"""
        ret = self.peek(num=num)
        self.idx = min(self.idx + num, len(self))
        self.eof = self.idx >= len(self)
        return ret
    
    def peek(self, num=1):
        """Reads the specified number of bytes WITHOUT moving the seek head forward"""
        if num == 0:
            return bytes()
        elif num < 0:
            raise ValueError("num must be non-negative: %d" % num)
        elif self.eof:
            raise ValueError("Attempted to read more bytes after EOF")
        
        return self._bytes[self.idx:self.idx + num]
    
    def read_int(self, num_bytes, signed=False):
        """Reads an integer with the specified size in big-endian, moving the seek head forward"""
        return self._to_int(self.read(num_bytes), signed=signed)
    
    def peek_int(self, num_bytes, signed=False):
        """Reads an integer with the specified size in big-endian, WITHOUT moving the seek head forward"""
        return self._to_int(self.peek(num_bytes), signed=signed)
    
    def read_float(self, num_bytes):
        """Reads a float with the specified size in big-endian, moving the seek head forward"""
        if num_bytes not in [4, 8]:
            raise ValueError("Cannot read a float of %d bytes, must be either 4 or 8" % num_bytes)
        return self._to_float(self.read(num_bytes))
    
    def peek_float(self, num_bytes):
        """Reads a float with the specified size in big-endian, WITHOUT moving the seek head forward"""
        if num_bytes not in [4, 8]:
            raise ValueError("Cannot read a float of %d bytes, must be either 4 or 8" % num_bytes)
        return self._to_float(self.peek(num_bytes))
    
    def bytes_remaining(self):
        return len(self) - self.idx
    
    @staticmethod
    def _to_int(b, signed=False):
        return int.from_bytes(b, byteorder='big', signed=signed)
    
    @staticmethod
    def _to_float(b):
        return struct.unpack('>' + ('f' if len(b) == 4 else 'd'), b)
    
    def __len__(self):
        return len(self._bytes)
