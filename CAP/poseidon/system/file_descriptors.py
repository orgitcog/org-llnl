"""
Handles file descriptors for linux-like system calls (includes stdin/out/err, networking, filesystems, etc.)
"""
from enum import Enum
import logging


class OpenFlags(Enum):
    """Flags that can be used when opening files
    
    Each file should have a set of these flags.
    """
    BUILTIN = -1  # If this is a builtin 'file' (IE: stdin/stdout/stderr)

    O_RDONLY = 0
    O_WRONLY = 1
    O_RDWR = 2
    O_APPEND = 3
    O_ASYNC = 4
    O_CLOEXEC = 5
    O_CREAT = 6
    O_DIRECT = 7
    O_DIRECTORY = 8
    O_DSYNC = 9
    O_EXCL = 10
    O_LARGEFILE = 11
    O_NOATIME = 12
    O_NOCTTY = 13
    O_NOFOLLOW = 14
    O_NONBLOCK = 15
    O_PATH = 16
    O_SYNC = 17
    O_TMPFILE = 18
    O_TRUNC = 19


class File:
    """Handles a single open file"""
    def __init__(self, fd, pathname, flags, start_data=None, start_loc=0):
        self.fd = fd
        self.pathname = pathname
        self.flags = set(flags)
        self.data = bytes() if start_data is None else start_data
        self.seek_loc = start_loc
    
    def is_writing(self):
        return OpenFlags.O_WRONLY in self.flags or OpenFlags.O_RDWR in self.flags

    def is_reading(self):
        return OpenFlags.O_RDONLY in self.flags or OpenFlags.O_RDWR in self.flags

    def write(self, data: bytes) -> int:
        """Writes the given data to file, returning the number of bytes actually written, and incrementing seek"""
        self.data += data
        self.seek_loc += len(data)
        logging.info("Wrote %d bytes to file %s: %s" % (len(data), repr(self.pathname), data))
        return len(data)
    
    def read(self, count: int):
        """Attempts to read count bytes from the current seek position, incrementing seek"""
        start_seek_loc = self.seek_loc
        self.seek_loc = min(self.seek_loc + count, len(self.data))
        return self.data[start_seek_loc: self.seek_loc]


# Stdin/out/err are of course, files
STD_IN = File(0, 'stdin', [OpenFlags.O_RDONLY, OpenFlags.BUILTIN])
STD_OUT = File(1, 'stdout', [OpenFlags.O_WRONLY, OpenFlags.BUILTIN])
STD_ERR = File(2, 'stderr', [OpenFlags.O_WRONLY, OpenFlags.BUILTIN])


# List of currently open file descriptor objects
OPEN_FILE_DESCRIPTORS = [STD_IN, STD_OUT, STD_ERR]


def get_file_from_descriptor(fd):
    """Attempts to get the open file for the given file descriptor, returning None if it doesn't exist"""
    return None if fd > len(OPEN_FILE_DESCRIPTORS) else OPEN_FILE_DESCRIPTORS[fd]


def get_next_open_descriptor():
    """Returns the next open file descriptor integer, resizing OPEN_FILE_DESCRIPTORS if needed"""
    global OPEN_FILE_DESCRIPTORS
    for i in range(len(OPEN_FILE_DESCRIPTORS)):
        if OPEN_FILE_DESCRIPTORS[i] is None:
            return i

    OPEN_FILE_DESCRIPTORS += [None] * 10
    return i + 1

