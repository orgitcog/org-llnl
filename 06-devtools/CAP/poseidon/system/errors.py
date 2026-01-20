"""Handles errors in the system"""

from enum import Enum


# The number of the last error, 0 if no error
ERRNO = 0


def get_errno():
    """Returns the last ERRNO value, 0 if there was no error"""
    return ERRNO


def set_errno(errno):
    """Sets the current ERRNO value to the given errno"""
    global ERRNO
    ERRNO = errno


def error_raised():
    """Returns True if an error has been raised, False otherwise"""
    return ERRNO != 0


class LinuxErrors(Enum):
    """Different errors for linux os"""
    EAGAIN = 1
    EBADF = 2
    EDESTADDRREQ = 3
    EDQUOT = 4
    EFBIG = 5
    EINTR = 6
    EINVAL = 7
    EIO = 8
    ENOSPC = 9
    EPERM = 10
    EPIPE = 11
