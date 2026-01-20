"""Handles system calls"""

from .syscalls import *
import logging


def handle_syscall(be):
    """Handles syscalls

    Incoming Register Values (x86):
        - RAX -> system call number
        - RDI -> first argument
        - RSI -> second argument
        - RDX -> third argument
        - R10 -> fourth argument
        - R8 -> fifth argument
        - R9 -> sixth argument
    
    Return values (x86):
        - 0: generally, success
        - -1: error occurred, error stored in ERRNO
    
    Args:
        be (BinaryExecutor): the current binary executor
    """
    num = be.context.getConcreteRegisterValue(be.context.registers.rax)
    args = [be.get_reg_val(a) for a in ['rdi', 'rsi', 'rdx', 'r10', 'r8', 'r9']]
    syscall_table = SYSCALL_TABLES[be.binary_info['os']]

    if num not in syscall_table:
        raise ValueError("Got an unknown/unimplemented syscall number for os %s: %d" % (repr(be.binary_info['os']), num))
    
    logging.info("Executing syscall: %s" % repr(syscall_table[num].__name__))
    syscall_table[num](be, args)
    

# Dictionary mapping syscall numbers to their associated functions
SYSCALL_TABLE_LINUX = {
    0: syscall_read,
    1: syscall_write,
    2: syscall_open,

    5: syscall_fstat,

    102: syscall_getuid,

    104: syscall_getgid,

    107: syscall_geteuid,
    108: syscall_getegid,
}


# The dict of all syscall tables we have
SYSCALL_TABLES = {
    'linux': SYSCALL_TABLE_LINUX,
}