"""
Syscall functions

Each function should take as input the binary executor and the args, and return either None or True if execution worked,
or False if execution failed and it should halt
"""
import lief
import logging
from .errors import set_errno, LinuxErrors
from .file_descriptors import File, OPEN_FILE_DESCRIPTORS, get_file_from_descriptor
from ..utils import read_bytes, get_float_bits
from triton import MemoryAccess, Instruction
from ..preassembled_code import ASSEMBLED_CODE_DICT


def _error(be, errno, ret_val=-1):
    """Set an error and write ret_val into return register"""
    set_errno(errno)
    be.set_reg_val('ret', ret_val)


def _success(be, ret_val=0):
    """There was a successful thing, write ret_val into return register"""
    be.set_reg_val('ret', ret_val)


def _set_symbolic_system_state_val(be, sysinfo_key, sysstate_key, default=None, allowed_types=None):
    """Sets a system state value with symbolic memory

    The memory size is inferred from the data type with alignment 1. The system_state dictionary will contain a 3-tuple
    of (memory_location, value, memory_access_size).

    NOTE: memory_access_size will be the size of the memory access, or None if there shouldn't be a memory access and
    instead should just be raw bytes.

    How types are handled:

        - int: stored as long long (4 bytes on 32-bit binary, 8 bytes on 64-bit)
        - float: stored as IEEE-754 encoded double (4 bytes on 32-bit, 8 bytes on 64-bit)
        - string: encoded as ascii and stored
    
    Args:
        be (BinaryExecutor): the binary executor
        sysinfo_key (str): the key for the system_inputs inputs
        sysstate_key (str): the key to store values at in system_state
        default (Optional[Any]): if not None, then a value to use as default in the system state if one doesn't exist
            in the system_inputs. If None, then a value must be passed in the system_inputs or an error will be raised
        allowed_types (Optional[Union[type, Tuple[type]]]): If not None, then either a type or tuple of types to make
            sure the default or sysinfo value isinstance of
    """
    if sysinfo_key in be.system_inputs:
        val = be.system_inputs[sysinfo_key]
        logging.info("Using system input %s = %s" % (repr(sysinfo_key), val))
    elif default is not None:
        val = default
        logging.info("Using DEFAULT system input %s = %s" % (repr(sysinfo_key), val))
    else:
        raise ValueError("Could not find needed sysinfo_key %s for sysstate_key %s, and no default was passed" 
                         % (repr(sysinfo_key), repr(sysstate_key)))

    if allowed_types is not None and isinstance(val, allowed_types):
        raise TypeError("Sysinfo/default value is of type %s, allowed types: %s" % (repr(type(val).__name__), allowed_types))
    
    # Different data types have different methods of encoding
    if isinstance(val, (int, float)):
        # Convert values if needed
        if isinstance(val, float):
            val = get_float_bits(val, be.pointer_size)

        loc = be.memory_manager.malloc(be.pointer_size)
        mem_acc = MemoryAccess(loc, be.pointer_size)
        be.context.setConcreteMemoryValue(mem_acc, val)
        # symbolize me
        
    elif isinstance(val, str):
        val = val.encode('ascii')
        loc = be.memory_manager.malloc(len(val))
        be.context.setConcreteMemoryAreaValue(loc, val)
        # symbolize me

    else:
        raise NotImplementedError("Unimplemented data type: %s" % repr(type(val).__name__))
    
    be.system_state[sysstate_key] = loc


def _read_mem(be, loc):
    """Reads a memory location
    
    Does this by placing the memory location to read in a known address, and executing pre-assembled code to move the
    value into the eax/rax register.

    Steps:

        1. Save the original pc since executing new code will change it
        2. Store loc into the first location in scratch space
        3. Load the first location in scratch space into the return register (EG: eax, rax, etc.)
        4. Load the value at the memory address in the return register into rax
        5. Reset the pc

    Args:
        be (BinaryExecutor): the executor being used
        loc (int): the memory location to read from
    """
    arch = be.binary_info['arch']

    # Check that the architecture is supported
    if arch not in ASSEMBLED_CODE_DICT:
        raise ValueError("Attempting to use assembled code for an unsupported architecture %s, supported arch's: %s"
                        % (arch, list(ASSEMBLED_CODE_DICT.keys())))
    
    # Keep track of the original program counter to reset it when done executing these codes
    pc_orig = be.get_reg_val('pc')

    # Store the location into the scratch space
    be.context.setConcreteMemoryValue(MemoryAccess(be.system_state['scratch'], be.pointer_size), loc)

    # Load the location into return register, then load the value at that location
    inst1 = Instruction(pc_orig, ASSEMBLED_CODE_DICT[arch]['mov_scratch_ret'])
    be.context.processing(inst1)
    logging.debug(inst1)
    inst2 = Instruction(pc_orig, ASSEMBLED_CODE_DICT[arch]['mov_ret_ret'])
    be.context.processing(inst2)
    logging.debug(inst2)

    # Reset the program counter
    be.set_reg_val('pc', pc_orig)


def syscall_open(be, args):
    """Opens a file descriptor"""
    raise NotImplementedError


def syscall_read(be, args):
    raise NotImplementedError


def syscall_write(be, args):
    """Writes to a file descriptor
    
    ssize_t write(int fd, const void *buf, size_t count);

    writes up to count bytes from the buffer starting at buf to the file referred to by the file descriptor fd.
    On success, the number of bytes written is returned.  On error, -1 is returned, and errno is set to indicate the error.

    For now, I wont worry about any errors other than the whole "file is not open, or not open for writing" one. Might
    add more later (like segfault), but who knows

    https://man7.org/linux/man-pages/man2/write.2.html
    """
    # Check that the file descriptor is recognized and open for writing
    file = get_file_from_descriptor(be)
    if file is None or not file.is_writing():
        return _error(LinuxErrors.EBADF)

    # Write the data to file, and return the number of bytes written
    _success(file.write(read_bytes(be, args[0], count=args[1])))


def syscall_fstat(be, args):
    """Gives information on an open file descriptor"""
    print(args)
    raise NotImplementedError


def syscall_getuid(be, args):
    """Returns the uid (the actual uid of the caller)"""
    if 'uid_pointer' not in be.system_state:
        _set_symbolic_system_state_val(be, 'uid', 'uid_pointer', default=0)
    _read_mem(be, be.system_state['uid_pointer'])


def syscall_geteuid(be, args):
    """Returns the euid (the effective uid of the caller)"""
    if 'euid_pointer' not in be.system_state:
        _set_symbolic_system_state_val(be, 'euid', 'euid_pointer', default=0)
    _read_mem(be, be.system_state['euid_pointer'])


def syscall_getgid(be, args):
    """Returns the gid (the actual gid of the caller)"""
    if 'gid_pointer' not in be.system_state:
        _set_symbolic_system_state_val(be, 'gid', 'gid_pointer', default=0)
    _read_mem(be, be.system_state['gid_pointer'])


def syscall_getegid(be, args):
    """Returns the egid (the effective gid of the caller)"""
    if 'egid_pointer' not in be.system_state:
        _set_symbolic_system_state_val(be, 'egid', 'egid_pointer', default=0)
    _read_mem(be, be.system_state['egid_pointer'])
