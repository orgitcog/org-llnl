"""These are functions that will be handled manually instead of emulating them"""

import logging
import inspect
from triton import MemoryAccess


def MF_indirect_func_resolver(name, ifunc_addr, address, offset):
    """This returns a function that can act as the indirect function resolver
    
    Args:
        name (Union[str, None]): the symbol name of the indirect function, or None if it is an IRELATIVE relocation
        ifunc_addr (int): the VIRTUAL memory location of the resolver function
        address (Union[int, None]): the VIRTUAL address to store the resolved function at (if it is an IRELATIVE relocation),
            or None if it is a GNU_IFUNC symbol
        offset (int): the virtual address offset for these values
    """

    def _mf_ret(be):
        if address is None:
            raise NotImplementedError("gnu_ifunc not implemented: %s" % repr(name))
        
        # Add in the offset to the ifunc_addr virtual address
        nonlocal ifunc_addr
        ifunc_addr += offset
        
        # Check if this function has already been resolved
        if ifunc_addr in be.resolved_funcs:
            logging.info("Indirect function at address 0x%x has already been resolved" % ifunc_addr)
            resolved_func = be.resolved_funcs[ifunc_addr]
        
        else:
            logging.info("Beginning IRELATIVE resolution. Resolver function address: 0x%x" % ifunc_addr)

            # Save the current registers so we can call the IRELATIVE resolver, and the program counter as it contains the
            #   address of this function
            ps = be.get_registers()
            pc = ps[be.get_reg_name('pc')]

            # Set the return address. Use a dummy one to let us know when to return
            ret_addr = 2 ** (be.pointer_size * 8) - 1
            be.push_stack(ret_addr)
            logging.info("Pushed return address 0x%x to stack. peek: 0x%x" % (ret_addr, be.peek_stack('pointer')))

            # Begin emulation from the ifunc_addr, and stop emulation upon reaching the ret_addr
            be.emulate(ifunc_addr, trace_path=False, stop_addr=ret_addr)
            resolved_func = be.get_reg_val('ret') + offset

            # Restore the registers
            be.set_registers(ps)

            # Delete the dummy function for the resolvers
            del be.manual_functions[pc]
            be.memory_manager.free(pc)

            # Set the resolved func in the binary executor for future calls
            be.resolved_funcs[ifunc_addr] = resolved_func

        # Set the instruction pointer to the resolved function to run
        be.set_reg_val('pc', resolved_func)

        # Set the memory location for this relocation to the resolved function for future calls
        be.set_mem(address + offset, resolved_func)

        logging.info("Setting ifunc located at 0x%x to resolved function value: 0x%x" % (address + offset, resolved_func))
    
    return _mf_ret


def mf_tls_get_addr(be):
    """Implementation of the __tls_get_addr rtld function"""
    raise NotImplementedError


def mf_dl_exception_create(be):
    """Creates an exception in the dynamic loader
    
    Code can be found here: https://codebrowser.dev/glibc/glibc/elf/dl-exception.c.html

    Function prototype: void _dl_exception_create (struct dl_exception *exception, const char *objname, const char *errstring)
    """
    raise NotImplementedError


def mf_tunable_get_val(be):
    """Gets dynamic loader tunable values
    
    Info on Tunables: https://www.gnu.org/software/libc/manual/html_node/Tunables.html
    """
    raise NotImplementedError


def mf_dl_find_dso_for_object(be):
    """From what I can tell, this is used to find dynamic libraries based on some object of info about them
    
    Function located here: https://codebrowser.dev/glibc/glibc/elf/dl-open.c.html
    """
    raise NotImplementedError


def mf_libpthread_freeres(be):
    """Exists just to warn of its call"""
    logging.warning("Binary called intentially unimplemented function '%s'" % inspect.stack()[0][3])


def mf_libdl_freeres(be):
    """Exists just to warn of its call"""
    logging.warning("Binary called intentially unimplemented function '%s'" % inspect.stack()[0][3])


def mf_ITM_deregisterTMCloneTable(be):
    """Exists just to warn of its call"""
    logging.warning("Binary called intentially unimplemented function '%s'" % inspect.stack()[0][3])


def mf_ITM_registerTMCloneTable(be):
    """Exists just to warn of its call"""
    logging.warning("Binary called intentially unimplemented function '%s'" % inspect.stack()[0][3])


def mf_gmon_start(be):
    """Exists just to warn of its call"""
    logging.warning("Binary called intentially unimplemented function '%s'" % inspect.stack()[0][3])

