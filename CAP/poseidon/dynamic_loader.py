import os
import logging
import lief
from .binary_parser import RelocationType, get_binary_info, SymbolType, get_symbol_type
from .utils import load_segments, load_binary_cfg
from .preassembled_code import ASSEMBLED_CODE_DICT
from .manual_functions import *
from triton import MemoryAccess


# These are functions used for manual relocation setup
def _MR_define_manual_function(func):
    """Takes a function, and returns a _mr_* like function that will create a fake function that handles things manually
    
    For example, this can be used with the manual function mf_tls_get_addr, and will return a _mr_* like function that:

        * takes as input the binary executor
        * allocates a pointer's worth of memory space for this 'fake' function (so that nothing else in the binary
          would try to use this memory location for something else, and so that the program counter will change to this
          address, and that can alert us that we are in a manual function)
        * adds that memory address along with the associated function to the binary executor's .manual_functions dict
    """

    def _mr_ret(be):
        mem_loc = be.memory_manager.malloc(be.pointer_size, alignment=1)
        be.manual_functions[mem_loc] = func
        return mem_loc

    return _mr_ret


def _MR_define_indirect_function(name, ifunc_addr, address, offset):
    """Makes a gnu_ifunc resolver that will be run at runtime as needed
    
    Args:
        name (str): the symbol name of the indirect function, or None if it is an IRELATIVE relocation
        ifunc_addr (int): the VIRTUAL memory location of the resolver function
        address (Union[int, None]): the VIRTUAL address to store the resolved function at (if it is an IRELATIVE relocation),
            or None if it is a GNU_IFUNC symbol
        offset (int): the virtual address offset for these values
    """

    def _mr_ret(be):
        mem_loc = be.memory_manager.malloc(be.pointer_size, alignment=1)
        be.manual_functions[mem_loc] = MF_indirect_func_resolver(name, ifunc_addr, address, offset)
        return mem_loc
    
    return _mr_ret


def _mr_rtld_global(be):
    """Sets up the _rtld_global symbol
    
    This contains global variables for the runtime loader. You can find the struct information at:
    https://github.com/bminor/glibc/blob/master/sysdeps/generic/ldsodefs.h

    And the initialization of the struct at: https://github.com/bminor/glibc/blob/master/elf/rtld.c

    This struct has quite a lot of things in it, so I'm just going to initialize it to all 0's and hope it doesn't
    break anything for the time being.

    I ran a basic binary with gcc-7.5.0 on ubuntu 18.04 and did "print sizeof(rtld_global)" which returned 3960 bytes.
    So, I will make the _rtld_global struct take up 4096 bytes in case later versions increase the size, and align it
    to 64 bytes.
    """
    # symbolize me
    if '_rtld_global' not in be.system_state:
        be.system_state['_rtld_global'] = be.memory_manager.calloc(4096, alignment=64)
    return be.system_state['_rtld_global']


def _mr_rtld_global_ro(be):
    """Sets up the _rtld_global_ro symbol
    
    I'm just assuming it is the same as _rtld_global for now.
    """
    # symbolize me
    if '_rtld_global_ro' not in be.system_state:
        be.system_state['_rtld_global_ro'] = be.memory_manager.calloc(4096, alignment=64)
    return be.system_state['_rtld_global_ro']


def _mr_libc_enable_secure(be):
    """Determines the __libc_enable_secure value in a way that ~shouldn't~ break the symbolic engine
    
    Computation of this value is found here: https://github.com/lattera/glibc/blob/master/elf/enbl-secure.c

    Essentially, this just checks if the program was run with different permissions than the user/group that called it.

    This should allow us to compute the value at les_loc in a symbolic-friendly way (based on user/group ids). We don't
    need to do the actual storing of the les_loc in a symbolic way as we don't care what the memory address actually
    is, just what the value at it would be
    """
    logging.info("Computing value for __libc_enable_secure...")

    if be.binary_info['os'] != 'linux':
        raise ValueError("Can only determine __libc_enable_secure on linux, not %s" % repr(be.binary_info['os']))
    
    ll_size = 4 if be.binary_info['mode'] == lief.MODES.M32 else 8

    # Allocate a long long size value for __libc_enable_secure to reside
    les_loc = be.memory_manager.malloc(ll_size)

    # Store the memory location of les_loc into start of scratch space
    be.context.setConcreteMemoryValue(MemoryAccess(be.system_state['scratch'], ll_size), les_loc)

    # Save the program counter
    pc_orig = be.get_reg_val('pc')

    # Execute the code needed to compute and store the __libc_enable_secure value
    be.emulate(ASSEMBLED_CODE_DICT[be.binary_info['arch']]['libc_enable_secure'])

    # Reset the program counter and return the location of __libc_enable_secure
    be.set_reg_val('pc', pc_orig)
    return les_loc


def _mr_dl_argc(be):
    """Returns the argc value from the binary executor"""
    return be.system_state['argc']


def _mr_dl_argv(be):
    """Returns the argv value from the binary_executor"""
    return be.system_state['argv']


# Dictionary of manual relocations. If a relocation's symbol does not exist, this dictionary will tell you what to
#   put there instead. Will default to the binary's relocation if it exists before using this dictionary value
# Values can either be integers which will be directly used, or functions which should be called with one argument
#   (the executor), and that will return an integer to use for the relocation. These functions should be able to detect
#   whether or not they have already been called/initialized and set up thing accordingly
MANUAL_RELOCATIONS = {

    # These functions do libpthread cleanup stuff. Looking at the code, it seems. We can just set it to NULL 
    # to avoid calling it: https://github.com/lattera/glibc/blob/master/malloc/set-freeres.c
    # JK, I now set it to a manual function that just prints that it was called so I know if they were called
    '__libpthread_freeres': _MR_define_manual_function(mf_libpthread_freeres),
    '__libdl_freeres': _MR_define_manual_function(mf_libdl_freeres),

    # Various information about the runtime loader
    '_rtld_global': _mr_rtld_global,
    '_rtld_global_ro': _mr_rtld_global_ro,

    # Checks if we need to be in secure mode (if this program was run with permissions differing that of the user that
    #   ran the program)
    '__libc_enable_secure': _mr_libc_enable_secure,

    # I'm not entirely sure if this should be the value itself, or a pointer to the value. I'm assuming it's just the
    #   value for now, but I doubt it's used anyways.
    # Looks like this is, as the name suggests, just to tell other code if the dynamic load is in its 'starting up'
    #   phase, which for us, will never be true since we are our own dynamic loader
    '_dl_starting_up': 0,

    # These should be the same values that will be pushed to the stack
    '_dl_argc': _mr_dl_argc,
    '_dl_argv': _mr_dl_argv,

    # Function implemented by rtld to get addresses of dynamically loaded libraries
    '__tls_get_addr': _MR_define_manual_function(mf_tls_get_addr),

    # Function that creates exceptions in the dynamic loader
    '_dl_exception_create': _MR_define_manual_function(mf_dl_exception_create),

    # Function that gets tunable dynamic loader values
    '__tunable_get_val': _MR_define_manual_function(mf_tunable_get_val),

    # From what I can tell, this is used to find dynamic libraries based on some object of info about them
    '_dl_find_dso_for_object': _MR_define_manual_function(mf_dl_find_dso_for_object),

    # These functions have something to do with C/C++ GNU Transactions: https://gcc.gnu.org/onlinedocs/libitm/index.html, 
    # Stackoverflow post about them: https://stackoverflow.com/questions/41274482/why-does-register-tm-clones-and-deregister-tm-clones-reference-an-address-past-t
    # It seems these can just do nothing and it should be fine. There ~might~ be a problem if we are using multi-threaded
    #   programs, but since we only emulate one thread at a time, i think there won't be any problems since these seem
    #   to just be used to stop multiple threads clobbering the same value at the same time
    '_ITM_deregisterTMCloneTable': _MR_define_manual_function(mf_ITM_deregisterTMCloneTable), 
    '_ITM_registerTMCloneTable': _MR_define_manual_function(mf_ITM_registerTMCloneTable), 

    # This is a function used for initializing things for gprof profiling: https://stackoverflow.com/questions/12697081/what-is-the-gmon-start-symbol
    # We can simply do nothing for now
    '__gmon_start__': _MR_define_manual_function(mf_gmon_start),
}


class DynamicLoader:
    """The dynamic loader for an executing binary"""

    def __init__(self, executor):
        self.executor = executor
        self.symbols = {}  # Dictionary mapping symbol names to (value, binary_name) tuples
        self.loaded_libs = {}  # Dictionary mapping loaded library names to their binary_info (which contains a key 'offset')
        self.tls_relocations = []  # List of thread-local storage relocations. These are performed after loading the 
                                   # binary and whenever the process creates new threads
        
        # Insert the main binary's name into loaded_libs
        self.loaded_libs[self.executor.binary_info['name']] = self.executor.binary_info

        # Parse all of the possible dynamic libraries for this binary type
        self.parsed_libs = _parse_dyn_libs(self.executor.binary_info['ld_path'])

    def load_binary(self, binary_info=None, virt_addr_start=None, dynlib_cfg_info=None):
        """Recursively loads all dynamic libraries needed for the given binary, and updates all relocations

        See the RelocationType docs for more info on how different relocation types are handled
        
        Args:
            binary_info (Optional[Dict]): dictionary of binary info for the binary to load. If None, will load the
                main executable in self.executor
            virt_addr_start (Optional[int]): the starting virtual address of the given binary. If None, will use
                binary_info['offset']
            dynlib_cfg_info (None): used for recursion, do not modify
        """
        binary_info = self.executor.binary_info if binary_info is None else binary_info
        virt_addr_start = binary_info['offset'] if virt_addr_start is None else virt_addr_start

        # This will be a dictionary mapping binary names to lists of their 4-tuple relocation information:
        #   (binary_name_needing_relocation, symbol_name, value, binary_name_containing_symbol)
        init_call = dynlib_cfg_info is None
        dynlib_cfg_info = {}

        for rel_dict in binary_info['relocations']:
            
            # Handle the relative offsets
            if rel_dict['type'] == RelocationType.RELATIVE:
                val = virt_addr_start
            
            # Create resolver functions for IRELATIVE relocations
            elif rel_dict['type'] in [RelocationType.IRELATIVE]:
                logging.info("Adding irelative indirect function resolver at location: 0x%x, points to a function that starts at: 0x%x" 
                             % (rel_dict['address'] + virt_addr_start, rel_dict['addend'] + virt_addr_start))
                val = _MR_define_indirect_function(None, rel_dict['addend'], rel_dict['address'], virt_addr_start)(self.executor)
               
                # Set the addend to 0
                rel_dict['addend'] = 0
            
            # Handle the immediate value stores
            elif rel_dict['type'] in [RelocationType.JUMP_SLOT, RelocationType.GLOB_DAT, RelocationType.R64]:
                
                # If the symbol already exists, no need to reload it
                if rel_dict['name'] in self.symbols:
                    pass
                    #logging.debug("Name has already been loaded!")

                # If the symbol doesn't exist in our currently loaded values, search for it
                elif rel_dict['name'] not in self.symbols:

                    # Search for the name in one of those loaded libraries
                    for dynlib in self.parsed_libs:
                        if rel_dict['name'] in [s.name for s in dynlib.exported_symbols]:
                            break
                    
                    # If we didn't break, then we never found the name in any library
                    else:
                        dynlib = None
                    
                    # We found the symbol in a dynlib, so we can load it in
                    if dynlib is not None:
                        # Now we can load that dynamic library since it has a name we needed, and add its function names
                        #   to the plt_symbols
                        dynlib_info = get_binary_info(dynlib, binary_info['os_version'])
                        logging.info("Found name in dynlib: %s, loading..." % dynlib_info['name'])
                        offset = load_segments(dynlib_info['segments'], self.executor.context, self.executor.memory_manager, force_addr=False)

                        # Keep track of the new offset, as well as new libraries that need to be added to the CFG (we
                        #   just check if the key exists)
                        dynlib_info.update({'offset': offset, 'new_lib': None})
                        
                        # Add the dynlib info to the list of loaded dynamic libraries
                        self.loaded_libs[dynlib_info['name']] = dynlib_info

                        # Compute the locations of all exported functions in this loaded library
                        new_symbols = list(dynlib.exported_symbols)
                        for s in new_symbols:
                            sym_type = get_symbol_type(s.type)
                            if sym_type in [SymbolType.OBJECT, SymbolType.FUNC]:
                                s_value = s.value
                            elif sym_type in [SymbolType.GNU_IFUNC]:
                                logging.info("Adding gnu_ifunc resolver for symbol %s" % repr(s.name))
                                s_value = _MR_define_indirect_function(s.name, s.value, None, dynlib_info['offset'])(self.executor)
                            else:
                                raise NotImplementedError(sym_type.name)
                            self.symbols[s.name] = (s_value + offset, dynlib_info['name'])
                        logging.info("Loaded %d new symbols" % len(new_symbols))

                        # Load anything that dynamic library needs
                        self.load_binary(dynlib_info, virt_addr_start=offset, dynlib_cfg_info=dynlib_cfg_info)
                    
                    # Check if the value is a manual relocation. If so, add it to our symbols
                    elif rel_dict['name'] in MANUAL_RELOCATIONS:
                        _man_val = MANUAL_RELOCATIONS[rel_dict['name']]
                        _man_val = _man_val if isinstance(_man_val, int) else _man_val(self.executor)
                        logging.info("Defaulted to manual relocation for symbol name '%s', value: %s" 
                                     % (rel_dict['name'], _man_val))
                        self.symbols[rel_dict['name']] = (_man_val, '__MANUAL__')
                    
                    # Otherwise we do not know the name, raise an error
                    else:
                        raise ValueError("Could not find external symbol info %s for binary: %s" % (rel_dict, binary_info['name']))

                # Get the function address as the value
                val, bn = self.symbols[rel_dict['name']]

                # If this relocation was for a function, add it to the list of cfg info
                if rel_dict['type'] in [RelocationType.JUMP_SLOT]:
                    for n in [binary_info['name'], bn]:
                        dynlib_cfg_info.setdefault(n, [])
                    
                    dynlib_cfg_info[binary_info['name']].append((binary_info['name'], rel_dict['name'], val, bn))

            # TLS offsets will be handled later by individual threads
            elif rel_dict['type'] in [RelocationType.TPOFF]:
                #logging.debug("Relocation is a TLS type. Passing for now, will be loaded later")
                rel_dict.update({'binary_name': binary_info['name'], 'virt_addr_start': virt_addr_start})
                self.tls_relocations.append(rel_dict)
                continue

            else:
                raise NotImplementedError(str(rel_dict['type']))
            
            # Add in the addend, and store in the given memory address (plus offset of start of virtual address of binary)
            #logging.debug("Linking %s value %d (+ %d addend) to 0x%016x (%d bytes)"
            #    % (extra_str, val, rel_dict['addend'], rel_dict['address'] + virt_addr_start, rel_dict['size']))
            self.executor.context.setConcreteMemoryValue(MemoryAccess(rel_dict['address'] + virt_addr_start, rel_dict['size']),
                                                         val + rel_dict['addend'])
        
        # Finally, if this is the main call (not a recursive one), add in all of the new libraries to the cfg if needed
        if init_call and hasattr(self.executor, 'cfg') and self.executor.cfg is not None:
            library_order = [k for k in dynlib_cfg_info.keys() if 'new_lib' in self.loaded_libs[k]]  # Determine a best order of these, and keep track of which ones have already been loaded
            for bn in library_order:
                function_mapping = {}  # Figure out the function mapping
                logging.info("Loading and inserting dynamic library cfg for binary: %s\n\tWith function_mapping: %s" %
                             (repr(bn), function_mapping))
                self.executor.cfg.insert_library(load_binary_cfg(self.loaded_libs[bn]), function_mapping, 
                                                 offset=self.loaded_libs[bn]['offset'])
                
                # Update the fact that this library has been inserted into the CFG
                del self.loaded_libs[bn]['new_lib']


def _parse_dyn_libs(path):
    """Uses lief to parse all dynamic libraries in the given path, and returns a list of those lief objects"""
    files = [f for f in os.listdir(path) if f.endswith(('.so', '.dll'))]
    logging.info("Found %d possible dynamic libraries: %s" % (len(files), files))
    return [lief.parse(os.path.join(path, f)) for f in files]
