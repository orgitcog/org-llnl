import logging
import lief
from .binary_parser import RelocationType
from .utils import twos_complement
from triton import ARCH, MemoryAccess


TLS_SUPPORTED_ARCHS = [ARCH.X86_64]
TCB_SIZE = 8  # The number of bytes used in the thread control block. Need 8 for a full 8-byte pointer if using


class TritonThread:
    """A single thread in a running triton instance

    Thread-Local Storage Information:
    We are currently following the SystemV ABI for ELF TLS: https://akkadia.org/drepper/tls.pdf
    Another helpful website: https://maskray.me/blog/2021-02-14-all-about-thread-local-storage

    The thread-local storage contains data local to each thread. This is useful for things like errno so if one thread
    has a system error, it doesn't necessarily stop all other threads, nor clobber the errno in other threads in case
    they had different errors.

    Key Terms:

        - TLS: thread-local storage
        - TCB: thread control block. Contains system information for the threads, but not associated with TLS. For us,
          since we are emulating and all information will be stored in this python object instead of in the triton
          context's memory, we can leave this essentially empty
        - tp: the thread pointer. Points to the memory location of the 'start' of the TCB

    The ELF TLS SystemV ABI documentation lists multiple variants for how to go about doing TLS. For now, we are
    focusing on x86_64 linux, and thus we will be using:

        - Variant II of the TLS data structure (See page 6 of the SystemV ABI TLS supplemental documentation).
          This data structure contains information for the TLS of dynamically loaded libraries. However, we don't
          actually need to fully implement it. The dynamic thread vector dtv isn't allowed to be accessed by any code
          emitted by the compiler. It exists solely so the system can keep track of where things are (EG: with calls
          to __tls_get_addr). So, we really only need to implement a fake TCB and put all of the intitally loaded
          libraries TLS data right before it, then keep track of where all those libraries start/stop in python instead
          of in the emulated Triton context.

          * Technically, for x86_64, accessing the dtv is implementation-defined. So while the compiler shouldn't emit
            assembly that directly accesses the dtv, it's possible someone could write code that does do that. I'm going
            to assume this won't happen for now to make things easier, but I might have to change it later

        - the fs segment register to contain the thread pointer tp. Traditionally, fs:0 contains the thread pointer
          (IE: the memory location stored in the fs register contains the memory location of the thread pointer).
          However, the x86_64 specifications require that the fs register also contains the thread pointer (to make
          tp lookups quicker). This means that fs contains some memory address, and fs:0 contains the memory address
          of itself.


    On creation of a new thread, the following steps are taken:

        1. The TLS static blocks as well as the TCB are initialized.
           This will search through all of the initial dynamically loaded libraries as well as the main executable's
           binary_info to find all of the tdata and tbss blocks for each. The main executable's will be placed directly
           before the TCB, and all others will be placed arbitrarily before that, with correct alignment/sizes. The
           TCB consists of one element currently: the memory address of itself (see above). Sections will be placed in
           the order .tdata, .tbss, with correct alignment on each (32 bytes). Each section will be 'shifted backwards'
           in order to have correct alignment, with the next TLS block being place immediately previous (again, continuing
           to shift backwards in order to have correct alignment)
        2. The thread pointer is set to the end of the main executable's TLS block + alignment padding. The value at this
           address is also set to the address itself
        3. Relocations are performed based on the thread pointer's offsets
    
    Parameters
    ----------
    executor: `TritonExecutor`
        the TritonExecutor this thread is in
    """

    executor = None
    """The TritonExecutor this thread is in"""

    state = None
    """Dictionary containing state information of this thread. 
    
    Will only be set and up-to-date when this thread stops running, and will be used to continue running the thread
    when the task-scheduler determines it is time. See TritonThread.get_current_state for an overview of keys/values
    """

    def __init__(self, executor):
        self.executor = executor
        self.thread_idx = self.executor._get_next_thread_idx()

        logging.info("Initializing thread idx: %d" % self.thread_idx)

        self.tls_end = None
        self.tp = None
        self.tp_offsets = {}  # Dictionary mapping dynamic library names to their associated offset from the tp (to start
                              #   of the .tdata section)
        self._init_TLS()

        # Set up the state
        self.state = self.get_current_state()
    
    def _init_TLS(self):
        """Initializes the Thread-local storage"""
        logging.info("Initializing TLS for thread index: %d" % self.thread_idx)

        # Check that we are using a supported architecture
        if self.executor.binary_info['arch'] not in TLS_SUPPORTED_ARCHS:
            raise ValueError("Unsupported TLS architecture %s, supported architectures: %s" 
                             % (repr(self.executor.binary_info['arch'].name), TLS_SUPPORTED_ARCHS))

        # Initialize the thread-local storage for this thread for the binary, and all loaded dynamic libraries.
        # It should be copied from the .tdata and .tbss sections
        # Doing this in reverse as the TLS data structure is in reverse for x86_64
        sections = []  # list of sections in order
        binary_names = []
        for binary_info in ([self.executor.binary_info] + list(self.executor.dynamic_loader.loaded_libs.values())):
            logging.info("Checking for .tdata and .tbss sections for binary: %s" % binary_info['name'])

            tdata, tbss = None, None
            for segment in binary_info['binary'].segments:
                for section in segment.sections:

                    if section.name in ['.tdata']:
                        if tdata is not None:
                            if tdata.virtual_address != section.virtual_address or tdata.size != section.size:
                                raise ValueError("Found multiple .tdata sections with the same name, but different addresses/sizes!")
                        else:
                            tdata = section

                    elif section.name in ['.tbss']:
                        if tbss is not None:
                            if tbss.virtual_address != section.virtual_address or tbss.size != section.size:
                                raise ValueError("Found multiple .tbss sections with the same name, but different addresses/sizes!")
                        else:
                            tbss = section
            
            if tdata is None and tbss is None:
                logging.info("Could not find .tdata nor .tbss sections, continuing...")
                continue
            if tdata is None or tbss is None:
                raise ValueError("Only found one of .tdata and .tbss sections, need both!")
            
            sections += [tdata, tbss]
            binary_names += [binary_info['name'], binary_info['name']]
        
        # Compute the total size needed. We assume we need the full alignment for each, wasting a bit of
        #   memory, but who cares, we have a ton of it. Also add on the TCB_SIZE of bytes
        size = sum(s.size for s in sections) + sum(s.alignment for s in sections) + TCB_SIZE + 32

        # Allocate (and clear) the TLS blocks and TCB. We don't need alignment since we assumed the max bytes for 
        #   alignment previously, and we don't care if the 'end' (but technically start since we're working in reverse) 
        #   of this is aligned
        # Set the thread pointer to the start of the TCB (aligned to 16 bytes), and set the first element at the TCB to 
        #   be a memory pointer to itself
        self.tls_end = self.executor.memory_manager.calloc(size)
        self.tp = (self.tls_end + size - TCB_SIZE) & 0xFFFF_FFFF_FFFF_FFF0
        logging.info("Thread pointer created for thread %d at %d (0x%x)" % (self.thread_idx, self.tp, self.tp))
        self.executor.context.setConcreteMemoryValue(MemoryAccess(self.tp, 8 if self.executor.binary_info['mode'] == lief.MODES.M64 else 4), self.tp)

        # Loop through the sections putting their data in the correct spot with alignment
        curr_tls_addr = self.tp
        for section, bn in zip(sections, binary_names):
            curr_tls_addr -= section.size
            curr_tls_addr &= ~(section.alignment - 1)
            logging.info("Placing TLS section \"%s-%s\" of size %d (alignment %d) at memory address %d (0x%x)" 
                         % (bn, section.name, section.alignment, section.size, curr_tls_addr, curr_tls_addr))
            self.executor.context.setConcreteMemoryAreaValue(curr_tls_addr, section.content.tobytes())
            self.tp_offsets['%s-%s' % (bn, section.name)] = curr_tls_addr - self.tp  # Get the negative offset
        
        # Perform all of the relocations needed
        for rel_dict in self.executor.dynamic_loader.tls_relocations:

            if rel_dict['type'] in [RelocationType.TPOFF]:
                extra_str = 'TP_OFFSET - "' + rel_dict['binary_name'] + '-.tdata"'
                val = self.tp_offsets[rel_dict['binary_name'] + '-.tdata']
            
            else:
                raise NotImplementedError(rel_dict['type'])

            # Add in the addend, and store in the given memory address (plus virtual address start of that relocation)
            logging.info("Linking %s value %d (+ %d addend) to 0x%016x (%d bytes)"
                % (extra_str, val, rel_dict['addend'], rel_dict['address'], rel_dict['size']))
            self.executor.context.setConcreteMemoryValue(MemoryAccess(rel_dict['address'] + rel_dict['virt_addr_start'], rel_dict['size']), 
                                                         twos_complement(val + rel_dict['addend'], rel_dict['size']))
    
    def get_current_state(self):
        """Returns a dictionary of the current state
        
        Keys/values:
        
            - 'registers': dictionary of register values. Keys are lowercase names of registers (same names as attributes
              in self.executor.context.registers), and values are the register values.
        """
        return {'registers': {'fs': self.tp}}
