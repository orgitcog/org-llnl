"""
Objects that allow for execution(s) of binaries using Triton
"""

import logging
import traceback
import lief
import re
from triton import TritonContext, Instruction, OPCODE, CPUSIZE, MemoryAccess, ARCH
from .binary_parser import get_binary_info
from .memory_manager import MemoryManager
from .system import handle_syscall, error_raised
from .utils import random_strings, load_segments, load_binary_cfg, get_float_bits, get_float_from_bits
from .dynamic_loader import DynamicLoader
from .triton_thread import TritonThread


# The starting address for the stack (grows upwards, towards 0). This is for 32-bit, but the 64-bit version adds on
#   all F's to the start
STACK_START_ADDRESS = 0xFFFF_FE00  

# The starting address of scratch space, used for system calls and remains un-symbolized. Everything from this start 
#   upwards is reserved for scratch. This is for 32-bit, but the 64-bit version adds on all F's to the start
# Currently has 256 bytes allocated to it
SCRATCH_SPACE_START_ADDRESS = 0xFFFF_FF00

# The default max size for input strings
DEFAULT_STRING_MAX_SIZE = 100

# The default max number of arguments/strings
DEFAULT_MAX_NUM_STRINGS = 5


class BinaryExecutor:
    """Class that allows for one or more executions of a binary, allowing for various command line inputs
    
    Parameters
    ----------
        bin_obj `Union[str, Iterable[int], TextIO]`
            the binary object to load. Can be:

                - str: a string path to the binary to load
                - Iterable[int]: an iterable of integers (including bytes() objects) of the bytes in the binary file
                - TextIO: an open file object to load from. Must be an object with a callable .read() attribute that returns
                    an iterable of integers of the bytes in the binary file

        cfg `Union[str, bincfg.CFG]`
            the CFG corresponding to the binary, or a string path to one to load
        os_version `str`
            the os version to use. Currently available versions:

            - 'ubuntu_22.10'
    """

    context = None
    """The Triton context for the current binary"""

    memory_manager = None
    """The MemoryManager object for the current binary"""

    binary_info = None
    """The dictionary of binary info"""

    cfg = None
    """The CFG object associated with this running binary"""

    threads = []
    """A list of running thread informations. Each element is a TritonThread() object"""

    running_thread = None
    """The currently running thread"""

    system_state = {}
    """Dictionary of current system state information
    
    Only contains information that is 'external' to the binary.

    Contains:

        - 
    """

    def __init__(self, binary, os_version):
        self.binary_info = get_binary_info(binary, os_version)
        self._finished_emulation = False
        logging.info("Created TritonExecutor with binary_info: %s" % _printable_binary_info(self.binary_info))
    
    def execute(self, system_inputs=None, entrypoint=None, trace_path=False, debug=False):
        """Executes the binary from the given starting address, keeping track of the path of execution

        Execution looks (roughly) like this:

            1. Create a new TritonContext (initializes all memory and registers to 0), and a new MemoryManager (empty).
               The TritonContext is what handles the main emulation and concolic execution, and the MemoryManager is a
               class I wrote that handles allocating memory without clobbering already existing memory.
               This requires information from the binary itself (specifically, the architecture), but that should already
               exist as it is parsed when this BinaryExecutor is initialized.
               We create a new TritonContext here instead of reusing already created ones (on multiple calls) since
               many memory locations need to be initialized to all 0's, and creating a new one does that automatically.
            2. Load the binary's associated CFG, if using trace_path
            3. Load the binary's loadable segments into memory.
               We iterate through all of the segments in the binary (should have already been parsed), and load them
               into their associated virtual address, while reserving their 'size' in bytes. Their size can be larger
               than the actual content of that segment, but never smaller, and needs to be reserved anyways and
               initialized to all 0's. This is done inherently when creating a new TritonContext.
            4. Load dynamic libraries and create relocations.
               This involves iterating through all of the relocations in the main binary, loading any needed dynamic
               libraries, and inserting relocation values at their associated locations. Any loaded dynamic libraries
               will recursively have their relocations dealt with as well. These relocations include needed functions
               or external variables, offsets from start of virtual address, offsets from thread-local storage, etc.
            5. Create a new TritonThread for the main thread.
               This object contains information about the currently running thread, as well as thread-local storage
               information (see below). This will handle copying the tbss and tdata sections, as well as any TLS related
               relocations that the dynamic loader postponed

               Thread Local Storage:

               Thread local storage allows threads to have memory for variable associated only with that thread. This is
               useful for things like ERRNO in case only one thread errors out, and you want to keep running, or other
               threads have errors that are different from one another without ERRNO being clobbered.

            6. Initialize the stack, argc, argv, and envp values.
               The stack is created at the end of memory and grows upwards. The argc/argv/envp are pushed onto the stack
               before execution is passed to the entrypoint.
            7. Initialize the system-level information. These are things that are unchanging throughout the life of the
               program that are determined by the system (EG: group_id or user_id of what executed the process)
            8. Find the entrypoint in the binary and set the instruction pointer (rip) to it
            9. If using trace_path, initialize a list of basic blocks reached based on the associated CFG, and put the 
               entrypoint block into it
            10. Execute the binary. Do a loop:

               a. Get the instruction from memory pointed to by the instruction pointer. To do this, we read 16 bytes
                  starting from the memory location in rip. Triton will process the instruction and only use up/increment
                  rip based on how many bytes are needed for the instruction.
               b. Process the instruction in Triton
               c. If the instruction is a system call, handle that ourselves (see the 'system' module for more info)
               d. If there was a syscall error, or if the instruction is a hlt, then halt the program
               e. If we are doing debug things, accept input from the user for debug info, until the user says next
               f. Load the next instruction from memory based on rip
               g. If the next block we are going to is a memory address that is not in the memory addresses of the block
                  we were just in, then add in the new block we are going to into the trace
        
        Args:
            system_inputs (Optional[Dict]): a dictionary defining system inputs, or None to use default. Can contain the keys:

                - 'cmd_args': the command line args. Should be a list of either strings, or null-terminated bytes. If 
                   None, or if the key doesn't exist, then it will default to between 0 and DEFAULT_MAX_ARGC (uniform 
                   random) strings randomly initialized (uniform random length between 0 and 'string_max_size' - 1
                   of uniform random bytes in range [1, 255], then null terminated).
                   These strings will have a null terminator added to them.
                - 'environ': the environ variables. A list of strings. Defaults to same random construction as cmd_args
                - 'string_max_size': the size to make all strings. Any strings smaller than this length will be padded with
                  null bytes until they reach this size. This is used for symbolic execution to allow for input strings
                  of larger sizes than the given inputs. If this is not passed or is None, then DEFAULT_MAX_STRING_SIZE
                  is used.

                - 'uid': the user id (https://en.wikipedia.org/wiki/User_identifier) (0 would be superuser). Defaults to 0
                - 'euid': the effective user id. Defaults to uid
                - 'gid': the group id (https://en.wikipedia.org/wiki/Group_identifier) (0 would be superuser login group).
                  Defaults to 0.
                - 'egid': the effective group id. Defaults to gid

            entrypoint (Optional[int]): entrypoint address, or None to start at the default entrypoint. Should be the
                address of a basic block
            trace_path (bool): if True, will load in the CFG (expected to be same name as binary but ending in either
                '.pkl' for a pickle file or '.cfg'/'.txt' for a rose output file). This will store the loaded CFG in the
                '.cfg' attribute of this executor, and will also keep track of a trace (list of basic blocks) through
                the program during execution, and store it in the '.trace' attribute
            debug (bool): if True, turns on debug mode. Acts kinda like gdb where you can step through assembly lines
                one at a time. Expects input from stdin. Expressions are evaluated using python's eval() function
                with the locals/globals that would exist within this function. Special keywords:
                    
                    - '': the empty string (or all whitespace), run the last command
                    - 'next', 'step': step to the next instruction
                    - 'break', 'return', 'exit', 'quit': immediately quit this section of the emulation
                    - 'sysexit': signal keyboard interrupt in program
                    - 'reg_val("REGISTER")': return the value stored in the register
        """
        # Create the context, memory manager, and dynamic loader. Also initialize the system_state
        self.debug = debug
        self.context = TritonContext(self.binary_info['arch'])
        self.memory_manager = MemoryManager(self)
        self.dynamic_loader = DynamicLoader(self)
        self.system_state = {}  # Dictionary of current system state information. Used for things like manual relocations
        self.manual_functions = {}  # Dictionary of manual functions that should be handled by us. Maps memory addresses to python functions
        self.resolved_funcs = {}  # Dictionary mapping ifunc addresses to their resolved address values

        # Load in the CFG if needed
        self.cfg = load_binary_cfg(self.binary_info) if trace_path else None

        # Get some initial system state values
        self.system_state['stack_start'] = _make_mode_bits(STACK_START_ADDRESS, self.binary_info['mode'])
        self.system_state['scratch'] = _make_mode_bits(SCRATCH_SPACE_START_ADDRESS, self.binary_info['mode'])

        # Initialize the system inputs
        self.system_inputs = system_inputs if system_inputs is not None else {}
        self.system_inputs.setdefault('string_max_size', DEFAULT_STRING_MAX_SIZE)

        # Load the segments of this binary
        load_segments(self.binary_info['segments'], self.context, self.memory_manager, force_addr=True)
        self.binary_info['offset'] = 0

        # Initialize argv and envp (do this after loading in the segments of the main binary so that gets memory priority)
        self._init_args()

        # Load any dynamic libraries for the main executable
        self.dynamic_loader.load_binary()

        # Initialize the main thread
        self.threads.append(TritonThread(self))
        self.running_thread = self.threads[0]

        sysstate_info = {
            'stack_start': '0x%x' % self.system_state['stack_start'],
            'scratch': '0x%x' % self.system_state['scratch'],
        }
        logging.info("Binary loaded successfully! Using system_state info: %s" % sysstate_info)
        
        # Push argc, argv, and envp onto the stack
        logging.info("Initializing stack at starting location 0x%16x\n\twith %d args: %s\n\twith %d env args: %s"
            % (self.system_state['stack_start'], self.system_state['argc'], self.system_inputs['cmd_args'], 
               len(self.system_inputs['environ']), self.system_inputs['environ']))
        
        self.set_reg_val('bp', self.system_state['stack_start'])
        self.set_mem(self.system_state['stack_start'] - 24, self.system_state['argc'])
        self.set_mem(self.system_state['stack_start'] - 16, self.system_state['argv'])
        self.set_mem(self.system_state['stack_start'] - 8, self.system_state['envp'])
        self.set_reg_val('sp', self.system_state['stack_start'] - 24)

        # Set the entrypoint
        entrypoint = entrypoint if entrypoint is not None else self.binary_info['entrypoint']

        # Emulate and trace
        logging.info("Beginning emulation at entrypoint 0x%x..." % entrypoint)
        self.trace = self.emulate(entrypoint, trace_path=trace_path)
        logging.info("Emulation complete!")
    
    def _init_args(self):
        """Initializes the argc, argv, and envp data in memory and in self.system_state
        
        Values:

            - 'argc': the arg count (number of arguments). This is a 4-byte value that is passed by value instead of
              pointers to it. During stack creation, the value itself is pushed onto the stack.
            - 'argv': the argument vector. This is a pointer to an array of pointers, each of those being pointers to
              the start of null-terminated strings. argv will have an extra value of 0 appended to the end of it such
              that one could iterate through argv without needing to know argc before hand
            - 'envp': the environment variable pointer. Much like argv, this is a pointer to an array of pointers to
              null-terminated strings, with a final null pointer appended onto the array so that one could iterate
              through envp without needing to know the number of values beforehand. 
        """
        if 'cmd_args' not in self.system_inputs or self.system_inputs['cmd_args'] is None:
            self.system_inputs['cmd_args'] = random_strings(max_num=DEFAULT_MAX_NUM_STRINGS, max_size=self.system_inputs['string_max_size'])
        if 'environ' not in self.system_inputs or self.system_inputs['environ'] is None:
            self.system_inputs['environ'] = random_strings(max_num=DEFAULT_MAX_NUM_STRINGS, max_size=self.system_inputs['string_max_size'])
        
        def _clean_inputs(vals):
            """Makes sure all values are null-terminated bytes. Returns (array, count)"""
            vals = [a if isinstance(a, bytes) else a.encode('ascii') if isinstance(a, str) else bytes(a) for a in vals]
            vals = [(a if a[-1] == 0 else (a + bytes([0]))) for a in vals]
            vals = [(a if len(a) == self.system_inputs['string_max_size'] else (a + bytes([0] * (self.system_inputs['string_max_size'] - len(a))))) for a in vals]
            if any(len(a) > self.system_inputs['string_max_size'] for a in vals):
                raise ValueError("Found a string that is larger than the string_max_size for argv/envp input!")
            return vals, len(vals)
        
        # Insert the binary name, and clean all the inputs, then get envp data
        argv, argc = _clean_inputs([self.binary_info['name']] + self.system_inputs['cmd_args'])
        envv, envc = _clean_inputs(self.system_inputs['environ'])

        def _insert_into_mem(vals):
            """Inserts bytes into memory (if needed), returning a list of pointers (plus null-terminating pointer)"""
            total_size = sum(len(a) for a in vals)
            if total_size == 0:
                return [0]
            
            mem_loc = self.memory_manager.malloc(total_size, alignment=8)
            
            ret = [mem_loc]
            for v in vals:
                self.set_mem(ret[-1], v, size=1)
                # symoblize me
                ret.append(ret[-1] + len(v))
            ret[-1] = 0
            
            return ret

        # Put values into memory if needed
        argparr = _insert_into_mem(argv)
        envparr = _insert_into_mem(envv)

        def _insert_pointer_arrays(parr):
            """Puts pointer arrays into memory and returns their location"""
            parr_loc = self.memory_manager.malloc(len(parr) * self.pointer_size, alignment=16)
            for i, p in enumerate(parr):
                self.set_mem(parr_loc + i * self.pointer_size, p)
            return parr_loc
        
        # Put pointer arrays into memory
        argp = _insert_pointer_arrays(argparr)
        envp = _insert_pointer_arrays(envparr)

        # Set our system argc, argv, envp
        self.system_state.update({'argc': argc, 'argv': argp, 'envp': envp})
    
    def emulate(self, entrypoint, trace_path=False, stop_addr=None):
        """Begins emulation starting at the given entrypoint
        
        Args:
            entrypoint (Union[int, bytes]): memory address of the entrypoint, or a bytes() object containing machine
                code to execute. If using bytes, then those bytes will be placed all the way at the end of scratch
                space (to leave the beginning of it open for use), and a 'hlt' instruction will be placed at the end
                of the bytes to stop emulation once reaching the end. A stack is created right before the instructions
                that grows upwards (towards the start of scratch space)
            trace_path (bool): if True, will trace the path of execution in the CFG
            stop_addr (Optional[int]): if not None, then an integer memory address that, if the program counter ever
                reaches, will trigger a non-errored end of the emulation
        """
        self._finished_emulation = False
        self._ensure_context()

        # If entrypoint is a bytes() object, place the bytes at the end of scratch space and set the entrypoint
        #   accordingly. Also add a 'hlt' instruction at the end
        if isinstance(entrypoint, bytes):
            logging.info("Emulating passed bytes: %s" % entrypoint)

            if self.binary_info['arch'] not in [ARCH.X86, ARCH.X86_64]:
                raise ValueError("Need to add in the 'hlt' instruction for bytes() emulation for arch %s" % self.binary_info['arch'])
            
            # Get the bytes and place them at the end of memory
            inst_bytes = entrypoint + b"\xF4"
            inst_start = 2**(self.pointer_size * 8) - len(inst_bytes)
            self.set_mem(inst_start, inst_bytes)

            # Create the stack. Align stack to 32 bytes downwards (towards scratch start)
            stack_start = (inst_start - 1) & (2 ** (self.pointer_size * 8) - 32)
            self.set_reg_val('bp', stack_start)
            self.set_reg_val('sp', stack_start)

            # Set the new entrypoint
            entrypoint = inst_start

        # The trace through the cfg of execution, a list of basic blocks visited in order
        trace = [self.cfg.get_block(entrypoint)] if trace_path else None

        # Prepare things for debugging and ending emulation
        stop_debugging = False
        def reg_val(value):
            return self.get_reg_val(value)
        def _log_end_emulation():
            logging.debug("End of emulation: %s" % {'error_raised': error_raised(), 'ending_pc': self.get_reg_val('pc')})
        def _debug_loop():
            nonlocal stop_debugging, self, reg_val
            last_inst = 'next'
            while True:
                usr_in = input().strip()

                if usr_in in ['']:
                    usr_in = last_inst
                
                last_inst = usr_in

                if usr_in in ['step', 'next']:
                    break
                
                if usr_in in ['break', 'return']:
                    return
                
                if usr_in in ['run', 'continue']:
                    stop_debugging = True
                    break

                if usr_in in ['sysexit', 'interrupt', 'exit', 'quit']:
                    raise KeyboardInterrupt()

                try:
                    output = eval(usr_in)
                except Exception:
                    output = traceback.format_exc()

                print(output)
        
        pc = entrypoint
        while True:
            # Fetch current instruction
            inst = self.read_mem(pc, bytes, size=16)

            # Create the instruction and process
            instruction = Instruction(pc, inst)
            self.context.processing(instruction)
            logging.debug(instruction)

            # If it is a syscall, deal with it
            if instruction.getType() == OPCODE.X86.SYSCALL:
                logging.debug("Detected syscall...")
                handle_syscall(self)
            
            # If we should stop executing
            if error_raised() or instruction.getType() == OPCODE.X86.HLT or self.get_reg_val('pc') == 0:
                _log_end_emulation()
                break

            # Check if we are debugging, and allow interaction if so
            if self.debug and not stop_debugging:
                _debug_loop()

            # pc now points to address of next instruction and check if the last instruction was a control flow
            pc = self.get_reg_val('pc')
            inst_cf = instruction.isControlFlow()

            # Check if the new pc points to the address of a manual function
            if pc in self.manual_functions:
                # Emulate the routine. It should set the program counter accordingly
                self.manual_functions[pc](self)
                pc = self.get_reg_val('pc')
                inst_cf = True
            
            # Check if the new pc is at a stop address
            if stop_addr is not None and pc == stop_addr:
                _log_end_emulation()
                break

            # Check if we finished an emulation and should stop any debug running
            if self._finished_emulation:
                logging.info("Finish sub-emulation")
                stop_debugging = False
                self._finished_emulation = False
                _debug_loop()
            
            # Now that we have returned from any possible manual functions, we can check if we've reached a new block
            if trace_path and inst_cf:
                trace.append(pc)
                logging.info("Reached new basic block at address: 0x%0x" % pc)
        
        # Keep track of when we end an emulation so we can un-run emulations in different stack frames
        self._finished_emulation = True
        return trace
    
    def get_reg_name(self, reg_name):
        """Returns the register name specific to this architecture for the given name"""
        if reg_name in REGISTER_MAPPING[self.binary_info['arch']]:
            return REGISTER_MAPPING[self.binary_info['arch']][reg_name]
        raise ValueError("Unknown normalized register name: %s" % repr(reg_name))
    
    def get_triton_reg(self, reg_name):
        """Returns the triton register object from the given name
        
        Args:
            reg_name (str): the string register name to get. Can either be a register name specific to this architecture,
                or a normalized register name. Current normalized register names available are in REGISTER_MAPPING
        """
        self._ensure_context()
        if self.binary_info['arch'] not in REGISTER_MAPPING:
            raise ValueError("Cannot get the PC register on unknown arch %s, supported arch's: %s" 
                             % (self.binary_info['arch'], list(REGISTER_MAPPING.keys())))
        
        # Check for a normalized name
        if reg_name in REGISTER_MAPPING[self.binary_info['arch']]:
            return getattr(self.context.registers, REGISTER_MAPPING[self.binary_info['arch']][reg_name])

        # Otherwise check that it is a register name
        if not reg_name.startswith('_') and reg_name in dir(self.context.registers):
            return getattr(self.context.registers, reg_name)
        
        # Otherwise this is an unknown register name, raise an error
        raise ValueError("Unknown register name: %s" % repr(reg_name))
    
    def get_reg_val(self, reg_name):
        """Returns the register value for this specific architecture
        
        Args:
            reg_name (str): the string register name to get. Can either be a register name specific to this architecture,
                or a normalized register name. Current normalized register names available are in REGISTER_MAPPING
        """
        return self.context.getConcreteRegisterValue(self.get_triton_reg(reg_name))
    
    def set_reg_val(self, reg_name, value):
        """Returns the register value for this specific architecture
        
        Args:
            reg_name (str): the string register name to get. Can either be a register name specific to this architecture,
                or a normalized register name. Current normalized register names available are in REGISTER_MAPPING
            value (int): the integer value to set into the register
        """
        self._ensure_context()

        if not isinstance(value, int):
            raise TypeError("Can only insert integer values into registers")
        
        self.context.setConcreteRegisterValue(self.get_triton_reg(reg_name), value)
    
    def push_stack(self, value):
        """Pushes the given value onto the stack, updating the stack pointer
        
        Args:
            value (int): the value to push onto the stack. Currently can only be an integer, and will consume self.pointer_size
                bytes of memory on the stack
        """
        if not isinstance(value, int):
            raise TypeError("Can only push integers onto the stack")
        
        # We decrement the stack pointer since we're pushing
        self.inc_reg_val('sp', -self.set_mem(self.get_reg_val('sp') - self.pointer_size, value, signed=False))

    def pop_stack(self, data_type, size=None, encoding='ascii'):
        """Pops the given data type off of the stack and returns its value, incrementing the stack pointer.
        
        Does no checks to make sure this data_type can actually be popped off the stack. Assumes the stack has already
        been initialized

        Args:
            data_type (Union[str, type]): either a python type object for the data type to pop off the stack, or a string
                name of the type to pop off the stack
            size (Optional[int]): either an integer number of bytes to pop off, or None to infer the size from the
                `data_type`
            encoding (str): encoding to use when popping string objects
        
        Returns:
            Union[int, float, str, bytes]: the value popped off the stack
        """
        read_val, bytes_read = self.read_mem(self.get_reg_val('sp'), data_type, size=size, encoding=encoding, 
                                             ret_num_bytes=True, force_known_size=True)
        self.inc_reg_val('sp', bytes_read)  # Increment since we are popping off the stack
        return read_val
    
    def peek_stack(self, data_type, size=None, encoding='ascii'):
        """Peeks at the data at the top of the stack, not moving the stack pointer
        
        Does no checks to make sure this data_type can actually be popped off the stack. Assumes the stack has already
        been initialized

        Args:
            data_type (Union[str, type]): either a python type object for the data type to pop off the stack, or a string
                name of the type to pop off the stack
            size (Optional[int]): either an integer number of bytes to pop off, or None to infer the size from the
                `data_type`
            encoding (str): encoding to use when popping string objects
        
        Returns:
            Union[int, float, str, bytes]: the value peeked off the stack
        """
        return self.read_mem(self.get_reg_val('sp'), data_type, size=size, encoding=encoding, ret_num_bytes=False, 
                             force_known_size=True)

    def read_mem(self, mem_loc, data_type, size=None, encoding='ascii', ret_num_bytes=False, force_known_size=True):
        """Reads the data type from the given memory location

        Args:
            mem_loc (int): the memory location to read from
            data_type (Union[str, type]): either a python type object for the data type to pop off the stack, or a string
                name of the type to pop off the stack. Names can be:

                - 'int[size]': an integer, optionally `size` bytes large
                - 'float[size]': a float, optionally `size` bytes large
                - 'str[size]': a string. Defaults to reading bytes until reaching a null terminator byte, but optionally
                  will read `size` bytes. Will decode the string using `encoding`
                - 'bytes[size]': reads the raw bytes. Defaults to reading bytes until reaching a null terminator byte,
                  but optionally will read `size` bytes.

            size (Optional[int]): either an integer number of bytes to pop off, or None to infer the size from the
                `data_type`
            encoding (str): encoding to use when reading in string objects
            ret_num_bytes (bool): if True, will return a 2-tuple of (read_value, num_bytes_read)
            force_known_size (bool): if True, then a size must be known prior to reading the object. Used mostly for the
                pop_stack() function since you'd need to know the size of the object you are popping. NOTE: when reading
                from the stack, you just need to pass the stack pointer in the mem_loc
        
        Returns:
            Union[Union[int, float, str, bytes], Tuple[Union[int, float, str, bytes], int]]: either the value read from
                memory, or a 2-tuple of (read_value, num_bytes_read)
        """
        if isinstance(data_type, type):
            if data_type in [int, float, str, bytes]:
                data_type = data_type.__name__
        
        elif isinstance(data_type, str):
            data_type = data_type.lower()
            re_match = DT_RE.fullmatch(data_type)

            if re_match is None:
                raise ValueError("Unknown data_type string: %s" % repr(data_type))
            
            data_type, *rest = re_match.groups()

            if data_type in ['int', 'integer', 'long']:
                data_type = 'int'
            elif data_type in ['float', 'floating', 'double']:
                data_type = 'float'
            elif data_type in ['str', 'string']:
                data_type = 'str'
            elif data_type in ['bytes', 'byte']:
                data_type = 'bytes'
            elif data_type in ['pointer']:  # Names that don't change
                pass
            else:
                raise NotImplementedError

            if len(rest) > 0:
                size = rest[0]
        
        else:
            raise TypeError("Unknown data_type type: %s" % repr(type(data_type).__name__))
        
        # Check that size is good
        if size is not None and (size <= 0 or \
                (data_type == 'float' and size not in [4, 8]) or (data_type == 'pointer' and size != self.pointer_size)):
            raise ValueError("Invalid size of %d bytes for datatype: %s" % (size, repr(data_type)))
        
        # Read in the value
        if data_type in ['int', 'float', 'pointer']:
            size = self.pointer_size if size is None else size
            read_val = self.context.getConcreteMemoryValue(MemoryAccess(mem_loc, size))
            read_val = get_float_from_bits(read_val) if data_type == 'float' else read_val
            ret_bytes = size

        elif data_type in ['str', 'bytes']:
            if size is not None:
                read_val = self.context.getConcreteMemoryAreaValue(mem_loc, size)
                ret_bytes = size

            else:
                # Can't do this if force_known_size is True
                if force_known_size:
                    raise ValueError("Cannot get arbitrary sized object with force_known_size=True")
                
                read_bytes = [self.context.getConcreteMemoryValue(mem_loc, 1)]
                while read_bytes[-1] != 0:
                    read_bytes.append(self.context.getConcreteMemoryValue(mem_loc + len(read_bytes), 1))
                
                read_val = bytes(read_bytes)
                ret_bytes = len(read_val)

            read_val = read_val.encode(encoding) if data_type == 'str' else read_val

        else:
            raise NotImplementedError
        
        # Return the value, and the number of bytes read if needed
        if ret_num_bytes:
            return read_val, ret_bytes
        return read_val
    
    def inc_reg_val(self, reg_name, increment=1):
        """Increments the given register by the given amount

        Args:
            reg_name (str): the string register name to get. Can either be a register name specific to this architecture,
                or a normalized register name. Current normalized register names available are in REGISTER_MAPPING
            increment (int): integer value to increment by
        """
        if not isinstance(increment, int):
            raise TypeError("Increment value must be integer")
        
        self.set_reg_val(reg_name, self.get_reg_val(reg_name) + increment)
    
    def set_mem(self, mem_loc, value, size=None, signed=True, encoding='ascii'):
        """Sets the given memory location to the given value. Memory location should have already been allocated
        
        Args:
            mem_loc (int): memory location to store value at (should have already been allocated, no checks are done
                to make sure you aren't overwriting memory where you shouldn't be)
            value (Union[int]): value to store at the given memory location
            size (Optional[int]): number of bytes to use. If None, number of bytes will be inferred based on object
                type:
                
                - integer: self.pointer_size bytes, stored as signed/unsigned based on the `signed` argument
                - float: self.pointer_size bytes, stored as IEE-754 format
                - bytes: length of bytes
                - str: string is ascii encoded, then stored as bytes() object

            signed (bool): only used when `value` is an integer. If True, then `value` will be stored as a signed integer
                of the given/inferred `memsize`, otherwise it will be unsigned
            encoding (str): only used when `value` is a string. The encoding to use to encode the string to bytes
        
        Returns:
            int: the number of bytes used to store the value
        """
        if size is not None and size <= 0:
            raise ValueError("Memsize must be >= 1")
        
        if isinstance(value, int):
            size = self.pointer_size if size is None else size

            # Check if the integer value is too large to store in the given number of bytes
            if (signed and (value > 2 ** (8 * size - 1) - 1 or value < -(2 ** (8 * size - 1)))) or \
                (not signed and (value > 2 ** (8 * size) - 1) or value < 0):
                raise ValueError("Value cannot be stored as a %d-byte %s integer: %d" 
                                 % (size, 'signed' if signed else 'unsigned', value))
            
        elif isinstance(value, float):
            size = self.pointer_size if size is None else size
            value = get_float_bits(value, size)
        
        elif isinstance(value, str):
            size = -1
            value = value.encode(encoding)
        
        elif isinstance(value, bytes):
            size = -1
        
        else:
            raise TypeError("Cannot store value of type %s in memory" % repr(type(value).__name__))
        
        # Store the actual value in memory
        if size == -1:
            self.context.setConcreteMemoryAreaValue(mem_loc, value)
            size = len(value)
        else:
            self.context.setConcreteMemoryValue(MemoryAccess(mem_loc, size), value)
        
        # Return the number of bytes stored
        return size

    def get_registers(self):
        """Returns a dictionary with keys being all registers, and values being their integer values"""
        if self.binary_info['arch'] not in ARCH_REG_NAMES:
            raise ValueError("Binary arch is not present in ARCH_REG_NAMES: %s" % repr(self.binary_info['arch']))
        return {reg: self.get_reg_val(reg) for reg in ARCH_REG_NAMES[self.binary_info['arch']]}
    
    def set_registers(self, values):
        """Sets register values based on the given dictionary of values
        
        Args:
            values (Dict[str, int]): dictionary mapping string register names to their integer values
        """
        for reg, val in values.items():
            self.set_reg_val(reg, val)
    
    @property
    def pointer_size(self):
        """Returns the pointer size. 4 bytes for 32-bit os, 8 bytes for 64-bit"""
        return 4 if self.binary_info['mode'] == lief.MODES.M32 else 8
    
    def _get_next_thread_idx(self):
        """Returns the next index of the next available thread"""
        if len(self.threads) == 0:
            self.threads = [None] * 10
            return 0
        
        for i in range(len(self.threads)):
            if self.threads[i] is None:
                return i
        
        self.threads += [None] * 10
        return i
    
    def _ensure_context(self):
        """Ensures the context has been initialized"""
        if self.context is None:
            raise TypeError("Context must be initialized before operation can be performed.")

    def __str__(self):
        return "BinaryExecutor with binary info: %s" % _printable_binary_info(self.binary_info)
    
    def __repr__(self):
        return str(self)


def _printable_binary_info(binary_info):
    ret = binary_info.copy()
    ret['segments'] = [(addr, size) for addr, size, content in ret['segments']]
    ret['relocations'] = [(rel['name'], rel['type'].name) for rel in ret['relocations']]
    return ret


def _make_mode_bits(value, mode):
    """Takes value (which should be 32-bit), and changes it to 64-bit if needed for the mode (by adding all 1's to start)"""
    return value if mode == lief.MODES.M32 else (0xFFFF_FFFF_0000_0000 | value)


# Keeps track of which registers do what between architectures
REGISTER_MAPPING = {
    ARCH.X86: {
        'pc': 'eip',
        'bp': 'ebp',
        'sp': 'esp',
        'ret': 'eax',
        'flags': 'eflags',
    },

    ARCH.X86_64: {
        'pc': 'rip',
        'bp': 'rbp',
        'sp': 'rsp',
        'ret': 'rax',
        'flags': 'rflags',
    },
}


# Lists of register names needed to keep track of all registers for the given architecture
ARCH_REG_NAMES = {
    ARCH.X86: ['eax', 'ebx', 'ecx', 'edx', 'cs', 'ds', 'es', 'fs', 'gs', 'ss', 'esi', 'edi', 'ebp', 'eip', 'esp', 'eflags'],
    ARCH.X86_64: [
        # General purpose
        'rax', 'rbx', 'rcx', 'rdx', 'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',

        # base pointer, stack pointer, instruction pointer, source index, destination index, flags
        # NOTE: triton only uses eflags, likely because the extra 32 bits allotted in rflags are all currently reserved
        'rbp', 'rsp', 'rip', 'rsi', 'rdi', 'eflags',

        # Control registers. These control the general behavior of the CPU (IE: things involving paging, virtual memory,
        #   errors, etc.)
        # See more here: https://en.wikipedia.org/wiki/Control_register
        # The MXCSR register contains control info for SSE (Streaming SIMD Extensions) registers:
        #   https://help.totalview.io/previous_releases/2019/html/index.html#page/Reference_Guide/Intelx86MXSCRRegister.html
        # Note: the MSW register is the lower 16 bits of CR0
        'cr0', 'cr1', 'cr2', 'cr3', 'cr4', 'cr5', 'cr6', 'cr7', 'cr8', 'cr9', 'cr10', 'cr11', 'cr12', 'cr13', 'cr14',
        'cr15', 'mxcsr',

        # Segment registers, contain memory locations of starts of different segments
        'cs', 'ss', 'ds', 'es', 'fs', 'gs',

        # AVX/SSE registers
        'zmm0', 'zmm1', 'zmm2', 'zmm3', 'zmm4', 'zmm5', 'zmm6', 'zmm7', 'zmm8', 'zmm9', 'zmm10', 'zmm11', 'zmm12', 'zmm13',
        'zmm14', 'zmm15', 'zmm16', 'zmm17', 'zmm18', 'zmm19', 'zmm20', 'zmm21', 'zmm22', 'zmm23', 'zmm24', 'zmm25',
        'zmm26', 'zmm27', 'zmm28', 'zmm29', 'zmm30', 'zmm31',
 
        # NOTE: there are debug registers (dr0-7), but these are ignored for now
    ],
}


# Regular expression for figuring out the needed data_type for reading memory
DT_RE = re.compile(r'(pointer|int|integer|long|float|floating|double|str|string|bytes|byte)([0-9]+)?')
