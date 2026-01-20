"""
Contains functions that help with loading the binaries, getting abi/version/bit-ness/endian-ness, etc.
"""

import os
import lief
import logging
import enum
from triton import ARCH, CPUSIZE, MemoryAccess


class ABIVersion(enum.Enum):
    """An Enum for the various ABI versions"""
    SYSTEMV = 0
    LINUX = 1  # Here for linux .so files


class RelocationType(enum.Enum):
    """Enum for different relocation types. Values are lists that contain all the possible string names of that 
    relocation in lief
    
    Extra information about relocation types can be found here: https://www.intezer.com/blog/malware-analysis/executable-and-linkable-format-101-part-3-relocations/

    Current relocation types and VALUES computation (addresses to store those values are in the .address value):

        - SYMBOL (VALUE_COMPUTATION) : DESCRIPTION

        - JUMP_SLOT (S): PLT entry
        - RELATIVE (B + A): store this value relative to the start of program memory (likely 0x0 for us since we are emulating)
        - GLOB_DAT (S): global variable. Can also be a function's address
        - R64 (S + A): store the literal value (should be 64-bit), also called R_X86_64_64 relocation
        - TPOFF (@tpoff(S + A)): store the @tpoff() of the address S + A, the offset to the TLS section (Thread-local storage)
        - IRELATIVE (indirect(S + A)): store the indirect function value at the given address + addend. 
    
    Uses the values:

        - S: the corresponding symbol's actual value
        - B: the program image base (virtual address of the start of the program itself. Likely 0x0 for us since we
          are emulating)
        - A: the addend value if using
        - @tpoff(addr): the negative tls offset value of addr. IE: the value to add to addr to get to the beginning
    
    NOTE: the size (number of bytes) of the relocation value to place can be determined by the relocation type, but it
    is already computed for us in the relocation.size attribute, so that's what we'll use

    Extra information about relocations (for systemv abi) can be found on page 70 of: 
    https://refspecs.linuxbase.org/elf/x86_64-abi-0.98.pdf

    Some of the relocations are just listed there instead of described because they are for thread-local storage:
    https://www.uclibc.org/docs/tls.pdf
    """
    JUMP_SLOT = ['JUMP_SLOT']
    RELATIVE = ['RELATIVE', 'RELATIVE64']
    IRELATIVE = ['IRELATIVE']
    GLOB_DAT = ['GLOB_DAT']
    R64 = ['R64']
    TPOFF = ['TPOFF64', 'TPOFF']


class SymbolType(enum.Enum):
    """Different symbol types. Values are lists that contain all the possible string names of that relocation in lief
    
    Different symbols may have different interpretations of their values, or other things to do before determining
    the value for that symbol at load time.

    Symbol types:

        - SYMBOL_TYPE (VALUE_CALULATION): description

        - OBJECT (V + O): this symbol is a reference to an object of some type (array, int, etc.). Value indicates the 
          virtual address of the object.
        
    Uses the values:

        - V: the symbol's value
        - O: the memory offset. Offset to where the binary was loaded (usually to add to virtual address)

    """
    OBJECT = ['OBJECT']
    FUNC = ['FUNC']
    GNU_IFUNC = ['GNU_IFUNC']


# lief.EXE_FORMATS.MACHO, lief.EXE_FORMATS.PE
SUPPORTED_BINARY_FORMATS = [lief.EXE_FORMATS.ELF]
SUPPORTED_BINARY_ARCHS = [ARCH.X86_64]

# Maps lief architectures for different file formats to triton architectures
LIEF_ARCH_TO_TRITON_ARCH = {

    # ELF files
    lief.ELF.ARCH.x86_64: ARCH.X86_64,
    lief.ELF.ARCH.AARCH64: ARCH.AARCH64,
    lief.ELF.ARCH.i386: ARCH.X86,
    lief.ELF.ARCH.ARM: ARCH.ARM32,

    # MachO files
    lief.MachO.CPU_TYPES.x86_64: ARCH.X86_64,
    lief.MachO.CPU_TYPES.x86: ARCH.X86,
    lief.MachO.CPU_TYPES.I386: ARCH.X86,
    lief.MachO.CPU_TYPES.ARM: ARCH.ARM32,
    lief.MachO.CPU_TYPES.ARM64: ARCH.AARCH64,

    # PE files
    lief.PE.MACHINE_TYPES.AMD64: ARCH.X86_64,
    lief.PE.MACHINE_TYPES.ARM: ARCH.ARM32,
    lief.PE.MACHINE_TYPES.ARM64: ARCH.AARCH64,
    lief.PE.MACHINE_TYPES.I386: ARCH.X86,
}

# Maps triton ARCH enum objects to their associated lief MODES enum object (for 32 vs 64 bit architectures)
TRITON_ARCH_TO_LIEF_MODE = {
    ARCH.AARCH64: lief.MODES.M64,
    ARCH.X86_64: lief.MODES.M64,
    ARCH.ARM32: lief.MODES.M32,
    ARCH.X86: lief.MODES.M32,
}

# Maps lief header mode values to a lief.MODES enum for 32 vs 64 bit machines
LIEF_HEADER_MODE_TO_MODES_ENUM = {

    # ELF files
    lief.ELF.ELF_CLASS.CLASS32: lief.MODES.M32,
    lief.ELF.ELF_CLASS.CLASS64: lief.MODES.M64,

    # MachO files
    lief.MachO.MACHO_TYPES.CIGAM: lief.MODES.M32,
    lief.MachO.MACHO_TYPES.CIGAM_64: lief.MODES.M64,
    lief.MachO.MACHO_TYPES.MAGIC: lief.MODES.M32,
    lief.MachO.MACHO_TYPES.MAGIC_64: lief.MODES.M64,

    # PE files
    lief.PE.PE_TYPE.PE32: lief.MODES.M32,
    lief.PE.PE_TYPE.PE32_PLUS: lief.MODES.M64,
}

# Maps lief ABI versions to their enum couterparts
LIEF_ABI_TO_ABI = {
    lief.ELF.OS_ABI.SYSTEMV: ABIVersion.SYSTEMV,
    lief.ELF.OS_ABI.LINUX: ABIVersion.LINUX,
}

# Maps lief format to dictionary that maps triton arch to its corresponding relocation type enum
LIEF_FORMAT_ARCH_TO_RELOCATION_TYPE_ENUM = {
    lief.EXE_FORMATS.ELF: {
        ARCH.AARCH64: lief.ELF.RELOCATION_AARCH64,
        ARCH.X86_64: lief.ELF.RELOCATION_X86_64,
        ARCH.ARM32: lief.ELF.RELOCATION_ARM,
        ARCH.X86: lief.ELF.RELOCATION_i386,
    },

    lief.EXE_FORMATS.MACHO: {
        ARCH.AARCH64: lief.MachO.ARM64_RELOCATION,
        ARCH.X86_64: lief.MachO.ARM_RELOCATION,
        ARCH.ARM32: lief.MachO.X86_64_RELOCATION,
        ARCH.X86: lief.MachO.X86_RELOCATION,
    },
}


# The segment types that we actually load into memory
LOAD_SEGMENT_TYPES = [lief.ELF.SEGMENT_TYPES.LOAD]


def parse_binary(bin_obj):
    """
    Parses a binary from the given binary/binary_path using lief

    Args:
        bin_obj (Union[str, Iterable[int], BytesIO, lief.Binary]): the binary object to load. Can be:

            - str: a string path to the binary to load
            - Iterable[int]: an iterable of integers (including bytes() objects) of the bytes in the binary file
            - BytesIO: an open file object to load from. Must be an object with a callable .read() attribute that returns
                an iterable of the bytes in the binary file
            - lief.Binary: an already parsed lief binary, will simply be returned
    
    Raises:
        FileNotFoundError: if bin_obj looks like a file path, but the path cannot be found
        BinaryLoadError: if the binary could not be loaded
    
    Returns:
        lief.Binary: the parsed lief binary object
    """
    # Load in the binary
    try:
        if isinstance(bin_obj, lief.Binary):
            return bin_obj
        elif isinstance(bin_obj, str):
            logging.info("Loading binary from path: %s..." % bin_obj)
            if not os.path.exists(bin_obj):
                raise FileNotFoundError("Could not find binary file to load: %s" % bin_obj)
            bin_obj = lief.parse(bin_obj)
        elif hasattr(bin_obj, 'read') and callable(bin_obj.read):
            logging.info("Loading binary from open file...")
            bin_obj = lief.parse(list(bin_obj.read()))
        else:
            logging.info("Loading binary from iterable of bytes...")
            bin_obj = lief.parse(list(bin_obj))
        
        if bin_obj is None:
            raise BinaryLoadError("Binary was not loaded!")
    except Exception as e:
        raise BinaryLoadError("Could not load binary from given `binary` of type: %s.\nMessage: %s" % 
            (repr(type(bin_obj).__name__), e))
    
    return bin_obj


def get_binary_format(binary):
    """Returns the lief.EXE_FORMATS enum for the given binary
    
    Args:
        binary (lief.binary): the parsed lief binary

    Returns:
        lief.EXE_FORMATS: the lief.EXE_FORMATS enum object
    """
    format = binary.format
    if format not in SUPPORTED_BINARY_FORMATS:
        raise BinaryLoadError("Unsupported binary format: %s. Currently supported formats: %s" % 
            (repr(format.name), SUPPORTED_BINARY_FORMATS))
    return format


def get_binary_arch(binary):
    """Returns the architecture (Triton.ARCH) of the given lief binary, raising an error if it is not supported
    
    Args:
        binary (lief.Binary): the parsed lief binary
    
    Returns:
        triton.ARCH: the triton ARCH enum for the architecture of the binary
    """
    format = get_binary_format(binary)

    if format == lief.EXE_FORMATS.ELF:
        bin_arch = binary.header.machine_type
    elif format == lief.EXE_FORMATS.MACHO:
        bin_arch = binary.header.cpu_type
    elif format == lief.EXE_FORMATS.PE:
        bin_arch = binary.header.machine
    
    bin_arch = LIEF_ARCH_TO_TRITON_ARCH[bin_arch]

    if bin_arch not in SUPPORTED_BINARY_ARCHS:
        raise ValueError("Binary arch %s not in supported architectures: %s" % (repr(bin_arch.name)), SUPPORTED_BINARY_ARCHS)

    # Check that the mode is the expected version for the arch
    if get_binary_mode(binary) != TRITON_ARCH_TO_LIEF_MODE[bin_arch]:
        raise ValueError("Binary mode doesn't match the expected arch mode: %s != %s" %
            (get_binary_mode(binary).name, TRITON_ARCH_TO_LIEF_MODE[bin_arch].name))

    return bin_arch


def get_binary_mode(binary):
    """Returns a lief.MODES enum object, 'M32' for 32-bit machine, 'M64' for 64-bit machine"""
    format = get_binary_format(binary)

    if format == lief.EXE_FORMATS.ELF:
        mode = binary.header.identity_class
    elif format == lief.EXE_FORMATS.MACHO:
        mode = binary.header.magic
    elif format == lief.EXE_FORMATS.PE:
        mode = binary.optional_header.magic
    
    return LIEF_HEADER_MODE_TO_MODES_ENUM[mode]


def get_binary_entrypoint(binary):
    """Gets the integer entrypoint address from the given lief parsed binary"""
    return binary.entrypoint


def get_binary_abi(binary):
    """Gets the ABIVersion enum object from the given lief parsed binary"""
    format = get_binary_format(binary)

    if format == lief.EXE_FORMATS.ELF:
        abi = binary.header.identity_os_abi
    elif format == lief.EXE_FORMATS.MACHO:
        raise NotImplementedError
    elif format == lief.EXE_FORMATS.PE:
        raise NotImplementedError
    
    if abi not in LIEF_ABI_TO_ABI:
        raise ValueError("Unsupported ABI version: %s" % abi)
    
    return LIEF_ABI_TO_ABI[abi]


def get_relocation_type_enum(binary):
    """Returns the relocation type enum for the given binary"""
    format, arch = get_binary_format(binary), get_binary_arch(binary)
    return LIEF_FORMAT_ARCH_TO_RELOCATION_TYPE_ENUM[format][arch]


def get_relocation_type(val, rel_type_enum):
    """Returns a RelocationType enum value
    
    Args:
        val (int): the rel.type value for a relocation
        rel_type_enum (Enum): the enum to get the names of the relocation types from the given val
    """
    for rel_type in [getattr(rel_type_enum, attr) for attr in [a for a in dir(rel_type_enum) if a.isupper()]]:
        if val == rel_type.value:
            name = rel_type.name
            break
    else:
        raise ValueError("Could not find the value '%s' in the given rel_type_enum: %s" % (val, rel_type_enum))
    
    for rel_type in RelocationType:
        if name in rel_type.value:
            return rel_type
    
    raise ValueError("Unknown relocation type name: %s" % repr(name))


def get_binary_relocations(binary):
    """Returns a list of dictionaries of relocation information, one dict per relocation
    
    Relocation dictionary contains:
        - name (Union[str, None]): the symbol name, or None if it doesn't have one
        - address (int): the memory address corresponding to this relocation. Might have different meanings depending
            on the type of relocation. See more info here: https://www.intezer.com/blog/malware-analysis/executable-and-linkable-format-101-part-3-relocations/
        - addend (int): integer value to add to the value that you are going to store
        - type (RelocationType): the type of relocation
        - size (int): size in bytes of the relocation
    """
    format = get_binary_format(binary)

    if format == lief.EXE_FORMATS.ELF:
        rel_type_enum = get_relocation_type_enum(binary)

        relocations = [{
                'name': rel.symbol.name if rel.symbol is not None else None, 
                'address': rel.address, 
                'addend': rel.addend, 
                'type': get_relocation_type(rel.type, rel_type_enum),
                'size': rel.size // 8,
                'relocation': rel,
            }
        for rel in binary.relocations]

    elif format == lief.EXE_FORMATS.MACHO:
        # Need to figure out how MACHO files handle their relocations
        raise NotImplementedError
    elif format == lief.EXE_FORMATS.PE:
        raise NotImplementedError
    
    return relocations


def get_binary_segments(binary):
    """Returns a list of (address: int, virt_size: int, content: bytes) tuples for information for the sections
    
    Does not return any sections with a segment type in IGNORED_SEGMENT_TYPES
    """
    return [(s.virtual_address, s.virtual_size, s.content.tobytes()) for s in binary.segments if s.type in LOAD_SEGMENT_TYPES]


def get_binary_os(binary):
    """Returns the os for the binary ('linux', 'macos', or 'windows')"""
    format = get_binary_format(binary)
    return 'linux' if format == lief.EXE_FORMATS.ELF else 'macos' if format == lief.EXE_FORMATS.MACHO else \
        'windows' if format == lief.EXE_FORMATS.PE else None
    


def get_binary_ld_library_path(binary, os_version):
    """Returns the directory to search for dynamic libraries based on the binary
    
    Args:
        binary (lief.Binary): the parsed lief binary
        os_version (str): the os version to use. Currently available versions:
    """
    lib_path = os.path.join(os.path.dirname(__file__), 'lib', get_binary_os(binary), os_version)

    arch = get_binary_arch(binary)

    if arch == ARCH.X86_64:
        lib_path = os.path.join(lib_path, 'x86_64')
    elif arch == ARCH.X86:
        lib_path = os.path.join(lib_path, 'x86')
    elif arch == ARCH.ARM32:
        lib_path = os.path.join(lib_path, 'arm32')
    elif arch == ARCH.AARCH64:
        lib_path = os.path.join(lib_path, 'aarch64')
    else:
        raise NotImplementedError

    if not os.path.exists(lib_path) or not os.path.isdir(lib_path):
        raise ValueError("Path doesn't exist, or is not a directory: %s" % lib_path)
    
    return lib_path


def get_binary_sys_info(binary, os_version):
    """Returns a dictionary of system info for use in system calls"""
    with open(os.path.join(get_binary_ld_library_path(binary, os_version), 'sys_info.txt'), 'r') as f:
        ret = eval(f.read())
    
    # Create the inverse mappings too
    updates = {}
    for k, v in ret.items():
        if isinstance(v, dict):
            updates[k + '_inv'] = {v2: k2 for k2, v2 in v.items()}
    ret.update(updates)
    
    return ret


def get_binary_info(binary, os_version):
    """Loads the given binary and returns a dictionary of all the information"""
    binary = parse_binary(binary)

    return {
        'binary': binary,
        'name': binary.name,
        'cfg_path': get_binary_cfg_path(binary),
        'os': get_binary_os(binary),
        'os_version': os_version,
        'format': get_binary_format(binary),
        'arch': get_binary_arch(binary),
        'mode': get_binary_mode(binary),
        'entrypoint': get_binary_entrypoint(binary),
        'abi': get_binary_abi(binary),
        'ld_path': get_binary_ld_library_path(binary, os_version),
        'sys_info': get_binary_sys_info(binary, os_version),
        'relocations': get_binary_relocations(binary),
        'segments': get_binary_segments(binary),
    }


def get_binary_cfg_path(binary):
    """Returns the path to the binary's associated CFG, or None if it doesn't exist
    
    A binary's associated CFG should be placed right next to the binary, and should have either a '.pkl' (for pre-loaded
    and pickled CFG objects), or '.cfg'/'.txt' (for disassembler output files) file extension.
    """
    dirname, basename = os.path.dirname(binary.name), os.path.basename(binary.name)
    for ext in ['.pkl', '.cfg', '.txt']:
        new_name = os.path.join(dirname, (basename.rpartition('.')[0] if '.' in basename else basename) + ext)
        if os.path.exists(new_name):
            return new_name
    return None


def get_symbol_type(stype):
    """Returns the SymbolType for different lief symbol types"""
    for symbol_type in SymbolType:
        if stype.name in symbol_type.value:
            return symbol_type
    
    raise ValueError("Unknown symbol type name: %s" % repr(stype.name))


class BinaryLoadError(Exception):
    "Error raised when attempting to load a binary"
