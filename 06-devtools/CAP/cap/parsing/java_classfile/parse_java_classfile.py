"""
This file parses java class files, as well as has decent documentation about the structure of java class files for when
I inevitably forget how all this works.

This is working off of the Java 20 documentation found here: https://docs.oracle.com/javase/specs/jvms/se20/jvms20.pdf
It is assumed that any previous Java version's class files will also be able to be parsed by this format.
"""

import traceback
from parsing.java_classfile.classfile_utils import ByteReader
from parsing.java_classfile.access_flags import *
from parsing.java_classfile.constant_pool import *
from parsing.java_classfile.constant_pool import _get_check_pool_object
from parsing.java_classfile.field_and_method_info import FieldInfo, parse_fields, MethodInfo, parse_methods
from parsing.java_classfile.attributes import AttributeInfo, parse_attributes


class JavaClassfile:
    """
    A class that can parse java files and return their information.


    Parameters
    ----------
    java_class: `Union[bytes, str, IO]`
        The java classfile to read. Can be:

            - bytes: a bytes-like object that contains the raw binary
            - str: a string filepath to a java class file
            - IO: an object with a .read() function that returns a bytes-like object
    """

    magic: 'int'
    """(unsigned 4-byte) Identifies this file type. Should always be '0xCAFEBABE'"""

    minor_version: 'int'
    """(unsigned 2-byte) The minor version, ranges between [0, 65535]
    
    Different major version have different customs with minor versions, see the docs for more info. 
    Total version number is "major_version.minor_version" 
    """
    
    major_version: 'int'
    """(unsigned 2-byte) The major version, ranges between [45, 64] (as of Java 20)
    
    Total version number is "major_version.minor_version"
    """
    
    constant_pool: 'list[CPInfo]'
    """(array of cp_info) The constant pool
    
    A list of structures representing various string constants, class and interface names, field names, and other 
    constants that are referred to within the ClassFile structure and its substructures. The format of each pool
    table entry is indicated by its first "tag" byte. The pool table is indexed from 1 to pool_count - 1.
    The 0-th element is left as the CP_EMPTY object.
    """

    access_flags: 'ClassAccessFlags'
    """(unsigned 2-byte) Bit flags with various meanings

    This is an already-parsed version of the u2 bit flags. It has been parsed into an AccessFlags object (see AccessFlags
    for more info)
    """

    this_class: 'CPClassInfo'
    """(unsigned 2-byte) An index in the constant pool for info corresponding to the class/interface defined in this file
    
    This has been parsed from its constant pool index. It is instead the CPInfo object that contains the class/interface info.
    """

    super_class: 'Union[CPClassInfo, None]' = None
    """(unsigned 2-byte) An index in the constant pool for info corresponding to the superclass of this class
    
    Can theoretically be 0 if this class represents the Object class as it is the only class with no superclass

    This has been parsed from its constant pool index. It is instead the CPInfo object that contains the class/interface
    info, or None if this the the 'Object' classfile
    """

    interfaces: 'list[CPClassInfo]'
    """(array of unsigned 2-byte) Array of indices in the constant pool corresponding to interfaces for this class/interface

    Each value in the interfaces array must be a valid index into the pool table. The pool entry at each
    value of interfaces[i], where 0 <= i < interfaces_count, must be a class info structure representing an interface 
    that is a direct superinterface of this class or interface type, in the left-to-right order given in the source for the type
    
    This has been parsed from its constant pool indices. It is instead a list of the CPClassInfo objects.
    """

    fields: 'list[FieldInfo]'
    """(array of field_info) Array of field information structs for this class/interface

    Each value in the fields table must be a field_info structure giving a complete description of a field in this class
    or interface. The fields table includes only those fields that are declared by this class or interface. It does not 
    include items representing fields that are inherited from superclasses or superinterfaces
    """

    methods: 'list[MethodInfo]'
    """(array of method_info) Array of method information structs for this class/interface
    
    Each value in the methods table must be a method_info structure giving a complete description of a method in this 
    class or interface. If neither of the ACC_NATIVE and ACC_ABSTRACT flags are set, then the bytecode instructions are 
    also supplied. The method_info structures represent all methods declared by this class or interface type including 
    instance methods, class methods, instance initialization methods, and any class or interface initialization method. 
    Does not include any methods inhereted from superclasses or superinterfaces
    """

    attributes: 'list[AttributeInfo]'
    """(array of attribute_info) Array of attribute information structs for this class/interface
    """

    def __init__(self, java_class):
        # Get the actual bytes to parse
        java_bytes = self._read_bytes(java_class)

        # Parse the bytes
        self._parse_bytes(java_bytes)
    
    def _read_bytes(self, java_class):
        """Takes the input and converts it into a bytes() object"""
        _byte_types = (bytes, bytearray)

        if isinstance(java_class, str):
            with open(java_class, 'rb') as f:
                return f.read()
        elif isinstance(java_class, _byte_types):
            return bytes(java_class)
        elif hasattr(java_class, 'read') and callable(java_class.read):
            try:
                ret = java_class.read()
            except Exception as e:
                raise TypeError("Exception during calling .read() on passed java_class of type %s\nMessage: %s\nTraceback: %s" 
                                % (repr(type(java_class).__name__), e, traceback.format_exc()))
            
            if not isinstance(ret, _byte_types):
                raise TypeError("java_class's .read() function must return a bytes-like object, not: %s" % repr(type(ret).__name__))

            return ret
        else:
            raise TypeError("Unknown java_class type %s, must be: str, bytes-like, or io-like" % repr(type(java_class).__name__))
    
    def _parse_bytes(self, java_bytes):
        """Parses the java_bytes to get the classfile info
        
        Classfile structure:

        ClassFile {
            u4 magic;
            u2 minor_version;
            u2 major_version;
            u2 pool_count;
            cp_info pool[pool_count-1];
            u2 access_flags;
            u2 this_class;
            u2 super_class;
            u2 interfaces_count;
            u2 interfaces[interfaces_count];
            u2 fields_count;
            field_info fields[fields_count];
            u2 methods_count;
            method_info methods[methods_count];
            u2 attributes_count;
            attribute_info attributes[attributes_count];
        }

        NOTE: the pool_count is the number of constants in the constant pool plus one, but the rest of the
        counts are just the plain counts (no plus one)

        NOTE: Multi-byte values are always stored in big-endian order (highest byte first)
        
        Constant Pool Structures:
        cp_info structures follow the basic format:

        cp_info {
            u1 tag,
            u1 info[]
        }

        where:

            * tag (unsigned 1-byte) - unsigned integer tag between [0, 255] identifying this type of constant pool information
            * info (array of unsigned 1-byte) - the information
        
        """
        br = ByteReader(java_bytes)

        self.magic = br.read_int(4, signed=False)
        if self.magic != 0xCAFEBABE:
            raise ValueError("Parsed an invalid magic number: %x, should always be 0xCAFEBABE" % self.magic)
        
        self.minor_version = br.read_int(2, signed=False)
        self.major_version = br.read_int(2, signed=False)
        _minv, _maxv = 45, 100
        if not _minv <= self.major_version <= _maxv:
            raise ValueError("Parsed an invalid major version number: %d, should be in range [%d, %d]" % (self.major_version, _minv, _maxv))
        
        self.constant_pool = parse_constant_pool(self, br)
        self.access_flags = ClassAccessFlags(br.read_int(2, signed=False))

        # Check that we don't have any module or package objects in the constant pool unless the ACC_MODULE flag is set
        if any(isinstance(cp, (CPModuleInfo, CPPackageInfo)) for cp in self.constant_pool) and not self.access_flags.ACC_MODULE:
            raise ValueError("Found a %s object in the constant pool, but the ACC_MODULE flag isn't set" 
                              % repr(type([cp for cp in self.constant_pool if isinstance(cp, (CPModuleInfo, CPPackageInfo))][0]).__name__))
        
        this_class_index = br.read_int(2, signed=False)
        self.this_class = _get_check_pool_object(self, this_class_index, self.constant_pool, CPClassInfo, 'this_class_index')

        super_class_index = br.read_int(2, signed=False)
        if super_class_index != 0:  # Have to check if this is the Object class
            self.super_class = _get_check_pool_object(self, super_class_index, self.constant_pool, CPClassInfo, 'super_class_index')
        
        interfaces_count = br.read_int(2, signed=False)
        self.interfaces = []
        for i in range(interfaces_count):
            next_interface_index = br.read_int(2, signed=False)
            self.interfaces.append(_get_check_pool_object(self, next_interface_index, self.constant_pool, CPClassInfo, 
                                                          '%d-th index in the interfaces list' % i))
        
        self.fields = parse_fields(br, self)
        self.methods = parse_methods(br, self)
        self.attributes = parse_attributes(br, self)

        # TODO:
        # Check DynamicInfo bootstrap_method_attr_index values are valid
        # Never checked access flags and their effects on fields or methods
        # Never finished parsing attributes
        # Check if a CPClassInfo corresponds to a class or an interface (different possible values for their attributes)

