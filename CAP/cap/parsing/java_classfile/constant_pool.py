"""Contains information for parsing constant pool objects"""

import re
import traceback
from enum import Enum
from typing import Type, Union, Optional
from parsing.java_classfile.classfile_utils import ByteReader
import parsing.java_classfile.parse_java_classfile as JavaClass


class CPTag(Enum):
    """Enum for contant pool tags"""
    FILL_SPACE = -2
    EMPTY = -1
    Utf8 = 1
    Integer = 3
    Float = 4
    Long = 5
    Double = 6
    Class = 7
    String = 8
    Fieldref = 9
    Methodref = 10
    InterfaceMethodref = 11
    NameAndType = 12
    MethodHandle = 15
    MethodType = 16
    Dynamic = 17
    InvokeDynamic = 18
    Module = 19
    Package = 20


class ParameterType(Enum):
    """Enum for the various types of parameters available for methods"""
    BYTE = 'B'      # signed byte
    CHAR = 'C'      # Unicode character code point in the Basic Multilingual Plane, encoded with UTF-16
    DOUBLE = 'D'    # double-precision floating-point value
    FLOAT = 'F'     # single-precision floating-point value
    INT = 'I'       # integer
    LONG = 'J'      # long integer
    SHORT = 'S'     # signed short
    BOOLEAN = 'Z'   # true or false
    OBJECT = 'L'    # an object
    ARRAY = '['     # an array of subparameters
    VOID = 'V'      # void


class MethodHandleKind(Enum):
    """Enum for the various kinds of MethodHandle cpinfo objectes there are
    
    There are various groupings of method handle kinds:

        * 1, 2, 3, 4 (getField, getStatic, putField, putStatic) - Represents a field for which a method handle is to be
          constructed. Must point to a CPFieldrefInfo object
        * 5, 8 (invokeVirtual, newinvokeSpecial) - Represents a class's method or constructor for which a method handle 
          is to be constructed. Must point to a CPMethodrefInfo object. Cannot be <init> or <clinit> if 5 (invokeVirtual),
          and MUST be <init> if 8 (newinvokeSpecial)
        * 6, 7 (invokeStatic, invokeSpecial) - Represents a class's or interface's method for which a method handle is 
          to be constructed. Must point to a CPMethodrefInfo object (if the classfile's version is < 52), or either
          a CPMethodrefInfo or CPInterfaceMethodrefInfo object. Cannot be <init> or <clinit>
        * 9 (invokeInterface) - represents an interface's method for which a method handle is to be constructed. Must
          point to a CPInterfaceMethodrefInfo object. Cannot be <init> or <clinit>
    
    """
    GET_FIELD = 1           
    GET_STATIC = 2
    PUT_FIELD = 3
    PUT_STATIC = 4

    INVOKE_VIRTUAL = 5
    NEW_INVOKE_SPECIAL = 8

    INVOKE_STATIC = 6       
    INVOKE_SPECIAL = 7

    INVOKE_INTERFACE = 9


_CP_INFO_MAP: 'dict[CPTag, Type[CPInfo]]' = {}
"""Dictionary mapping CPTag enums to their associated classes"""


class _CPInfoMetaMapper(type):
    """Metaclass that keeps track of new constant pool information objects being created, and adds them to the mapping"""
    def __new__(cls, name, bases, dct):
        global _CP_INFO_MAP

        ret: 'Type[CPInfo]' = super().__new__(cls, name, bases, dct)
        uninit_types = ['CPInfo', 'CPRefInfo']

        if not hasattr(ret, 'tag') or (ret.tag is None and ret.__name__ not in uninit_types):
            raise AttributeError("Could not find 'tag' attribute on class of type: %s" % repr(ret.__name__))
        if not hasattr(ret, 'loadable') or (ret.loadable is None and ret.__name__ not in uninit_types):
            raise AttributeError("Could not find 'loadable' attribute on class of type: %s" % repr(ret.__name__))
        if not hasattr(ret, 'pool_size') or (ret.pool_size is None and ret.__name__ not in uninit_types):
            raise AttributeError("Could not find 'pool_size' attribute on class of type: %s" % repr(ret.__name__))
        
        if ret.__name__ not in uninit_types:
            if ret.tag in _CP_INFO_MAP:
                raise ValueError("Class %s has a tag that already exists in the _CP_INFO_MAP: %s" % (repr(ret.__name__), ret.tag.name))
            _CP_INFO_MAP[ret.tag] = ret
        
        def _check_method_run(kwarg):
            def wrap(func):
                def new_func(self, *args, **kwargs):
                    if not getattr(self, kwarg):
                        func(self, *args, **kwargs)
                    setattr(self, kwarg, True)
                return new_func
            return wrap
        
        ret._parse_info = _check_method_run('_parsed')(ret._parse_info)
        ret._post_parse_finalize = _check_method_run('_finalized')(ret._post_parse_finalize)
        
        
        return ret


class CPInfo(metaclass=_CPInfoMetaMapper):
    """A generic class containing information in the constant pool
    
    Parameters
    ----------
    index: `int`
        The integer index in the constant pool of this object (remember, these are offset by 1 and, if 0, then this
        should be a CPEmpty() object)
    br: `ByteReader`
        The bytereader to reads bytes from for this constant pool information object
    """

    tag: 'CPInfo' = None
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = None
    """Whether or not this class is loadable onto the JVM stack"""

    pool_size: 'int' = 1
    """The number of indices in the constant pool that this takes up"""

    index: 'int'
    """The integer index in the constant pool of this object"""

    _parsed: 'bool' = False
    _finalized: 'bool' = False

    def __init__(self, index: 'int', br: 'ByteReader'):
        self.index = index
        self.br = br
    
    def _parse_info(self):
        """Parses the info for this object from the ByteReader. Should be overridden for every subclass."""
        raise NotImplementedError("Did not implement _parse_info() method for object of type: %s" % repr(type(self).__name__))
    
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        """Called on all pool objects in order once all of them have been parsed"""
        pass

    def __str__(self):
        return repr(self)


class CPEmpty(CPInfo):
    """The initial empty CPInfo object in the constant pool"""

    tag: 'CPTag' = CPTag.EMPTY
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = False
    """Whether or not this class is loadable onto the JVM stack"""

    def _parse_info(self): ...
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'): ...

    def __repr__(self):
        return "%s()" % repr(type(self).__name__)


class CPFillSpace(CPInfo):
    """Used to signify empty values in the constant pool that are filled because of previous constants taking up more
       space (EG: longs/doubles)"""

    tag: 'CPTag' = CPTag.FILL_SPACE
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = False
    """Whether or not this class is loadable onto the JVM stack"""

    def _parse_info(self): ...
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'): ...

    def __repr__(self):
        return "%s()" % repr(type(self).__name__)


class CPClassInfo(CPInfo):
    """Represents a class or an interface
    
    Struct Format:

    CPClassInfo {
        u1 tag,
        u2 name_index,
    }
    """

    tag: 'CPTag' = CPTag.Class
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = True
    """Whether or not this class is loadable onto the JVM stack"""

    name_index: 'int'
    """The index in the constant pool for the CPUtf8 structure representing a valid binary class or interface name 
       encoded in internal form (see uninternalize_name() method in this file for info on 'internal form')"""
    
    name: 'str'
    """The parsed utf8 string name, in internal form"""
    
    def _parse_info(self):
        self.name_index = self.br.read_int(2, signed=False)
    
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        self.name = _get_check_pool_object(self, self.name_index, constant_pool, CPUtf8Info, 'name_index', classfile).string
        try:
            _check_qualified_name(self.name)
        except:
            try:
                _check_field_descriptor(self.name)
            except:
                raise ValueError("Invalid class name: %s" % repr(self.name))

    def __repr__(self):
        return "%s(name=%s)" % (repr(type(self).__name__), repr(self.name))


class CPUtf8Info(CPInfo):
    """Represents a utf-8 string value

    Struct Format:

    CPUtf8Info {
        u1 tag,
        u2 length,
        u1 bytes[length],
    }

    NOTE: for the bytes[] array, no byte value may be 0, nor lie in the range [0xF0, 0xFF]
    
    Utf8 string information is stored in a slightly different format that normal Utf8 such that:

        1. There are never any embedded 1-byte null (0) characters, all of these are converted to 2-byte characters
        2. All 4-byte unicode representations are converted into a 6-byte format
        3. There is no trailing null byte
        4. All non-null ASCII characters can be represented in one byte
    
    They can be parsed like so:

        * Code points in the range '\\u0001' to '\\u007F' are represented by a single byte:

            x = |   0   |                       bits 6-0                        |
        
          The byte is simple the value of the character
        
        * The null code point ('\\u0000') and code points in the range '\\u0080' to '\\u07FF' are represented by a pair 
          of bytes x and y :

            x = |   1   |   1   |   0   |              bits 10-6                |
            y = |   1   |   0   |                   bits 5-0                    |


          The two bytes represent the code point with the value: ((x & 0x1f) << 6) + (y & 0x3f)

        * Code points in the range '\\u0800' to '\\uFFFF' are represented by 3 bytes x, y, and z :

            x = |   1   |   1   |   1   |   0   |          bits 15-12           |
            y = |   1   |   0   |                   bits 11-6                   |
            z = |   1   |   0   |                   bits 5-0                    |

          The three bytes represent the code point with the value: ((x & 0xf) << 12) + ((y & 0x3f) << 6) + (z & 0x3f)
        
        * Characters with code points above '\\uFFFF' (so-called supplementary characters) are represented by separately
          encoding the two surrogate code units of their UTF-16 representation. Each of the surrogate code units is 
          represented by three bytes. This means supplementary characters are represented by six bytes, u, v, w, x, y, and z:

            u = |   1   |   1   |   1   |   0   |   1   |   1   |   0   |   1   |
            v = |   1   |   0   |   1   |   0   |        (bits 20-16) - 1       |
            w = |   1   |   0   |                  bits 15-10                   |
            x = |   1   |   1   |   1   |   0   |   1   |   1   |   0   |   1   |
            y = |   1   |   0   |   1   |   1   |            bits 9-6           |
            w = |   1   |   0   |                   bits 5-0                    |

          The six bytes represent the code point with the value:
          0x10000 + ((v & 0x0f) << 16) + ((w & 0x3f) << 10) + ((y & 0x0f) << 6) + (z & 0x3f)
    """

    tag: 'CPTag' = CPTag.Utf8
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = False
    """Whether or not this class is loadable onto the JVM stack"""

    bytes: 'bytes'
    """The raw bytes in the classfile"""

    string: 'str'
    """The parsed utf8 string"""

    def _parse_info(self):
        length = self.br.read_int(2, signed=False)
        raw = self.br.read(length)

        # Check that no bytes lie in the range [0xF0, 0xFF]
        for b in raw:
            if 0xF0 <= b <= 0xFF or b == 0:
                raise ValueError("Found an invalid byte value 0x%x in raw utf8 bytes: %s" % (b, raw))

        self.string = ''

        i = 0
        def _check_len(v):
            if i + v >= len(raw):
                raise ValueError("Cannot get %d more bytes from index %d in bytes: %s" % (v, i, raw))
            return True

        while i < len(raw):
            # Starting with a 0 means only one byte
            if not raw[i] & 0b1000_0000:
                self.string += chr(raw[i])
                i += 1
            
            # Starting with a 110 means two bytes
            elif not 0b110_0_0000 ^ (raw[i] & 0b111_0_0000):
                _check_len(2)
                if 0b10_00_0000 ^ (raw[i + 1] & 0b11_00_0000):
                    raise ValueError("Second byte 0x%x of a 2-byte utf8 character doesn't start with '10' in bytes: %s" % (raw[i + 1], raw))
                self.string += chr(((raw[i] & 0x1f) << 6) + (raw[i + 1] & 0x3f))
                i += 2
            
            # Swapping order a bit to do the 6-byte value here
            # We detect it by checking:
            #       1. there are at least 6 more bytes
            #       2. the next three bytes match 1110_1101, 1010_bbbb, and 10bb_bbbb respectively
            #       3. the three bytes after also match those values
            # Theoretically, this could also correspond to two 3-byte values with unicode characters in the range 
            #   [U+D800, U+DBFF]. This, however, corresponds to characters reserved for the 'surrogate' range, which
            #   looks like they shouldn't occur? I'm not too certain on this, but I don't care about it being 100%
            #   accurate right now
            elif _check_len(6) and (not raw[i] ^ 0b1110_1101) and (not 0b1010_0000 ^ (raw[i + 1] & 0x1111_0000)) and (not 0b10_00_0000 ^ (raw[i + 2] & 0b1100_0000)) \
                and (not raw[i + 3] ^ 0b1110_1101) and (not 0b1010_0000 ^ (raw[i + 4] & 0x1111_0000)) and (not 0b10_00_0000 ^ (raw[i + 5] & 0b1100_0000)):
                _check_len(6)
                self.string += chr(0x10000 + ((raw[i + 1] & 0x0f) << 16) + ((raw[i + 2] & 0x3f) << 10) + ((raw[i + 4] & 0x0f) << 6) + (raw[i + 5] & 0x3f))
                i += 6

            # Finally, the 3-byte values
            elif not 0b1110_0000 ^ (raw[i] & 0b1111_0000):
                _check_len(3)
                if 0b10_00_0000 ^ (raw[i + 1] & 0b11_00_0000):
                    raise ValueError("Second byte 0x%x of a 3-byte utf8 character doesn't start with '10' in bytes: %s" % (raw[i + 1], raw))
                if 0b10_00_0000 ^ (raw[i + 2] & 0b11_00_0000):
                    raise ValueError("Third byte 0x%x of a 3-byte utf8 character doesn't start with '10' in bytes: %s" % (raw[i + 2], raw))
                self.string += chr(((raw[i] & 0xf) << 12) + ((raw[i + 1] & 0x3f) << 6) + (raw[i + 2] & 0x3f))
                i += 3
            
            # If we get here, then something has gone wrong
            else:
                raise ValueError("Could not parse a byte 0x%x, this should not be reachable!!! In bytes: %s" % (raw[i], raw))

    def __repr__(self):
        return "%s(string=%s)" % (repr(type(self).__name__), repr(self.string))


class CPStringInfo(CPInfo):
    """Represents a loadable string
    
    Struct Format:

    CPStringInfo {
        u1 tag,
        u2 string_index,
    }

    Contains only an index in the constant pool of a utf8 info object that contains the string
    """

    tag: 'CPTag' = CPTag.String
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = True
    """Whether or not this class is loadable onto the JVM stack"""

    string_index: 'int'
    """The index of the CPUtf8Info object containing the string information in the constant pool"""

    string: 'str'
    """The string value"""

    def _parse_info(self):
        self.string_index = self.br.read_int(2, signed=False)

    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        self.string = _get_check_pool_object(self, self.string_index, constant_pool, CPUtf8Info, 'string_index', classfile).string

    def __repr__(self):
        return "%s(string=%s)" % (repr(type(self).__name__), repr(self.string))


class CPIntegerInfo(CPInfo):
    """Represents a signed integer
    
    Struct Format:

    CPStringInfo {
        u1 tag,
        u4 bytes,
    }

    Contains raw big-endian bytes of the integer value
    """

    tag: 'CPTag' = CPTag.Integer
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = True
    """Whether or not this class is loadable onto the JVM stack"""

    value: 'int'
    """The integer value"""

    def _parse_info(self):
        self.value = self.br.read_int(4, signed=True)

    def __repr__(self):
        return "%s(value=%s)" % (repr(type(self).__name__), repr(self.value))


class CPFloatInfo(CPInfo):
    """Represents a float
    
    Struct Format:

    CPStringInfo {
        u1 tag,
        u4 bytes,
    }

    Contains raw big-endian bytes of the float value in IEEE 754 format
    """

    tag: 'CPTag' = CPTag.Float
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = True
    """Whether or not this class is loadable onto the JVM stack"""

    value: 'float'
    """The float value"""

    def _parse_info(self):
        self.value = self.br.read_float(4)

    def __repr__(self):
        return "%s(value=%s)" % (repr(type(self).__name__), repr(self.value))


class CPLongInfo(CPInfo):
    """Represents a signed long
    
    Struct Format:

    CPStringInfo {
        u1 tag,
        u4 high_bytes,
        u4 low_bytes,
    }

    Contains raw big-endian bytes of the long value
    """

    tag: 'CPTag' = CPTag.Long
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = True
    """Whether or not this class is loadable onto the JVM stack"""

    pool_size: 'int' = 2
    """The number of indices in the constant pool that this takes up. The JVM documentation says it best: "In retrospect,
       making 8-byte constants take two constant pool entries was a poor choice." """

    value: 'int'
    """The integer value"""

    def _parse_info(self):
        self.value = self.br.read_int(8, signed=True)

    def __repr__(self):
        return "%s(value=%s)" % (repr(type(self).__name__), repr(self.value))


class CPDoubleInfo(CPInfo):
    """Represents a double float
    
    Struct Format:

    CPStringInfo {
        u1 tag,
        u4 high_bytes,
        u4 low_bytes,
    }

    Contains raw big-endian bytes of the double float value in IEEE 754 format
    """

    tag: 'CPTag' = CPTag.Double
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = True
    """Whether or not this class is loadable onto the JVM stack"""

    pool_size: 'int' = 2
    """The number of indices in the constant pool that this takes up. The JVM documentation says it best: "In retrospect,
       making 8-byte constants take two constant pool entries was a poor choice." """

    value: 'float'
    """The float value"""

    def _parse_info(self):
        self.value = self.br.read_float(8)

    def __repr__(self):
        return "%s(value=%s)" % (repr(type(self).__name__), repr(self.value))


class CPNameAndTypeInfo(CPInfo):
    """Represents a field or method, without indicating which class or interface type it belongs to
    
    Struct Format:

    CPStringInfo {
        u1 tag,
        u2 name_index,
        u2 descriptor_index
    }

    name_index and descriptor_index must both be valid indices in the constant pool pointing to a CPUtf8Info object.
    name_index must point to an unqualified name, but not "<clinit>". descriptor_index must point to a valid field_descriptor
    or method_descriptor CPUtf8Info object

    See _check_unqualified_name(), _check_field_descriptor(), and _check_method_descriptor() for more info
    """

    tag: 'CPTag' = CPTag.NameAndType
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = False
    """Whether or not this class is loadable onto the JVM stack"""

    name_index: 'int'
    """The index in the constant pool for the name CPUtf8Info object"""

    name: 'str'
    """The parsed name"""

    descriptor_index: 'int'
    """The index in the constant pool for the descriptor CPUtf8Info object"""

    descriptor: 'str'
    """The parsed descriptor"""

    is_field_descriptor: 'bool' = False
    """True if this is a valid field descriptor, False otherwise. It is impossible for both this and self.is_method_descriptor
       to both be True. And, if they are both False, then an error should have been raised"""
    
    field_parameter: 'Union[MethodOrFieldParameterType, None]'
    """If this is a field_descriptor, then this will be the parameter type of this field. Otherwise will be None"""

    is_method_descriptor: 'bool' = False
    """True if this is a valid method descriptor, False otherwise. It is impossible for both this and self.is_field_descriptor
       to both be True. And, if they are both False, then an error should have been raised"""
    
    method_parameters: 'Union[list[MethodOrFieldParameterType], None]' = None
    """If this is a method_descriptor, then this will be a list (possibly empty) of all input parameters to the method.
       Otherwise will be None"""
    
    method_return: 'Union[MethodOrFieldParameterType, None]' = None
    """If this is a method_descriptor, then this will be the return type. Otherwise will be None"""

    def _parse_info(self):
        self.name_index = self.br.read_int(2, signed=False)
        self.descriptor_index = self.br.read_int(2, signed=False)
    
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        name_cp = _get_check_pool_object(self, self.name_index, constant_pool, CPUtf8Info, 'name_index', classfile)
        descriptor_cp = _get_check_pool_object(self, self.descriptor_index, constant_pool, CPUtf8Info, 'descriptor_index')
        
        self.name = name_cp.string
        _check_unqualified_name(self.name, message='Invalid CPNameAndTypeInfo name')
        if self.name == '<clinit>':
            raise ValueError("CPNameAndTypeInfo name value cannot be '<clinit>'")
        
        self.descriptor = descriptor_cp.string
        
        try:
            self.field_parameter = _check_field_descriptor(self.descriptor, message='Invalid CPNameAndTypeInfo descriptor')
            self.is_field_descriptor = True
        except Exception as e:
            e_traceback = traceback.format_exc()
            try:
                self.method_parameters, self.method_return = _check_method_descriptor(self.descriptor, message='Invalid CPNameAndTypeInfo descriptor')
                self.is_method_descriptor = True
            except Exception as e2:
                raise ValueError(("Invalid CPNameAndTypeInfo descriptor: %s\n\nFieldDescriptor Error Message: %s\nField "
                                  "Descriptor Error Traceback: %s\n\nMethodDescriptor Error Message: %s\nMethodDescriptor "
                                  "Error Traceback: %s") % (repr(self.descriptor), e, e_traceback, e2, traceback.format_exc()))

    def __repr__(self):
        return "%s(name=%s, descriptor=%s)" % (repr(type(self).__name__), repr(self.name), repr(self.descriptor))


class CPRefInfo(CPInfo):
    """Base class for the three 'ref' info objects: CPFieldrefInfo, CPMethodrefInfo, CPInterfaceMethodrefInfo
    
    Represents either a field or a method in a class or interface
    
    Struct Format:

    CPFieldrefInfo {
        u1 tag,
        u2 class_index,
        u2 name_and_type_index,
    }

    class_index is the index of the CPClassInfo object that this field belongs to. It can belong to either a class or
    an interface for a CPFieldrefInfo, a class for a CPMethodrefInfo, and an interface for a CPInterfaceMethodInfo object.
    name_and_type_index is the index of the CPNameAndTypeInfo object associated with this field which must contain a 
    field_descriptor if this is a CPFieldredInfo, and a method_descriptor otherwise.
    """

    loadable: 'bool' = False
    """Whether or not this class is loadable onto the JVM stack"""

    class_index: 'int'
    """The index in the constant pool of the CPClassInfo object that this belongs to"""

    class_info: 'CPClassInfo'
    """The CPClassInfo object that this field belongs to"""

    class_name: 'str'
    """The string name of this object's class, pulled from the class_info object"""

    name_and_type_index: 'int'
    """The index in the constant pool of the CPNameAndTypeInfo object associated with this object"""

    name_and_type_info: 'CPNameAndTypeInfo'
    """The CPNameAndTypeInfo object associated with this object"""

    name: 'str'
    """The string name of this object, pulled from the name_and_type_info"""

    descriptor: 'str'
    """The string descriptor of this object, pulled from the name_and_type_info"""

    def _parse_info(self):
        self.class_index = self.br.read_int(2, signed=False)
        self.name_and_type_index = self.br.read_int(2, signed=False)

    def __repr__(self):
        return "%s(name=%s, descriptor=%s)" % (repr(type(self).__name__), repr(self.name), repr(self.descriptor))


class CPFieldrefInfo(CPRefInfo):
    """Represents a field in a class or interface"""

    tag: 'CPTag' = CPTag.Fieldref
    """The tag enum for this contant pool information type"""
    
    field_parameter: 'MethodOrFieldParameterType'
    """The parameter type of this field"""
    
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        _ref_finalize(self, constant_pool, classfile)
        self.field_parameter = self.name_and_type_info.field_parameter


class CPMethodrefInfo(CPRefInfo):
    """Represents a method in a class"""

    tag: 'CPTag' = CPTag.Methodref
    """The tag enum for this contant pool information type"""
    
    method_parameters: 'list[MethodOrFieldParameterType]'
    """A list (possibly empty) of all input parameters to the method"""
    
    method_return: 'MethodOrFieldParameterType'
    """The method return type"""
    
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        _ref_finalize(self, constant_pool, classfile)

        if '<' in self.name:
            if self.name != '<init>':
                raise ValueError("%ss name contains the '<' character, but is not '<init>', instead is: %s"
                                 % (repr(type(self).__name__), repr(self.name)))
            if self.method_return.param_type != ParameterType.VOID:
                raise ValueError("%ss name is '<init>' but it has a non-void return type: %s"
                                 % (repr(type(self).__name__), repr(self.method_return.param_type.name)))


class CPInterfaceMethodrefInfo(CPMethodrefInfo):
    """Represents a method in an interface"""

    tag: 'CPTag' = CPTag.InterfaceMethodref
    """The tag enum for this contant pool information type"""
    
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        _ref_finalize(self, constant_pool, classfile)
        if '<' in self.name:
            raise ValueError("%ss name contains a '<', but this is an interface. Name: %s"
                             % (repr(type(self).__name__), repr(self.name)))


class CPMethodHandleInfo(CPInfo):
    """Represents a method handle
    
    Struct Format:

    CPMethodHandleInfo {
        u1 tag,
        u1 reference_kind,
        u2 reference_index,
    }

    reference_kind is an integer determining what type of method handle this is. reference_index is an index in the
    constant pool to one of the 'ref' cpinfo objects, depending on reference_kind. See MethodHandleKind enum for more
    information
    """

    tag: 'CPTag' = CPTag.MethodHandle
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = True
    """Whether or not this class is loadable onto the JVM stack"""

    reference_kind_int: 'int'
    """The integer reference kind for this method handle. See MethodHandleKind Enum for more info"""

    reference_kind: 'MethodHandleKind'
    """The reference kind for this method handle. See MethodHandleKind Enum for more info"""

    reference_index: 'int'
    """The index in the constant pool for the 'ref' object that this object uses"""

    reference: 'CPRefInfo'
    """The underlying 'ref' object that this object uses"""

    def _parse_info(self):
        self.reference_kind_int = self.br.read_int(1, signed=False)
        self.reference_index = self.br.read_int(2, signed=False)
    
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        if self.reference_kind_int not in MethodHandleKind._value2member_map_:
            raise ValueError("Unknown %s reference_kind value: %d" % (repr(type(self).__name__), self.reference_kind))
        self.reference_kind = MethodHandleKind._value2member_map_[self.reference_kind_int]

        self.reference = _get_check_pool_object(self, self.reference_index, constant_pool, CPRefInfo, 'reference_index', classfile)
        rks = 'reference_index (kind=%d)' % self.reference_kind
        
        # Check for fieldref values
        if self.reference_kind in [MethodHandleKind.GET_FIELD, MethodHandleKind.GET_STATIC, MethodHandleKind.PUT_FIELD, MethodHandleKind.PUT_STATIC]:
            _get_check_pool_object(self, self.reference_index, constant_pool, CPFieldrefInfo, rks)
        
        # Check for only method types
        elif self.reference_kind in [MethodHandleKind.INVOKE_VIRTUAL, MethodHandleKind.NEW_INVOKE_SPECIAL]:
            _get_check_pool_object(self, self.reference_index, constant_pool, CPMethodrefInfo, rks)
        
        # Check for only interface types
        elif self.reference_kind in [MethodHandleKind.INVOKE_INTERFACE]:
            _get_check_pool_object(self, self.reference_index, constant_pool, CPInterfaceMethodrefInfo, rks)
            
        # Check for what can be one or the other, depending on classfile version
        elif self.reference_kind in [MethodHandleKind.INVOKE_STATIC, MethodHandleKind.INVOKE_SPECIAL]:
            accepted = (CPMethodrefInfo,) if classfile.major_version < 52 else (CPMethodrefInfo, CPInterfaceMethodrefInfo)
            _get_check_pool_object(self, self.reference_index, constant_pool, accepted, rks + "(major version: %d)")

        # Check for not implemented
        else:
            raise NotImplementedError("Unimplemented reference_kind enum value in %s: %s" 
                                      % (repr(type(self).__name__), self.reference_kind.name))
        
        # Check that the name is never <init> nor <clinit>, unless we are MethodHandleKind.NEW_INVOKE_SPECIAL, in which
        #   case we must be <init>
        if self.reference_kind in [MethodHandleKind.NEW_INVOKE_SPECIAL]:
            if self.reference.name != "<init>":
                raise ValueError("%ss reference_kind is %s, but its reference's name is not '<init>': %s"
                                 % (repr(type(self).__name__), self.reference_kind.name, repr(self.reference.name)))
        elif self.reference.name in ['<init>', '<clinit>']:
            raise ValueError("%ss reference_kind is %s, but it has an invalid reference name: %s"
                             % (repr(type(self).__name__), self.reference_kind.name, repr(self.reference.name)))

    def __repr__(self):
        return "%s(reference_kind=%s, reference=%s)" % (repr(type(self).__name__), repr(self.reference_kind), repr(self.reference))


class CPMethodTypeInfo(CPInfo):
    """Represents a method type
    
    Struct Format:

    CPMethodTypeInfo {
        u1 tag,
        u2 descriptor_index
    }

    descriptor_index is an index in the constant pool table to a CPUtf8Info object representing a method descriptor
    """

    tag: 'CPTag' = CPTag.MethodType
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = True
    """Whether or not this class is loadable onto the JVM stack"""

    descriptor_index: 'int'
    """The index in the constant pool to a CPUtf8Info object representing the method descriptor for this method type"""

    descriptor: 'CPUtf8Info'
    """The CPUtf8Info obejct representing the method descriptor for this method type"""

    descriptor_string: 'str'
    """The parsed string representing the method descriptor for this method type"""

    method_parameters: 'list[MethodOrFieldParameterType]'
    """A list (possibly empty) of all input parameters to this method type"""
    
    method_return: 'MethodOrFieldParameterType'
    """The return type of this method type"""

    def _parse_info(self):
        self.descriptor_index = self.br.read_int(2, signed=False)
    
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        self.descriptor = _get_check_pool_object(self, self.descriptor_index, constant_pool, CPUtf8Info, 'descriptor_index', classfile)
        self.descriptor_string = self.descriptor.string
        self.method_parameters, self.method_return = _check_method_descriptor(self.descriptor_string)

    def __repr__(self):
        return "%s(descriptor_string=%s)" % (repr(type(self).__name__), repr(self.descriptor_string))


class CPModuleInfo(CPInfo):
    """Represents a module
    
    Struct Format:

    CPMethodTypeInfo {
        u1 tag,
        u2 name_index
    }

    name_index is an index in the constant pool table to a CPUtf8Info object representing a valid module name
    (see _check_module_name() method for more info).

    A CPModuleInfo structure is permitted only in the constant pool of a class file that declares a module, that is, a 
    ClassFile structure where the access_flags item has the ACC_MODULE flag set. This check is performed in the
    JavaClassfile() object after it has finished parsing the constant_pool and access_flags structs.
    """

    tag: 'CPTag' = CPTag.Module
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = False
    """Whether or not this class is loadable onto the JVM stack"""

    name_index: 'int'
    """index in the constant pool table to a CPUtf8Info object representing a valid module name"""

    name_info: 'CPUtf8Info'
    """The CPUtf8Info object representing a valid module name"""

    name: 'str'
    """The module name"""

    cleaned_name: 'str'
    """The module name with any of the escaped characters in the raw name replaced with their respective values"""

    def _parse_info(self):
        self.name_index = self.br.read_int(2, signed=False)
    
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        self.name_info = _get_check_pool_object(self, self.name_index, constant_pool, CPUtf8Info, 'name_index', classfile)
        self.name = self.name_info.string
        self.cleaned_name = clean_module_name(self.name)

    def __repr__(self):
        return "%s(cleaned_name=%s)" % (repr(type(self).__name__), repr(self.cleaned_name))


class CPPackageInfo(CPInfo):
    """Represents a package
    
    Struct Format:

    CPPackageInfo {
        u1 tag,
        u2 name_index
    }

    name_index is an index in the constant pool table to a CPUtf8Info object representing a valid package name
    (see _check_package_name() method for more info).

    A CPPackageInfo structure is permitted only in the constant pool of a class file that declares a module, that is, a 
    ClassFile structure where the access_flags item has the ACC_MODULE flag set. This check is performed in the
    JavaClassfile() object after it has finished parsing the constant_pool and access_flags structs.
    """

    tag: 'CPTag' = CPTag.Package
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = False
    """Whether or not this class is loadable onto the JVM stack"""

    name_index: 'int'
    """index in the constant pool table to a CPUtf8Info object representing a valid package name"""

    name_info: 'CPUtf8Info'
    """The CPUtf8Info object representing a valid package name"""

    name: 'str'
    """The package name"""

    cleaned_name: 'str'
    """The package name with any of the escaped characters in the raw name replaced with their respective values, and
       with all forward slashes '/'s converted back into periods '.'s"""

    def _parse_info(self):
        self.name_index = self.br.read_int(2, signed=False)
    
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        self.name_info = _get_check_pool_object(self, self.name_index, constant_pool, CPUtf8Info, 'name_info', classfile)
        self.name = self.name_info.string
        self.cleaned_name = clean_package_name(self.name)

    def __repr__(self):
        return "%s(cleaned_name=%s)" % (repr(type(self).__name__), repr(self.cleaned_name))


class CPDynamicInfo(CPInfo):
    """Represents a dynamic constant
    
    Some constants can be computed dynamically at runtime. See the java documentation for more info
    
    Struct Format:

    CPDynamicInfo {
        u1 tag,
        u2 bootstrap_method_attr_index,
        u2 name_and_type_index
    }

    bootstrap_method_attr_index is a valid index into the bootstrap_methods array of the bootstrap method table of this 
    classfile (the checks for this are done in JavaClassfile once both the constant_pool and the method table have been 
    loaded). name_and_type_index is a valid index into the constant pool of a CPNameAndTypeInfo object for a field_descriptor
    """

    tag: 'CPTag' = CPTag.Dynamic
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = True
    """Whether or not this class is loadable onto the JVM stack"""

    bootstrap_method_attr_index: 'int'
    """A valid index into the bootstrap_methods array of the bootstrap method table of this classfile"""

    name_and_type_index: 'int'
    """A valid index into the constant pool of a CPNameAndTypeInfo object for a field_descriptor"""

    name_and_type_info: 'CPNameAndTypeInfo'
    """The CPNameAndTypeInfo object for a field_descriptor for this constant"""

    name: 'str'
    """The parsed name from the CPNameAndTypeInfo object"""

    descriptor: 'str'
    """The field_descriptor string"""
    
    field_parameter: 'MethodOrFieldParameterType'
    """The parameter type of this field"""

    def _parse_info(self):
        self.bootstrap_method_attr_index = self.br.read_int(2, signed=False)
        self.name_and_type_index = self.br.read_int(2, signed=False)
    
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        self.name_and_type_info = _get_check_pool_object(self, self.name_and_type_index, constant_pool, CPNameAndTypeInfo,
                                                         "name_and_type_index", classfile)

        if not self.name_and_type_info.is_field_descriptor:
            raise ValueError("%ss name_and_type_info points to a %s object that is NOT a field_descriptor"
                             % (repr(type(self).__name__), repr(CPNameAndTypeInfo.__name__)))

        self.field_parameter = self.name_and_type_info.field_parameter

    def __repr__(self):
        return "%s(name_and_type_info=%s, bootstrap_index=%d)" % (repr(type(self).__name__), repr(self.name_and_type_info), 
                                                                  repr(self.bootstrap_method_attr_index))


class CPInvokeDynamicInfo(CPInfo):
    """Represents a dynamic call site
    
    Some constants can be computed dynamically at runtime. See the java documentation for more info
    
    Struct Format:

    CPInvokeDynamicInfo {
        u1 tag,
        u2 bootstrap_method_attr_index,
        u2 name_and_type_index
    }

    bootstrap_method_attr_index is a valid index into the bootstrap_methods array of the bootstrap method table of this 
    classfile (the checks for this are done in JavaClassfile once both the constant_pool and the method table have been 
    loaded). name_and_type_index is a valid index into the constant pool of a CPNameAndTypeInfo object for a method_descriptor
    """

    tag: 'CPTag' = CPTag.InvokeDynamic
    """The tag enum for this contant pool information type"""

    loadable: 'bool' = False
    """Whether or not this class is loadable onto the JVM stack"""

    bootstrap_method_attr_index: 'int'
    """A valid index into the bootstrap_methods array of the bootstrap method table of this classfile"""

    name_and_type_index: 'int'
    """A valid index into the constant pool of a CPNameAndTypeInfo object for a method_descriptor"""

    name_and_type_info: 'CPNameAndTypeInfo'
    """The CPNameAndTypeInfo object for a method_descriptor for this constant"""

    name: 'str'
    """The parsed name from the CPNameAndTypeInfo object"""

    descriptor: 'str'
    """The method_descriptor string"""

    method_parameters: 'list[MethodOrFieldParameterType]'
    """A list (possibly empty) of all input parameters to this method type"""
    
    method_return: 'MethodOrFieldParameterType'
    """The return type of this method type"""

    def _parse_info(self):
        self.bootstrap_method_attr_index = self.br.read_int(2, signed=False)
        self.name_and_type_index = self.br.read_int(2, signed=False)
    
    def _post_parse_finalize(self, constant_pool: 'list[CPInfo]', classfile: 'JavaClass.JavaClassfile'):
        self.name_and_type_info = _get_check_pool_object(self, self.name_and_type_index, constant_pool, CPNameAndTypeInfo, 
                                                         'name_and_type_index', classfile)

        if not self.name_and_type_info.is_method_descriptor:
            raise ValueError("%ss name_and_type_info points to a %s object that is NOT a method_descriptor"
                             % (repr(type(self).__name__), repr(CPNameAndTypeInfo.__name__)))

        self.method_parameters, self.method_return = self.name_and_type_info.method_parameters, self.name_and_type_info.method_return

    def __repr__(self):
        return "%s(name_and_type_info=%s, bootstrap_index=%d)" % (repr(type(self).__name__), repr(self.name_and_type_info), 
                                                                  repr(self.bootstrap_method_attr_index))


def _ref_finalize(self: 'Union[CPFieldrefInfo, CPMethodrefInfo, CPInterfaceMethodrefInfo]', constant_pool: 'list[CPInfo]', 
                  classfile: 'JavaClass.JavaClassfile'):
    """Does a bunch of finalization stuff that is used in the various 'ref' CPInfo objects"""
    self.class_info = _get_check_pool_object(self, self.class_index, constant_pool, CPClassInfo, 'class_index', classfile)
    self.name_and_type_info = _get_check_pool_object(self, self.name_and_type_index, constant_pool, CPNameAndTypeInfo, 
                                                     'name_and_type_index', classfile)
   
    if isinstance(self, CPFieldrefInfo) and not self.name_and_type_info.is_field_descriptor:
        raise ValueError("%ss name_and_type_info references something that is not a field_descriptor" % repr(type(self).__name__))
    elif isinstance(self, (CPMethodrefInfo, CPInterfaceMethodrefInfo)) and not self.name_and_type_info.is_method_descriptor:
        raise ValueError("%ss name_and_type_info references something that is not a method_descriptor" % repr(type(self).__name__))

    self.class_name, self.name, self.descriptor = self.class_info.name, self.name_and_type_info.name, self.name_and_type_info.descriptor

    if self.name_and_type_info.is_method_descriptor:
        self.method_parameters, self.method_return = self.name_and_type_info.method_parameters, self.name_and_type_info.method_return


def _check_valid_pool_idx(obj, idx, constant_pool, message='Invalid index'):
    """Checks that the given index is valid in the given constant pool, raising an error if not"""
    if idx <= 0 or idx >= len(constant_pool):
        raise ValueError("Invalid constant pool index %d parsed for object %s. Pool size: %d. Message: %s" 
                         % (idx, repr(type(obj).__name__), len(constant_pool), message))
    if isinstance(constant_pool[idx], CPFillSpace):
        raise ValueError("Constant pool index points to a filled space - %d parsed for object %s. Pool size: %d. Message: %s" 
                         % (idx, repr(type(obj).__name__), len(constant_pool), message))


def _get_check_pool_object(obj, idx: 'int', constant_pool: 'list[CPInfo]', obj_type: 'Union[type, tuple[type]]', 
                           attr_name: 'str', classfile: 'Optional[JavaClass.JavaClassfile]' = None):
    """Checks the given index is a valid index, and that the object at the location is the correc type
    
    If you pass the classfile, then _post_parse_finalize() will be called
    """
    _check_valid_pool_idx(obj, idx, constant_pool, message='Invalid %s index' % attr_name)
    ret = constant_pool[idx]
    if not isinstance(ret, obj_type):
        raise ValueError("%ss %s points to an object of type %s, should be %s" 
                            % (repr(type(obj).__name__), attr_name, repr(type(ret).__name__), 
                               repr(obj_type.__name__) if isinstance(obj_type, type) else [repr(t.__name__) for t in obj_type]))
    
    if classfile is not None:
        ret._post_parse_finalize(constant_pool, classfile)
    return ret


def _cp_read_next(index: 'int', br: 'ByteReader') -> 'CPInfo':
    """Reads the next constant pool info object from the given bytereader"""
    tag_int = br.read_int(1, signed=False)
    if tag_int not in CPTag._value2member_map_:
        raise ValueError("Unknown CPInfo tag int: %d" % tag_int)
    
    ret: 'CPInfo' = _CP_INFO_MAP[CPTag._value2member_map_[tag_int]](index, br)
    ret._parse_info()
    return ret


def uninternalize_name(name: str):
    """Converts the string internal name representation into an uninternalized version
    
    Class and interface names are essentially just multiple 'unqualified' names (see _check_unqualified_name() method 
    for more info) separated by a '.' period. This is their 'uninternalized' version, however and 'internalized' version
    is used inside the CPUtf8Info objects which replaces all '.'s with '/'s. This method just converts those values back.
    """
    return name.replace('/', '.')


def _check_qualified_name(name: str):
    """Checks that the given name is a valid 'qualified' name, raising an error otherwise
    
    A name is a valid 'qualified' name if it is either a valid unqualified name (see _check_unqualifed_name() method
    for more info), or if all substrings of name when split on forward slash '/' are valid unqualified names
    """
    for s in name.split('/'):
        _check_unqualified_name(s, message="Found as substring in qualified name: %s" % repr(name))


def _check_unqualified_name(name: str, valid_angles=('<init>', '<clinit>'), message: str = 'Invalid character used'):
    """Checks that the given name is a valid 'unqualified' name, raising an error otherwise
    
    Unqualified names can use any character from the unicode character codeset except for the characters: '.;[/'.
    They must be at least one character long. And, with the exception of the names '<init>' and '<clinit', they may
    not contain the characters '<' or '>'.
    """
    _bad_chars = '.;[/'

    if len(name) == 0:
        raise ValueError("Invaild unqualified name due to it being the empty string ''. Message: %s" % message)

    for i, c in enumerate(name):
        if c in _bad_chars or (c in '<>' and name not in valid_angles):
            raise ValueError("Invaild unqualified name due to character %s at position %d: %s. Message: %s" 
                             % (repr(c), i, repr(name), message))


def _check_module_name(name: str, message: str = 'Invalid module name'):
    """Checks that the given name is a valid module name
    
    Module names:

        * must be at least one character
        * must not include any unicode value in the range ['\\u0000' to '\\u001F']
        * are NOT converted to internal representation. That is, all periods '.' are NOT converted to '/'s and are left alone
        * have the ascii backslash '\\' used as an escape character, and may not appear unless immediately followed by
          one of: [':', '@', '\\']
        * cannot have the characters ':' or '@' unless immediately preceeded by the escape character '\\'
    
    """
    if len(name) == 0:
        raise ValueError("Module name cannot be the empty string ''. Message: %s" % message)
    
    if any(ord(c) < 0x1F for c in name):
        raise ValueError(("Module name cannot contain any unicode value in the range ['\\u0000' to '\\u001F'], found "
                         "'0x%x' in name: %s. Message: %s") % ([c for c in name if ord(c) < 0x1F][0], repr(name), message))
    
    matches = re.findall(r'(?:^|[^\\])[\\:@]')
    if len(matches) > 0:
        raise ValueError("Found unescaped special character %s in name: %s. Message: %s" % (matches[0], repr(name), message))


def clean_module_name(name: str, message: 'str' = 'Invalid module name'):
    """Cleans the module name, un-escaping all of the escaped special characters"""
    _check_module_name(name, message=message)
    return name.replace('\\\\', '\\').replace('\\@', '@').replace('\\:', ':')


def _check_package_name(name: str, message: str = 'Invalid package name'):
    """Checks that the given name is a valid package name
    
    Package names seem to be the same as module names, except that they ARE stored in 'internal' representation. That is,
    all periods '.' have been replaced with forward slashes '/'
    """
    _check_module_name(name, message=message)
    if '.' in name:
        raise ValueError("Found period '.' character in package name: %s. Message: %s" % (repr(name), message))


def clean_package_name(name: str, message: 'str' = 'Invalid package name'):
    """Cleans the package name, un-escaping all of the escaped special characters as well as replacing all '/' with '.'"""
    _check_package_name(name, message=message)
    return clean_module_name(name).replace('/', '.')


_RE_FIELD_DESCRIPTOR = r'\[*(?:[BCDFIJSZ]|L(?:[^.;[]*);)'
def _check_field_descriptor(name: str, message: str = 'Invalid field descriptor'):
    """Checks that the given name is a valid field descriptor
    
    A field descriptor represents the type of a class, instance, or local variable. They can be a:

        * BaseType - single character, one of 'BCDFIJSZ'. The values correspond to different base object types:

            BaseType    Object Type         Interpretation
                B           byte            signed byte
                C           char            Unicode character code point in the Basic Multilingual Plane, encoded with UTF-16
                D           double          double-precision floating-point value
                F           float           single-precision floating-point value
                I           int             integer
                J           long            long integer
                S           short           signed short
                Z           boolean         true or false
        
        * ObjectType - will be the string "L{classname};" where {classname} represents a binary class or interface name 
          encoded in internal form. See uninternalize_class_name() for more info. Ex: "Ljava/lang/object";
        * ArrayType - will be the string "[{field_type}" where {field_type} can be any other field descriptor, including
          other arrays. Ex: "[I", or "[[[D"
        
    Field descriptors representing an array type can be at most 255 dimensions large.

    Returns:
        MethodOrFieldParameterType: the parameter type
    """
    if re.fullmatch(_RE_FIELD_DESCRIPTOR, name) is None:
        raise ValueError("Invalid field descriptor name: %s. Message: %s" % (repr(name), message))
    if name.count('[') > 255:
        raise ValueError("Invalid field descriptor name (too many array dimensions, %d, when max is 255): %s. Message: %s" 
                         % (name.count('['), repr(name), message))
    
    return MethodOrFieldParameterType(name, is_return=False)


_RE_METHOD_DESCRIPTOR = r'\((?:{fd})*\)(?:{fd}|V)'.format(fd=_RE_FIELD_DESCRIPTOR)
_RE_METHOD_ITER = r'(?P<open>\()|(?P<param>{fd})|(?P<close>\))|(?P<void>V)|(?P<mismatch>.)'.format(fd=_RE_FIELD_DESCRIPTOR)
def _check_method_descriptor(name: str, message: str = 'Invalid method descriptor'):
    """Checks that the given name is a valid method descriptor
    
    A method descriptor contains zero or more parameter descriptors, representing the types of parameters that the method
    takes, and a return descriptor, representing the type of the value (if any) that the method returns. They will
    take the form:

    "({parameters}){return_type}"

    That is, everything inside the parentheses () are the argument types in order, and immediately after comes one
    return type. {parameters} can be zero or more field_descriptors one after another (see _check_field_descriptor() for
    more information). {return_type} can either be a field_descriptor, or the void descriptor 'V'

    Method descriptors can be at most 255 'types' long, where a long/double type counts as 2, and everything else counts
    as 1.

    Returns:
        Tuple[List[ParameterType], ReturnType]: a 2-tuple of (parameters, return_type), where parameters is a list of
            ParameterType objects and return_type is a ReturnType object
    """
    if re.fullmatch(_RE_METHOD_DESCRIPTOR, name) is None:
        raise ValueError("Invalid method descriptor name: %s. Message: %s" % (repr(name), message))
    
    count = 0
    parameters, return_type = [], None
    on_return = False
    for mo in re.finditer(_RE_METHOD_ITER, name):
        tn, token = mo.lastgroup, mo.group()

        if tn in ['open']:
            continue
        elif tn in ['close']:
            on_return = True
            continue
        elif tn in ['mismatch']:
            raise ValueError("Found a token mismatch in method descriptor iterator with token %s in name: %s. Message: %s" 
                             % (repr(token), repr(name), message))
        elif tn not in ['void', 'param']:
            raise NotImplementedError(tn)
        
        # update the count
        count += 2 if token in ['D', 'J'] else 1

        # Add this token to either the parameter list or as the return type
        if on_return:
            if return_type is not None:
                raise ValueError("Found multiple return_types (%s, %s) in name: %s. Message: %s"
                                 % (repr(return_type.value), repr(token), repr(name), message))
            return_type = MethodOrFieldParameterType(token, is_return=True)
        else:
            parameters.append(MethodOrFieldParameterType(token, is_return=False))
            
    if count > 255:
        raise ValueError("Found too many parameters in method descriptor (%d > 255 max) in name: %s. Message: %s" 
                         % (count, repr(name), message))
    if return_type is None:
        raise ValueError("Did not find a return type in name: %s. Message: %s" % (repr(name), message))
    
    return parameters, return_type


class MethodOrFieldParameterType:
    """A single parameter or return type of a method
    
    Parameters
    ----------
    string: `str`
        the string value for this object
    is_return: `bool`
        if True, then this is the return parameter of a method
    """

    string: 'str'
    """The string value of this parameter"""

    is_return: 'bool'
    """True if this is a return parameter of a method, False if an input parameter"""

    param_type: 'ParameterType'
    """The ParameterType enum value of this method parameter"""

    obj_classname: 'Union[str, None]' = None
    """The internalized classname if this is an object type or an array of objects, or None otherwise"""

    array_depth: 'int' = 0
    """The depth of the array if this is an array, or 0 if not an array"""

    array_type: 'Union[ParameterType, None]' = None
    """A ParameterType of the object type of this array if this object is an array, or None otherwise"""

    def __init__(self, string, is_return=False):
        self.string = string
        self.is_return = is_return

        if string[0] not in ParameterType._value2member_map_:
            raise ValueError("Unknown %s string %s, doesn't exist in ParameterType" % (repr(type(self).__name__), repr(string)))
        self.param_type = ParameterType._value2member_map_[string[0]]

        if self.param_type == ParameterType.OBJECT:
            self.obj_classname = self.string[1:-1]
        elif self.param_type == ParameterType.ARRAY:
            self.array_depth = self.string.count('[')
            
            array_type_string = self.string.replace('[', '')
            if array_type_string[0] not in ParameterType._value2member_map_:
                raise ValueError("Unknown %s array string %s, doesn't exist in ParameterType. Full string: %s"
                                  % (repr(type(self).__name__), repr(array_type_string), repr(string)))
            
            self.array_type = ParameterType._value2member_map_[array_type_string[0]]
            if self.array_type == ParameterType.OBJECT:
                self.obj_classname = array_type_string[1:-1]
            elif self.array_type == ParameterType.VOID and not self.is_return:
                raise ValueError("%s has an array_type of VOID, but is not a return parameter" % repr(type(self).__name__))
        elif self.param_type == ParameterType.VOID and not self.is_return:
            raise ValueError("%s has a param_type of VOID, but is not a return parameter" % repr(type(self).__name__))
    

def parse_constant_pool(classfile: 'JavaClass.JavaClassfile', br: 'ByteReader') -> 'list[CPInfo]':
    """Parses the constant pool from the given byte reader
    
    Assumes the constant_pool_length has not yet been parsed

    Args:
        classfile (JavaClass.JavaClassfile): the JavaClassfile object we are adding these to. Needed for some info like
            classfile version, the ByteReader, etc.
        br (ByteReader): the byte reader currently reading bytes for this classfile
    """
    pool_count = br.read_int(2, signed=False)
    if pool_count == 0:
        raise ValueError("Parsed a zero contant_pool_count")
    
    constant_pool: 'list[CPInfo]' = []
    constant_pool.append(CPEmpty(0, br))

    i = 1
    while i < pool_count:
        constant_pool.append(_cp_read_next(i, br))
        if constant_pool[-1].pool_size > 1:
            constant_pool += [CPFillSpace(i, br)] * (constant_pool[-1].pool_size - 1)
        i += constant_pool[-1].pool_size
    
    for i in range(pool_count):
        constant_pool[i]._post_parse_finalize(constant_pool, classfile)
    
    return constant_pool

