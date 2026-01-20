"""Parsing the attribute info objects in a java classfile"""
from enum import Enum
from parsing.java_classfile.constant_pool import *
from parsing.java_classfile.constant_pool import _get_check_pool_object
from parsing.java_classfile.classfile_utils import ByteReader
import parsing.java_classfile.parse_java_classfile as JavaClass


class AttributeType(Enum):
    """The type of attribute and its name (or the empty string for the 'UNKNOWN' attribute)"""

    CONSTANT_VALUE = 'ConstantValue'
    UNKNOWN = ''


class AttributeInfo:
    """Base class for attribute information objects
    
    All attributes have the base structure:

    AttributeInfo {
        u2 attribute_name_index,
        u4 attribute_length,
        u1 info[attribute_length],
    }

    attribute_name_index is an index in the constant pool of a CPUtf8Info object containing the name of this attribute.
    There are a list of known attribute names in the AttributeType enum, however it is possible for compilers to emit
    new attributes which the JVM would subsequently ignore.

    attribute_length is the number of bytes making up the 'info' object (not including the 6 bytes of attribute_name_index
    and attribute_length)

    Parameters
    ----------
    index: `int`
        he index in the currently-being-parsed attribute array for this object
    br: `ByteReader`
        The bytereader to read the bytes for this attribute from
    classfile: `JavaClass.JavaClassfile`
        The JavaClassfile object that this attribute is being loaded in
    """

    index: 'int'
    """The index in the currently-being-parsed attribute array for this object"""

    attribute_type: 'AttributeType'
    """The type of attribute this is"""

    attribute_name_index: 'int'
    """An index in the constant pool of a CPUtf8Info object containing the name of this attribute"""

    attribute_name: 'str'
    """The name of this attribute"""

    attribute_bytes: 'bytes'
    """The bytes making up all of the data for this attribute"""

    def __init__(self, index: 'int', br: 'ByteReader', classfile: 'JavaClass.JavaClassfile'):
        self.index = index
        self.attribute_name_index = br.read_int(2, signed=False)
        self.attribute_name = _get_check_pool_object(self, self.attribute_name_index, classfile.constant_pool, 
                                                     CPUtf8Info, 'attribute_name_index')
        
        self.attribute_type = AttributeType.UNKNOWN if self.attribute_name not in AttributeType._value2member_map_ else\
                                AttributeType._value2member_map_[self.attribute_name]
        
        num_bytes = br.read_int(4, signed=False)
        self.attribute_bytes = br.read(num_bytes)
        temp_br = ByteReader(self.attribute_bytes)
        self._parse_bytes(temp_br, classfile)

        # Not doing this check until I implement all fo the Attributes
        #if not temp_br.eof:
        #    raise ValueError("%ss _parse_bytes method left bytes in the ByteReader" % repr(type(self).__name__))
    
    def _parse_bytes(self, br: 'ByteReader', classfile: 'JavaClass.JavaClassfile'):
        """br is a bytereader built on only this attribute's bytes, separate from the classfile one"""
        pass


class AttributeConstantValue:
    """A constant value in a field
    
    struct {
        u2 constantvalue_index
    }

    Allowable types: Integer, Float, Long, Double, String
    """

    constant_value_index: 'int'
    """The integer index in the constant pool of this constant"""

    constant_value: 'CPInfo'
    """The CPInfo object corresponding to this constant"""

    def _parse_bytes(self, br: 'ByteReader', classfile: 'JavaClass.JavaClassfile'):
        """br is a bytereader built on only this attribute's bytes, separate from the classfile one"""
        _cp_types = [CPIntegerInfo, CPFloatInfo, CPLongInfo, CPDoubleInfo, CPStringInfo]
        self.constant_value_index = br.read_int(2, signed=False)
        self.constant_value = _get_check_pool_object(self, self.constant_value_index, classfile.constant_pool, _cp_types, 'constant_value')


class AttributeCode:
    """A section of code information
    
    struct {
        u2 max_stack;
        u2 max_locals;
        u4 code_length;
        u1 code[code_length];
        u2 exception_table_length;
        exc_info exception_table[exception_table_length];
        u2 attributes_count;
        attribute_info attributes[attributes_count];
    }

    exc_info { 
        u2 start_pc;
        u2 end_pc;
        u2 handler_pc;
        u2 catch_type;
    }
    """
    
    max_stack: 'int'
    """"""

    max_locals: 'int'
    """"""

    def _parse_bytes(self, br: 'ByteReader', classfile: 'JavaClass.JavaClassfile'):
        """br is a bytereader built on only this attribute's bytes, separate from the classfile one"""
        pass


def parse_attributes(br: 'ByteReader', classfile: 'JavaClass.JavaClassfile') -> 'list[AttributeInfo]':
    """Parse a list of attributes"""
    num_attributes = br.read_int(2, signed=False)
    return [AttributeInfo(i, br, classfile) for i in range(num_attributes)]
