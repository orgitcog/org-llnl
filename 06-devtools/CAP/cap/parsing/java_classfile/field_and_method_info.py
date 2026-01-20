"""Classes for the field_info and method_info objects in a classfile"""

from typing import Union
from parsing.java_classfile.access_flags import FieldAccessFlags, MethodAccessFlags
from parsing.java_classfile.constant_pool import MethodOrFieldParameterType, _check_unqualified_name, _check_field_descriptor, \
    _get_check_pool_object, _check_method_descriptor, CPUtf8Info
from parsing.java_classfile.classfile_utils import ByteReader
from parsing.java_classfile.attributes import AttributeInfo, parse_attributes
import parsing.java_classfile.parse_java_classfile as JavaClass


class FieldInfo:
    """Each field is described by a field_info structure.

    FieldInfo Struct {
        u2          access_flags,
        u2          name_index,
        u2          descriptor_index,
        u2          attributes_count,
        attribute   attributes[attributes_count]
    }
    
    No two fields in one class file may have the same name and descriptor.

    name_index and descriptor_index are indices into the constant pool for CPUtf8Info objects that represent a valid
    unqualified field name and field descriptor string respectively. See constant_pool._check_unqualified_name() for
    more info on unqualifed names and constant_pool._check_field_descriptor() for more info on field descriptor values

    attributes is a list of attributes of this field. See parsing.java_classfile.attributes.py for more info

    Parameters
    ----------
    index: `int`
        the index of this object in the field_info list
    flags: `int`
        raw parsed unsigned 2-byte interger flags from the classfile
    classfile: `JavaClassfile`
        the JavaClassfile object that is loading this field
    """

    index: 'int'
    """The index of this object in the field_info list"""

    access_flags: 'FieldAccessFlags'
    """The access flags for this field. See FieldAccessFlags for more info"""

    name_index: 'int'
    """The index into the constant pool for a CPUtf8Info object representing a valid unqualified field name"""

    name: 'str'
    """A valid unqualified field name"""

    descriptor_index: 'int'
    """The index into the constant pool for a CPUtf8Info object representing a valid field descriptor string"""

    descriptor: 'str'
    """A valid field descriptor string"""

    field_parameter: 'MethodOrFieldParameterType'
    """The field parameter type"""
    
    attributes: 'list[AttributeInfo]'
    """List of field attributes"""

    def __init__(self, index: 'int', br: 'ByteReader', classfile: 'JavaClass.JavaClassfile'):
        self.index = index
        self.br = br
        self.access_flags = FieldAccessFlags(br.read_int(2, signed=False))

        self.name_index = br.read_int(2, signed=False)
        self.name = _get_check_pool_object(self, self.name_index, classfile.constant_pool, CPUtf8Info, 'name_index').string
        _check_unqualified_name(self.name, valid_angles=[], message='Invalid name')

        self.descriptor_index = br.read_int(2, signed=False)
        self.descriptor = _get_check_pool_object(self, self.descriptor_index, classfile.constant_pool, CPUtf8Info, 'descriptor_index').string
        self.field_parameter = _check_field_descriptor(self.descriptor)

        self.attributes = parse_attributes(br, classfile.constant_pool)


class MethodInfo:
    """Each method is described by a method_info structure

    MethodInfo Struct {
        u2          access_flags,
        u2          name_index,
        u2          descriptor_index,
        u2          attributes_count,
        attribute   attributes[attributes_count]
    }
    
    No two methods in one class file may have the same name and descriptor.

    name_index and descriptor_index are indices into the constant pool for CPUtf8Info objects that represent a valid
    unqualified method name and method descriptor string respectively. See constant_pool._check_unqualified_name() for
    more info on unqualifed names and constant_pool._check_method_descriptor() for more info on method descriptor values

    attributes is a list of attributes of this method. See parsing.java_classfile.attributes.py for more info

    Parameters
    ----------
    index: `int`
        the index of this object in the method_info list
    flags: `int`
        raw parsed unsigned 2-byte interger flags from the classfile
    classfile: `JavaClassfile`
        the JavaClassfile object that is loading this field
    """

    index: 'int'
    """The index of this object in the method_info list"""

    access_flags: 'MethodAccessFlags'
    """The access flags for this method. See MethodAccessFlags for more info"""

    name_index: 'int'
    """The index into the constant pool for a CPUtf8Info object representing a valid unqualified method name"""

    name: 'str'
    """A valid unqualified method name"""

    descriptor_index: 'int'
    """The index into the constant pool for a CPUtf8Info object representing a valid method descriptor string"""

    descriptor: 'str'
    """A valid method descriptor string"""
    
    method_parameters: 'list[MethodOrFieldParameterType]'
    """If this is a method_descriptor, then this will be a list (possibly empty) of all input parameters to the method.
       Otherwise will be None"""
    
    method_return: 'MethodOrFieldParameterType'
    """If this is a method_descriptor, then this will be the return type. Otherwise will be None"""
    
    attributes: 'list[AttributeInfo]'
    """List of field attributes"""

    def __init__(self, index: 'int', br: 'ByteReader', classfile: 'JavaClass.JavaClassfile'):
        self.index = index
        self.br = br
        self.access_flags = FieldAccessFlags(br.read_int(2, signed=False))

        self.name_index = br.read_int(2, signed=False)
        self.name = _get_check_pool_object(self, self.name_index, classfile.constant_pool, CPUtf8Info, 'name_index').string
        _check_unqualified_name(self.name, message='Invalid name')

        self.descriptor_index = br.read_int(2, signed=False)
        self.descriptor = _get_check_pool_object(self, self.descriptor_index, classfile.constant_pool, CPUtf8Info, 'descriptor_index').string
        self.method_parameters, self.method_return = _check_method_descriptor(self.descriptor)

        self.attributes = parse_attributes(br, classfile)


def parse_fields(br: 'ByteReader', classfile: 'JavaClass.JavaClassfile'):
    """Parse a list of fields"""
    num_fields = br.read_int(2, signed=False)
    return _check_unique([FieldInfo(i, br, classfile) for i in range(num_fields)])


def parse_methods(br: 'ByteReader', classfile: 'JavaClass.JavaClassfile'):
    """Parse a list of methods"""
    num_fields = br.read_int(2, signed=False)
    return _check_unique([MethodInfo(i, br, classfile) for i in range(num_fields)])


def _check_unique(vals: 'list[Union[FieldInfo, MethodInfo]]'):
    """Checks that each value has a unique (name, descriptor). Returns the list back"""
    uniques = set()
    for v in vals:
        t = (v.name, v.descriptor)
        if t in uniques:
            raise ValueError("Found %s object with duplicate name/descriptor: %s" % (repr(type(v).__name__), t))
        uniques.add(t)
    return vals
