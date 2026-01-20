"""A bunch of various access flags objects for different sections of the classfile"""


class BaseFlags:
    """Base class for flags objects"""
    
    def as_dict(self):
        """Returns a dictionary mapping flag names to their boolean values"""
        return {k: getattr(self, k) for k in dir(self) if k.isupper() and not k.startswith("_")}
    
    def __getitem__(self, item):
        if item not in self.as_dict():
            raise KeyError("Unknown flag: %s. Available flags: %s" % (repr(item), list(self.as_dict().keys())))
        return self.as_dict()[item]

    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return str(self.as_dict())


class ClassAccessFlags(BaseFlags):
    """The access flags in a java classfile
    
    Available Flags:

        Flag Name           Mask        Interpretation
        ACC_PUBLIC          0x0001      Declared public; may be accessed from outside its package
        ACC_FINAL           0x0010      Declared final; no subclasses allowed
        ACC_SUPER           0x0020      Treat superclass methods specially when invoked by the invokespecial instruction
        ACC_INTERFACE       0x0200      Is an interface, not a class
        ACC_ABSTRACT        0x0400      Declared abstract; must not be instantiated
        ACC_SYNTHETIC       0x1000      Declared synthetic; not present in the source code
        ACC_ANNOTATION      0x2000      Declared as an annotation interface
        ACC_ENUM            0x4000      Declared as an enum class
        ACC_MODULE          0x8000      Is a module, not a class or interface
    
    See the java documentation for more info on what specific flags do

    Parameters
    ----------
    flags: `int`
        The 16-bit integer flags
    """

    ACC_PUBLIC: 'bool'
    """(0x0001) Declared public; may be accessed from outside its package"""

    ACC_FINAL: 'bool'
    """(0x0010) Declared final; no subclasses allowed"""

    ACC_SUPER: 'bool'
    """(0x0020) Treat superclass methods specially when invoked by the invokespecial instruction"""

    ACC_INTERFACE: 'bool'
    """(0x0200) Is an interface, not a class"""

    ACC_ABSTRACT: 'bool'
    """(0x0400) Declared abstract; must not be instantiated"""

    ACC_SYNTHETIC: 'bool'
    """(0x1000) Declared synthetic; not present in the source code"""

    ACC_ANNOTATION: 'bool'
    """(0x2000) Declared as an annotation interface"""

    ACC_ENUM: 'bool'
    """(0x4000) Declared as an enum class"""

    ACC_MODULE: 'bool'
    """(0x8000) Is a module, not a class or interface"""

    flags: 'int'
    """The raw integer flags from the classfile"""

    def __init__(self, flags: int):
        self.flags = flags
        self.ACC_PUBLIC = bool(flags & 0x0001)
        self.ACC_FINAL = bool(flags & 0x0010)
        self.ACC_SUPER = bool(flags & 0x0020)
        self.ACC_INTERFACE = bool(flags & 0x0200)
        self.ACC_ABSTRACT = bool(flags & 0x0400)
        self.ACC_SYNTHETIC = bool(flags & 0x1000)
        self.ACC_ANNOTATION = bool(flags & 0x2000)
        self.ACC_ENUM = bool(flags & 0x4000)
        self.ACC_MODULE = bool(flags & 0x8000)
    

class FieldAccessFlags(BaseFlags):
    """The access flags in a field object
    
    Available Flags:

        Flag Name           Mask        Interpretation
        ACC_PUBLIC          0x0001      Declared public; may be accessed from outside its package.
        ACC_PRIVATE         0x0002      Declared private; accessible only within the defining class and other classes belonging to the same nest.
        ACC_PROTECTED       0x0004      Declared protected; may be accessed within subclasses.
        ACC_STATIC          0x0008      Declared static.
        ACC_FINAL           0x0010      Declared final; never directly assigned to after object construction.
        ACC_VOLATILE        0x0040      Declared volatile; cannot be cached.
        ACC_TRANSIENT       0x0080      Declared transient; not written or read by apersistent object manager.
        ACC_SYNTHETIC       0x1000      Declared synthetic; not present in the source code.
        ACC_ENUM            0x4000      Declared as an element of an enum class.
    
    See the java documentation for more info on what specific flags do

    Parameters
    ----------
    flags: `int`
        The 16-bit integer flags
    """

    ACC_PUBLIC: 'int'
    """(0x0001)      Declared public; may be accessed from outside its package."""

    ACC_PRIVATE: 'int'
    """(0x0002) Declared private; accessible only within the defining class and other classes belonging to the same nest."""

    ACC_PROTECTED: 'int'
    """(0x0004) Declared protected; may be accessed within subclasses."""

    ACC_STATIC: 'int'
    """(0x0008) Declared static."""

    ACC_FINAL: 'int'
    """(0x0010) Declared final; never directly assigned to after object construction."""
 
    ACC_VOLATILE: 'int'
    """(0x0040) Declared volatile; cannot be cached."""

    ACC_TRANSIENT: 'int'
    """(0x0080) Declared transient; not written or read by apersistent object manager."""

    ACC_SYNTHETIC: 'int'
    """(0x1000) Declared synthetic; not present in the source code."""

    ACC_ENUM: 'int'
    """(0x4000) Declared as an element of an enum class."""

    flags: 'int'
    """The raw integer flags from the classfile"""

    def __init__(self, flags: int):
        self.flags = flags
        self.ACC_PUBLIC = bool(flags & 0x0001)
        self.ACC_PRIVATE = bool(flags & 0x0002)
        self.ACC_PROTECTED = bool(flags & 0x0004)
        self.ACC_STATIC = bool(flags & 0x0008)
        self.ACC_FINAL = bool(flags & 0x0010)
        self.ACC_VOLATILE = bool(flags & 0x0040)
        self.ACC_TRANSIENT = bool(flags & 0x0080)
        self.ACC_SYNTHETIC = bool(flags & 0x1000)
        self.ACC_ENUM = bool(flags & 0x4000)


class MethodAccessFlags(BaseFlags):
    """The access flags for a method object
    
    Available Flags:

        Flag Name           Mask        Interpretation
        ACC_PUBLIC          0x0001      Declared public; may be accessed from outside its package.
        ACC_PRIVATE         0x0002      Declared private; accessible only within the defining class and other classes belonging to the same nest.
        ACC_PROTECTED       0x0004      Declared protected; may be accessed within subclasses.
        ACC_STATIC          0x0008      Declared static.
        ACC_FINAL           0x0010      Declared final; must not be overridden (§5.4.5).
        ACC_SYNCHRONIZED    0x0020      Declared synchronized; invocation is wrapped by a monitor use.
        ACC_BRIDGE          0x0040      A bridge method, generated by the compiler.
        ACC_VARARGS         0x0080      Declared with variable number of arguments.
        ACC_NATIVE          0x0100      Declared native; implemented in a language other than the Java programming language.
        ACC_ABSTRACT        0x0400      Declared abstract; no implementation is provided.
        ACC_STRICT          0x0800      In a class file whose major version number is at least 46 and at most 60: Declared strictfp.
        ACC_SYNTHETIC       0x1000      Declared synthetic; not present in the source code.
    
    See the java documentation for more info on what specific flags do

    Parameters
    ----------
    flags: `int`
        The 16-bit integer flags
    """

    ACC_PUBLIC: 'int'
    """(0x0001) Declared public; may be accessed from outside its package."""

    ACC_PRIVATE: 'int'
    """(0x0002) Declared private; accessible only within the defining class and other classes belonging to the same nest."""

    ACC_PROTECTED: 'int'
    """(0x0004) Declared protected; may be accessed within subclasses."""

    ACC_STATIC: 'int'
    """(0x0008) Declared static."""

    ACC_FINAL: 'int'
    """(0x0010) Declared final; must not be overridden"""

    ACC_SYNCHRONIZED: 'int'
    """(0x0020) Declared synchronized; invocation is wrapped by a monitor use."""

    ACC_BRIDGE: 'int'
    """(0x0040) A bridge method, generated by the compiler."""

    ACC_VARARGS: 'int'
    """(0x0080) Declared with variable number of arguments."""

    ACC_NATIVE: 'int'
    """(0x0100) Declared native; implemented in a language other than the Java programming language."""

    ACC_ABSTRACT: 'int'
    """(0x0400) Declared abstract; no implementation is provided."""

    ACC_STRICT: 'int'
    """(0x0800) In a class file whose major version number is at least 46 and at most 60: Declared strictfp."""

    ACC_SYNTHETIC: 'int'
    """(0x1000) Declared synthetic; not present in the source code."""

    flags: 'int'
    """The raw integer flags from the classfile"""

    def __init__(self, flags: int):
        self.flags = flags
        self.ACC_PUBLIC = bool(flags & 0x0001)
        self.ACC_PRIVATE = bool(flags & 0x0002)
        self.ACC_PROTECTED = bool(flags & 0x0004)
        self.ACC_STATIC = bool(flags & 0x0008)
        self.ACC_FINAL = bool(flags & 0x0010)
        self.ACC_SYNCHRONIZED = bool(flags & 0x0020)
        self.ACC_BRIDGE = bool(flags & 0x0040)
        self.ACC_VARARGS = bool(flags & 0x0080)
        self.ACC_NATIVE = bool(flags & 0x0100)
        self.ACC_ABSTRACT = bool(flags & 0x0400)
        self.ACC_STRICT = bool(flags & 0x0800)
        self.ACC_SYNTHETIC = bool(flags & 0x1000)
