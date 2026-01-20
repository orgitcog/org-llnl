"""
ndarray with `meta` attribute added.
This should keep track of additional (image data) info as
    filename
    pixel / time resolution
    file type
    ...
"""
#from __future__ import absolute_import

__author__  = "Sebastian Haase <seb.haase+Priithon@gmail.com>"
__license__ = "BSD license - see LICENSE file"

#import numpy as N
from numpy import ndarray, asanyarray

class nd_meta_attribute(object):
    def __init__(self, a=None, **kwargs):
        if a is None:
            return
        if type(a) is dict:
            for k,v in a.iteritems():
                self.__dict__[k] = v
        for k,v in kwargs.iteritems():
                self.__dict__[k] = v
    def __setitem__(self, n, v):
            self.__dict__[n] = v
    def __getitem__(self, n):
        return self.__dict__[n]
    def __repr__(self):
        return "nd_meta_attribute="+repr(self.__dict__)
    def __str__(self):
        return "\n".join((("%s: %r"%(k,v)) 
                          for (k,v) in sorted(self.__dict__.iteritems())))

class ndarray_meta(ndarray):
    def __new__(cls, input_array, meta=None):
        obj = asanyarray(input_array).view(cls)
        obj.meta = nd_meta_attribute( meta )
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.meta = getattr(obj, 'meta', nd_meta_attribute())
        self.__dict__.update(getattr(obj, "__dict__", {}))
