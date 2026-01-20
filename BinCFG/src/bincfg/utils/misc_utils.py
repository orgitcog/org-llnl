"""
Miscellaneous utility functions
"""

import re
import time
import string
import pickle
import dis
import numpy as np
from enum import Enum
from threading import Thread
from hashlib import sha256
from inspect import currentframe, signature, _empty as EmptyDefaultParam
from types import MethodType, FunctionType
from collections import OrderedDict
from copy import deepcopy


# The actual progressbar object, once it has been determined
_IMPORT_PROGRESSBAR = None


# Python opcodes to ignore when checking similarity of functions
# Needed because starting python 3.11, there is a new 'RESUME' opcode that is essentially a nop, but does some stuff
#   for CPython internal debugging/tracing
IGNORE_OPNAMES = ['RESUME', 'NOP', 'CACHE']


def get_smallest_np_dtype(val, signed=False):
    """Returns the smallest numpy integer dtype needed to store the given max value.

    Args:
        val (int): the largest magnitude (furthest from 0) integer value that we need to be able to store
        signed (bool, optional): if True, then use signed ints. Defaults to False.

    Raises:
        ValueError: if a bad value was passed, or if the value was too large to store in a known integer size

    Returns:
        np.dtype: the smallest integer dtype needed to store the given max value
    """
    val = abs(val) if signed else val
    if val < 0:
        raise ValueError("Val must be >0 if using unsigned values: %d" % val)
    
    for dtype in ([np.int8, np.int16, np.int32, np.int64] if signed else [np.uint8, np.uint16, np.uint32, np.uint64]):
        if val < np.iinfo(dtype).max and (not signed or -val > np.iinfo(dtype).min):
            return dtype

    raise ValueError("Could not find an appropriate size for given integer: %d" % val)


def scatter_nd_numpy(target, indices, values):
    """Sets the values at `indices` to `values` in numpy array `target`
    
    Shamelessly stolen from: https://stackoverflow.com/questions/46065873/how-to-do-scatter-and-gather-operations-in-numpy

    Args:
        target (np.ndarray): the target ndarray to modify
        indices (np.ndarray): n-d array (same ndim as target) of the indices to set values to
        values (np.ndarray): 1-d array of the values to set

    Returns:
        np.ndarray: the resultant array, modified inplace
    """
    indices = tuple(indices.reshape(-1, indices.shape[-1]).T)
    np.add.at(target, indices, values.ravel())  # type: ignore
    return target


def arg_array_split(length, sections, return_index=None, dtype=np.uint32):
    """Like np.array_split(), but returns the indices that one would split at

    This will always return `sections` sections, even if `sections` > length (in which case, any empty sections will
    come at the end). If `sections` does not perfectly divide `length`, then any extras will be front-loaded, one per
    split array as needed.

    NOTE: this code was modified from the numpy array_split() source
    
    Args:
        length (int): the length of the sequence to split
        sections (int): the number of sections to split into
        return_index (Optional[int]): if not None, then an int to determine which tuple of (start, end) indices to
            return (IE: if you were splitting an array into 10 sections, and passed return_index=3, this would return
            the tuple of (start, end) indicies for the 4th split array (since we start indexing at 0))
        dtype (np.dtype): the numpy dtype to use for the returned array
    
    Returns:
        Union[np.ndarray, Tuple[int, int]]: a numpy array of length `sections + 1` where the split array at index `i`
            would use the start/end endices `[returned_array[i]:returned_array[i+1]]`, unless return_index is not None,
            in which case a 2-tuple of the (start_idx, end_idx) will be returned
    """
    if sections <= 0:
        raise ValueError('Number of sections must be > 0. Got %d' % sections)
    if length < 0:
        raise ValueError("Length must be >= 0. Got %d" % length)
    if return_index is not None and (return_index < 0 or return_index >= sections):
        raise ValueError("return_index, if not None, must be a positive integer in the range [0, sections). Got "
            "sections=%d, return_index=%d" % (sections, return_index))

    # If sections > length, then we would return the array [0, 1, 2, 3, ..., length - 1, length, length, length, ...]
    # NOTE: this also handles the case where length == 0 since sections > 0 always
    if sections >= length:
        ret_arr = np.arange(sections + 1, dtype=dtype)
        ret_arr[length:] = length

    # Otherwise we can do the normal divmod method
    # NOTE: this also handles the case where length == sections
    else:
        num_per_section, extras = divmod(length, sections)
        section_sizes = [0] + extras * [num_per_section + 1] + (sections - extras) * [num_per_section]
        ret_arr = np.array(section_sizes, dtype=dtype).cumsum()
    
    if return_index is not None:
        return ret_arr[return_index], ret_arr[return_index + 1]
    
    return ret_arr


# Some object types
_SingletonObjects = [None, Ellipsis, NotImplemented]
_DictKeysType = type({}.keys())
_GeneratorType = type((x for x in [1]))

_MAX_STR_LEN = 1000

# Types that are all able to be checked against one another using default '==' equality check
_DUNDER_EQ_TYPES = (int, float, np.number, complex, bytes, bytearray, memoryview, str, range, type, set, frozenset, _DictKeysType)

# Keep track of the current kwargs being used in equal()
_CURR_EQUAL_KWARGS = None
_EQ_DEFAULT_STRICT_TYPES = object()
_EQ_DEFAULT_UNORDERED = object()
_EQ_DEFAULT_RAISE_ERR = object()


# Context manager to return _CURR_EQUAL_KWARGS back to expected
class _ReturnCurrEqualKwargs:
    def __init__(self, kwargs):
        self.kwargs = kwargs
    def __enter__(self):
        return self
    def __exit__(self, *args):
        global _CURR_EQUAL_KWARGS
        if self.kwargs['control_kwargs']:
            _CURR_EQUAL_KWARGS = None
        else:
            for k in self.kwargs:
                if k.startswith('prev_'):
                    _CURR_EQUAL_KWARGS[k[len('prev_'):]] = self.kwargs[k]


def eq_obj(a, b, selector=None, strict_types=_EQ_DEFAULT_STRICT_TYPES, unordered=_EQ_DEFAULT_UNORDERED, raise_err=_EQ_DEFAULT_RAISE_ERR):
    """
    Determines whether a == b, generalizing for more objects and capabilities than default __eq__() method.
    Equal() is an equivalence relation, and thus:
    
        1. equal(a, a) is always True                       (reflexivity)
        2. equal(a, b) implies equal(b, a)                  (symmetry)
        3. equal(a, b) and equal(b, c) implies equal(a, c)  (transitivity)
    
    NOTE: This method is not meant to be very fast. I will apply as many optimizations as feasibly possible that I can
    think of, but there will be various inefficient conversions of types to check equality.
    
    NOTE: kwargs passed to the initial :func:`~gstats_utils.pythonutils.equality.equal` function call will be passed to 
    all subcalls, including those done in other objects using their built-in __eq__ function. Any objects can override
    those kwargs for any later subcalls (but not those above/adjacent). 
    NOTE: The `selector` kwarg is only used once, then consumed for any later subcalls
    
    Args:
        a (Any): object to check equality
        b (Any): object to check equality
        selector (Optional[str]): if not None, then a string that determines the 'selector' to use on both objects for
            determining equality. It should start with either a letter (case-sensitive), underscore '_', dot '.' or
            bracket '['. This string will essentially be appended to each object to get some attribute to determine
            equality of instead of the objects themselves. For example, if you have two lists, but only want to check
            if their element at index '2' are equal, you could pass `selector='[2]'`. This is useful for debugging purposes
            as the error messages on unequal objects will be far more informative. Defaults to None.
            NOTE: if you pass a `selector` string that starts with an alphabetical character, it will be assumed to be
            an attribute, and this will check equality on `a.SELECTOR` and `b.SELECTOR`
        strict_types (bool): if True, then the types of both objects must exactly match. Otherwise objects which are 
            equal but of different types will be considered equal. Defaults to False.
        unordered (bool): if True, then all known sequential objects (list, tuple, numpy array, etc.) will be considered
            equal even if elements are in a different order (eg: a multiset equality). Otherwise, sequential objects are
            expected to have their subelements appear in the same order. If the passed objects are not sequential, then
            this has no effect. Defaults to False.
        raise_err (bool): if True, then an ``EqualityError`` will be raised whenever `a` and `b` are unequal, along with
            an informative stack trace as to why they were determined to be unequal. Defaults to False.
    
    Raises:
        EqualityError: if the two objects are not equal, and `raise_err=True`
        EqualityCheckingError: if there was an error raised during equality checking
    
    Returns:
        bool: True if the two objects are equal, False otherwise
    """
    # Check if we are the first call and thus should controll the kwargs
    global _CURR_EQUAL_KWARGS
    _stack_kwargs = {'control_kwargs': False}
    if _CURR_EQUAL_KWARGS is None:
        _stack_kwargs['control_kwargs'] = True
        _CURR_EQUAL_KWARGS = {'strict_types': False, 'unordered': False, 'raise_err': False}
    
    # Update the kwargs if needed, otherwise grab them from the curr kwargs
    _stack_kwargs.update({'prev_strict_types': _CURR_EQUAL_KWARGS['strict_types'], 'prev_unordered': _CURR_EQUAL_KWARGS['unordered'],
        'prev_raise_err': _CURR_EQUAL_KWARGS['raise_err']})
    if strict_types is not _EQ_DEFAULT_STRICT_TYPES:
        _CURR_EQUAL_KWARGS['strict_types'] = strict_types
    else:
        strict_types = _CURR_EQUAL_KWARGS['strict_types']
    
    if unordered is not _EQ_DEFAULT_UNORDERED:
        _CURR_EQUAL_KWARGS['unordered'] = unordered
    else:
        unordered = _CURR_EQUAL_KWARGS['unordered']

    if raise_err is not _EQ_DEFAULT_RAISE_ERR:
        _CURR_EQUAL_KWARGS['raise_err'] = raise_err
    else:
        raise_err = _CURR_EQUAL_KWARGS['raise_err']

    # Cover with a context manager to reset _CURR_EQUAL_KWARGS back to expected values
    with _ReturnCurrEqualKwargs(_stack_kwargs):
        
        # Get the right selector, raising an error if it's bad
        if selector is not None:
            if not isinstance(selector, str):
                raise TypeError("`selector` arg must be str, not %s" % repr(type(selector).__name__))
            if selector == '':
                selector = None
            elif selector[0].isalpha():
                selector = '.' + selector
            elif selector[0] not in '._[':
                raise ValueError("`selector` string must start with a '.', '_', '[', or alphabetic character: %s" % repr(selector))
        
        # Use `selector` if needed
        if selector is not None:
            try:
                _failed_obj_name = 'a'
                _check_a = eval('a' + selector)
                _failed_obj_name = 'b'
                _check_b = eval('b' + selector)
                _failed_obj_name = None

                return eq_obj(_check_a, _check_b, selector=None, strict_types=strict_types, unordered=unordered, raise_err=raise_err)
            except EqualityError:
                raise EqualityError(a, b, "Objects had different sub-objects using `selector` %s" % repr(selector))
            except Exception:
                if _failed_obj_name is None:
                    raise EqualityCheckingError("Could not determine equality between objects a and b using `selector` %s\na: %s\nb: %s" %
                        (repr(selector), _limit_str(a), _limit_str(b)))
                raise EqualityCheckingError("Could not use `selector` with value %s on object `%s`" % (repr(selector), _failed_obj_name))

        # Wrap everything in a try/catch in case there is an error, so it will be easier to spot
        try:

            # Do a quick first check for 'is' as they should always be equal, no matter what
            if a is b:
                return True
            
            # Check if there are strict types
            if strict_types and type(a) != type(b):
                return _eq_check(False, a, b, raise_err, message='Objects are of different types and `strict_types=True`.')
            
            ##################
            # Checking types #
            ##################

            # We already checked 'is', so this must be an error
            if any(a is x for x in _SingletonObjects) or isinstance(a, Enum):
                return _eq_check(False, a, b, raise_err)
            
            # Check for bool first that way int's and bool's cannot be equal
            elif isinstance(a, bool) or isinstance(b, bool):
                # Enforce that this is a bool no matter what. Bool's are NOT int's. I will die on this hill...
                if not _eq_enforce_types(bool, a, b, raise_err):
                    return False
                return _eq_check(a == b, a, b, raise_err, message=None)
            
            # Check for objects using '=='
            elif isinstance(a, _DUNDER_EQ_TYPES):
                if not _eq_enforce_types(_DUNDER_EQ_TYPES, a, b, raise_err):
                    return False
                return _eq_check(a == b, a, b, raise_err, message=None)
            
            # Check for sequences list/tuple
            elif isinstance(a, (list, tuple)):
                
                # Check that b is something that could be converted into a list/tuple nicely

                # If check_b is a numpy array, convert check_a to one and do a numpy comparison
                if isinstance(b, np.ndarray):
                    # Check if check_b is an object array, and if so, use lists, otherwise use numpy
                    if b.dtype == object:
                        return _check_with_conversion(a, None, b, list, unordered, raise_err, strict_types)
                    return _check_with_conversion(a, np.ndarray, b, None, unordered, raise_err)

                # Check for things to convert to list
                elif isinstance(b, (_GeneratorType, _DictKeysType)):
                    return _check_with_conversion(a, None, b, list, unordered, raise_err)
                
                # Otherwise, make sure check_b is a list/tuple
                elif not isinstance(b, (list, tuple)):
                    return _eq_check(False, a, b, raise_err, message="checked b type could not be converted into list/tuple")
                
                # This is where we handle the actual checking.
                # Check that they are the same length
                if len(a) != len(b):
                    return _eq_check(False, a, b, raise_err, message="Objects had different lengths: %d != %d" % (len(a), len(b)))
                
                # If we are using ordered, then we can just naively check, otherwise, we have to do some other things...
                if not unordered:
                    # Check each element in the lists
                    for i, (_checking_a, _checking_b) in enumerate(zip(a, b)):
                        try:
                            # It will have returned an error if raise_err, so just return False
                            if not eq_obj(_checking_a, _checking_b, selector=None, strict_types=strict_types, unordered=unordered, raise_err=raise_err):
                                return False
                        except EqualityError:  # If we get an equality error, then raise_err must be true
                            raise EqualityError(a, b, "Values at index %d were not equal" % i)
                        except Exception:
                            raise EqualityCheckingError("Could not determine equality between elements at index %d" % i)
                    
                    # Now we can return True
                    return True

                # Unordered list checking
                else:
                    raise NotImplementedError
            
            # Check for numpy array
            elif isinstance(a, np.ndarray):

                # Ensure the other value can be converted into an array
                if not isinstance(b, np.ndarray):
                    # If check_a is an object array, then just convert it to a list now and have that check it
                    if a.dtype == object:
                        return _check_with_conversion(a, list, b, None, unordered, raise_err, strict_types)
                    
                    # Otherwise, if it is a known convertible, convert it
                    if isinstance(b, (list, tuple, _GeneratorType)):
                        return _check_with_conversion(a, None, b, np.array, unordered, raise_err, strict_types)
                    
                    # Otherwise, assume not equal
                    return _eq_check(False, a, b, raise_err, message="Could not convert b object of type %s to numpy array" % type(b).__name__)

                # Check if we are using objects or a different dtype
                if a.dtype == object:
                    # Attempt to check using lists at this point
                    return _check_with_conversion(a, list, b, list, unordered, raise_err, strict_types)

                # Otherwise, check if we are doing unordered or ordered.
                if not unordered:
                    # we can use the builtin numpy assert equal thing
                    try:
                        np.testing.assert_equal(a, b)
                        return True
                    except AssertionError as e:
                        return _eq_check(False, a, b, raise_err, message='Numpy assert_equal found discrepancies:\n%s' % e)
                
                # Otherwise we need to do an unordered equality check. Just convert to a list at this point and check it
                else:
                    return _check_with_conversion(a, list, b, list, unordered, raise_err, strict_types)
            
            # Check for dictionaries
            elif isinstance(a, dict):
                # b must be a dictionary
                if not _eq_enforce_types(dict, a, b, raise_err, message='Dictionaries must be same type to compare'):
                    return False
                
                # Check all the keys are the same
                try:
                    if not eq_obj(a.keys(), b.keys(), selector=None, strict_types=strict_types, unordered=unordered, raise_err=raise_err):
                        return False
                except EqualityError:  # If we get an equality error, then raise_err must be true
                    a_un = set(k for k in a if k not in b)
                    b_un = set(k for k in b if k not in a)
                    raise EqualityError(a, b, message="Dictionaries had different .keys()\n`a`-unique keys: %s\n`b`-unique keys: %s" 
                                        % (_limit_str(a_un), _limit_str(b_un)))
                except Exception:
                    raise EqualityCheckingError("Could not determine equality between dictionary keys\na: %s\nb: %s" %
                        (_limit_str(a.keys()), _limit_str(b.keys())))
                
                # Check all the values are the same
                for k in a:
                    try:
                        if not eq_obj(a[k], b[k], selector=None, strict_types=strict_types, unordered=unordered, raise_err=raise_err):
                            return False
                    except EqualityError:  # If we get an equality error, then raise_err must be true
                        raise EqualityError(a, b, message="Values at key %s differ" % repr(k))
                    except Exception:
                        raise EqualityCheckingError("Could not determine equality between dictionary values at key %s" % repr(k))
                
                # Now we can return True
                return True
            
            elif isinstance(a, (FunctionType, MethodType)):
                if not _eq_enforce_types((FunctionType, MethodType), a, b, raise_err, message='Functions must be same type to compare'):
                    return False
                
                # NOTE: don't do a '==' check here cause in earlier python versions (EG: 3.7), it causes an infinite
                #   recursion. For some reason, the '==' check also checks if the outer object a method is a part of
                #   is also equal, but only on older python versions. On 3.9, it works just fine

                da = _get_function_bytecode(a)
                db = _get_function_bytecode(b)
                
                try:
                    return eq_obj(da, db, selector=None, strict_types=strict_types, unordered=unordered, raise_err=raise_err)
                except EqualityError:  # If we get an equality error, then raise_err must be true
                    raise EqualityError(a, b, message="Functions contain different bytecode:\nA: %s\nB: %s" % (da, db))
                except Exception as e:
                    raise EqualityCheckingError("Could not determine equality between functions\nFor Reason: %s" % str(e))
            
            # Otherwise, use the default equality measure
            else:
                try:
                    return _eq_check(a == b, a, b, raise_err, message='Using built-in __eq__ equality measure')
                except EqualityError:  # If we get an equality error, then raise_err must be true
                    raise EqualityError(a, b, message="Values were not equal using built-in __eq__ method")
                except Exception as e:
                    raise EqualityCheckingError("Could not determine equality between dictionary values using built-in __eq__ method\nFor Reason: %s" % str(e))
        
        except EqualityError:
            raise
        except Exception as e:
            raise EqualityCheckingError("Could not determine equality between objects\na: %s\nb: %s\nFor reason: %s" % (_limit_str(a), _limit_str(b), str(e)))


def _check_with_conversion(a, type_a, b, type_b, unordered, raise_err, strict_types=False):
    """Attempts to convert check_a into type_a and check_b into type_b (by calling the types), then check equality on those
    
    Gives better error messages when things go wrong. You can pass None to one of the types to not change type. Pass the
    type itself (instead of a function) for better nameing on error messages about what they were being converted into.
    The name is given by type_a.__name__ if type_a is a type, or 'a lambda function' if it is an annonymus function, or
    the module + function name if a function
    """
    ca_type, check_a_str = _get_check_type(type_a)
    cb_type, check_b_str = _get_check_type(type_b)

    conversion_str = ('(with a value being converted using %s and b value being converted using %s)' % (check_a_str, check_b_str))\
            if check_a_str and check_b_str else \
        ('(with a value being converted using %s)' % check_a_str) if check_a_str else \
        ('(with b value being converted using %s)' % check_b_str) if check_b_str else \
        ''

    try:
        return eq_obj(ca_type(a), cb_type(b), selector=None, strict_types=strict_types, unordered=unordered, raise_err=raise_err)
    except EqualityCheckingError:
        raise
    except Exception:
        _eq_check(False, a, b, raise_err, message="Values were not equal %s" % conversion_str)


def _get_check_type(t):
    """Returns a function to call and a string describing what is being used to convert type given the type to convert
    
    Returns a tuple of (conversion_callable, type_description_string). The string will be empty if the conversion is
    the identity, t.__name__ if t is a type, 'a lambda function' if it is an anonymous function, or the module + 
    function/class name if it is a callable.
    """
    if t is None:
        return lambda x: x, ''

    if not callable(t):
        raise EqualityCheckingError("Cannot convert object types as given `type` is not callable: %s" % repr(t))
    
    return t, ('type ' + repr(t.__name__)) if isinstance(t, type) else repr(t)


def _eq_enforce_types(types, a, b, raise_err, message=None):
    """enforces check_b is of the given types using isinstance"""
    if not isinstance(a, types) or not isinstance(b, types):
        return _eq_check(False, a, b, raise_err, 'Objects were of incompatible types. %s' % message)
    return True


def _eq_check(checked, a, b, raise_err, message=None):
    """bool equal check, determine whether or not we need to raise an error with info, or just return true/false"""
    if not checked:
        if raise_err:
            raise EqualityError(a, b, message)
        return False
    return True


def _get_function_bytecode(func):
    """Returns a list of (OPNAME, ARG) tuples, normalized as much as I care to"""
    bytecode = [(inst.opname, inst.arg) for inst in dis.Bytecode(func) if inst.opname not in IGNORE_OPNAMES]

    # This is really just so our tests pass in 3.12...
    ret = []
    for opname, arg in bytecode:
        if opname == 'RETURN_CONST':
            ret += [('LOAD_CONST', arg), ('RETURN_VALUE', None)]
        else:
            ret.append((opname, arg))
    
    return ret


class _TimeoutFuncThread(Thread):
    """
    A simple Thread class to call the passed function with passed args/kwargs
    """
    def __init__(self, func, *args, **kwargs):
        """
        :param func: the function to call
        :param args: *args to pass to function when calling
        :param kwargs: **kwargs to pass to function when calling
        """
        super().__init__()
        self._func, self._args, self._kwargs = func, args, kwargs
        self._return = None
    
    def run(self):
        """
        This should never be called. Instead, call TimeoutFuncThread.start() to start thread
        """
        self._return = self._func(*self._args, **self._kwargs)


def timeout_wrapper(timeout=3, timeout_ret_val=None):
    """
    Wraps a function to allow for timing-out after the specified time. If the function has not completed after timeout
        seconds, then the function will be terminated.
    """
    def decorator(func):
        def wraped_func(*args, **kwargs):
            thread = _TimeoutFuncThread(func, *args, **kwargs)
            thread.start()

            init_time = time.time()
            sleep_time = 1e-8
            while time.time() - init_time < timeout:
                if thread.is_alive():
                    time.sleep(sleep_time)
                    sleep_time = min(0.1, sleep_time * 1.05)
                else:
                    return thread._return
            
            # If we make it here, there is an error, return value
            return timeout_ret_val
    
        return wraped_func
    return decorator


# Fail if string conversion takes > 10 seconds
_STR_CONV_TIMEOUT_SECONDS = 10

@timeout_wrapper(timeout=_STR_CONV_TIMEOUT_SECONDS, timeout_ret_val="[ERROR: String conversion timed out. Max time: %d seconds]" % _STR_CONV_TIMEOUT_SECONDS)
def _limit_str(a, limit=_MAX_STR_LEN):
    a_str = repr(a)
    return a_str if len(a_str) < limit else (a_str[:limit] + '...')


class EqualityError(Exception):
    """Error raised whenever an :func:`~gstats_utils.pythonutils.equality.equal` check returns false and `raise_err=True`"""

    def __init__(self, a, b, message=None):
        message = "Values are not equal" if message is None else message
        super().__init__("Object a (%s) is not equal to object b (%s)\na: %s\nb: %s\nMessage: %s" % \
            (repr(type(a).__name__), repr(type(b).__name__), _limit_str(a), _limit_str(b), message))


class EqualityCheckingError(Exception):
    """Error raised whenever there is an unexpected problem attempting to check equality between two objects"""


def eq_obj_err(obj1, obj2):
    """Same as eq_obj, but always raises an error"""
    return eq_obj(obj1, obj2, raise_err=True)


def hash_obj(obj, return_int=False):
    """Hashes the given object

    Args:
        obj (Any): the object to hash
        return_int (bool, optional): by default this method returns a hex string, but setting return_int=True will 
            return an integer instead. Defaults to False.

    Returns:
        Union[str, int]: hash of the given object
    """
    string = ""
    if obj is None:
        string += '[None]'
    elif isinstance(obj, (str, bool)):
        string += '(' + type(obj).__name__ + ') ' + str(obj)
    elif isinstance(obj, (int, np.integer)):
        string += '(int) ' + str(obj)
    elif isinstance(obj, (float, np.floating)):
        string += '(float) ' + str(obj)
    elif isinstance(obj, (list, tuple)):
        string += '(' + type(obj).__name__ + ') '
        for o in obj:
            string += hash_obj(o)
    elif isinstance(obj, (set, frozenset)):
        string += '(' + type(obj).__name__ + ') ' + str(sum(hash_obj(o, return_int=True) for o in obj))
    elif isinstance(obj, dict):
        string += '(' + type(obj).__name__ + ') '
        string += str(sum(hash_obj(hash_obj(k) + ', ' + hash_obj(v), return_int=True) for k, v in obj.items()))
    elif isinstance(obj, np.ndarray):
        string += '(' + type(obj).__name__ + ') '
        if obj.dtype == object:
            for a in obj:
                string += hash_obj(a) + ' '
        else:
            string += str(obj.data.tobytes())
    elif isinstance(obj, re.Pattern):
        string += '(' + type(obj).__name__ + ') ' + hash_obj(obj.pattern)
    elif isinstance(obj, _DictKeysType):
        string += '(' + type(obj).__name__ + ') ' + hash_obj(list(obj))
    elif isinstance(obj, (MethodType, FunctionType)):
        string += '(' + type(obj).__name__ + ') ' + repr(_get_function_bytecode(obj))
    else:
        string += str(hash(obj))
    
    hasher = sha256()
    hasher.update(string.encode('utf-8'))
    return int(hasher.hexdigest(), 16) if return_int else hasher.hexdigest()
    


def get_module(package, raise_err=True, err_message=''):
    """Checks that the given package is installed, returning it, and raising an error if not

    Args:
        package (str): string name of the package
        raise_err (bool, optional): by default, this will raise an error if attempting to load the module and it doesn't 
            exist. If False, then None will be returned instead if it doesn't exist. Defaults to True.
        err_message (str): an error message to add on to any import errors raised

    Raises:
        ImportError: if the package cannot be found, and `raise_err=True`

    Returns:
        Union[ModuleType, None]: the package
    """
    try:
        import importlib
        return importlib.import_module(package)
    except ImportError:
        if raise_err:
            raise ImportError("Could not find `%s` package.%s" % (package, err_message))
        return None


def isinstance_with_iterables(obj, types, recursive=False, ret_list=False):
    """Checks that obj is one of the given types, allowing for iterables of these types

    Args:
        obj (Any): the obj to test type
        types (Union[type, Tuple[type, ...]]): either a type, or tuple of types that obj can be
        recursive (bool, optional): by default, this method will only allow iterables to contain objects of a type in 
            `types`. If `recursive=True`, then this will accept arbitrary-depth iterables of types in `types`. 
            Defaults to False.
        ret_list (bool, optional): if True, will return a single list of all elements (or None if the isinstance check 
            fails). Defaults to False.

    Returns:
        Union[List[Any], bool, None]: the return value
    """
    if isinstance(obj, types):
        return [obj] if ret_list else True
    
    try:
        if ret_list:
            ret = []
            for elem in obj:
                ret += [elem] if isinstance(elem, types) else isinstance_with_iterables(elem, types, recursive=True, ret_list=True) if recursive else None
            return ret
        else:
            for elem in obj:
                if not (isinstance(elem, types) or (recursive and isinstance_with_iterables(elem, types, recursive=True, ret_list=False))):
                    return False
    except:
        return None if ret_list else False
    

def paramspec_name(obj, file_ext=None, savedparam_funcname=None, valid_filename=None):
    """Returns a string name for the given object based on save paramspec info
    
    Requires that the @parameter_saver function decorator was used on at least one function on the given object and
    was called at least once.

    Args:
        obj (Any): the object to get the string name from
        file_ext (Optional[str]): optional file extension to add to the end of the returned string. A period '.' will be
            inserted between the paramspec name and the file_ext if it is not already present at the beginning of file_ext
        savedparam_funcname (Optional[str]): the name of the function to use to generate the paramspec name. If None, then
            it will default first to '__init__' if it exists, then to the first saved paramspec attached to the object
            (in order of when the functions were called). Otherwise, should be a string name of the function to use
        valid_filename (Optional[bool]): if True, then the returned string will be modified so that it works as a valid
            filename. If False, then no such transformation will be applied. Otherwise if None, then this will be True
            if file_ext is not None and False otherwise.
    """
    valid_filename = file_ext is not None if valid_filename is None else valid_filename

    # Get the paramspec we will be using
    if not hasattr(obj, '__savedparams__') or len(obj.__savedparams__) == 0:
        raise ValueError("Could not find the '__savedparams__' attribute on the given object, or it was empty. Was a function decorated with @parameter_saver called yet?")
    elif savedparam_funcname is None:
        savedparam = obj.__savedparams__['__init__'] if '__init__' in obj.__savedparams__ else list(obj.__savedparams__.values())[0]
    elif savedparam_funcname not in obj.__savedparams__:
        raise ValueError("Could not find passed savedparam_funcname (%s) in object's __savedparams__" % repr(savedparam_funcname))
    else:
        savedparam = obj.__savedparams__[savedparam_funcname]

    # Get all of the parameters in order, in case we need to go by naming
    all_params = OrderedDict()
    for param_dict in [savedparam['args'], savedparam['kwargs']]:
        for key, val in param_dict.items():
            all_params[key] = val
    
    # Convert all of the args/kwargs to an appropriate string name
    strings = [type(obj).__name__]
    for key in savedparam['naming']:
        strings.append(_clean_paramspec_str(all_params[key], valid_filename=valid_filename))
    
    # Combine the strings together
    ret = '_'.join(strings)

    # Add on the file extension if needed
    file_ext = None if file_ext is None else file_ext if file_ext.startswith('.') else ('.' + file_ext)
    return ret if file_ext is None else (ret + file_ext)


_CLEAN_PARAMSPEC_VALID_CHARS = set(string.ascii_letters + string.digits + "-_(){}.")
def _clean_paramspec_str(val, valid_filename=False):
    """Converts the given value to a string and cleans the string
    
    This will:
        - convert 'val' to a string
        - if valid_filename is True:
          * ensures all characters are either alphanumeric or in "-_(){}." - any characters that do not fit those rules
            will be replaced with '%xHH' where HH is the hex code of the character, or '%uHHHH' where HHHH is the hex
            code of the unicode character. The '%' character will be replaced with its hexcode like so: '%x25'
          * Any spaces will be removed, with no replacement
    """
    ret_str = ""
    val_str = ('{' + val.__paramspec_name__() + '}') if hasattr(val, '__paramspec_name__') else str(val)

    if valid_filename:
        for c in val_str:
            if c not in _CLEAN_PARAMSPEC_VALID_CHARS:
                ret_str += ('%%x%02x' % ord(c)) if ord(c) <= 255 else ('%%u%04x' % ord(c))
            else:
                ret_str += c
    else:
        ret_str = val_str

    return ret_str.replace(' ', '')


def parameter_saver(func=None, naming=None, not_naming=None, ignore=None, not_ignore=None, insert_functions=False, copy=True):
    """A function that can wrap object methods to save calls to those methods
    
    Should only be used on __init__, or some other function which is only called once in that object's lifecycle.

    Can be used both like:

    .. code-block:: python

        @parameter_saver
        def __init__(self, *args, **kwargs):
            ...
    
    or like:

    .. code-block:: python

        @parameter_saver()
        def __init__(self, *args, **kwargs):
            ...
    
    Subsequent calls to wrapped functions will not have their parameters saved.

    Adds two new attributes: '__savedparams__' and '__paramspec_name__':

        - '__savedparams__': a dictionary that has keys being the function names that this wrapper was applied to
          (EG: '__init__'), and values being a subdictionary with keys/values:

          * 'args' (OrderedDict[str, Any]): args that were passed on function call, in order with their argument names
          * 'kwargs' (OrderedDict[str, Any]): kwargs that were passed on function call, in order. NOTE: any extra args 
            that would spill over into kwargs will be saved here 
          * 'naming' (Set[str]): set of strings for parameters that will be used when calling paramspec_name()
          * 'ignore' (Set[str]): set of strings for parameters to ignore all together

    Args:
        func (Callable): the function to wrap, or None if we should return a function that will later wrap another function
        naming (Optional[Iterable[str]]): iterable of strings for which parameters should be used for naming. Only the 
            parameters with these names will be used when generating a name with paramspec_name() or obj.__paramspec_name__, 
            and they will be used in the order that they appear here. Default (None) is to use all parameters in the order 
            that they appear in the method signature. Mutually exclusive with `not_naming`
        not_naming (Optional[Iterable[str]]): iterable of strings for which parameters should NOT be used for naming. All
            other parameters will be used. Mutually exclusive with `naming`
        ignore (Optional[Iterable[str]]): iterable of strings for which parameters should be ignored. These parameters
            do not appear when calling paramspec_name() and will not be saved. Default (None) is to not ignore any
            parameters. Mutually exclusive with `not_ignore`
            NOTE: only keyword arguments can be ignored
        not_ignore (Optional[Iterable[str]]): iterable of strings for which parameters should NOT be ignored. All other
            parameters will be used. Mutually exclusive with `ignore`
            NOTE: only keyword arguments can be ignored
        insert_functions (bool): if True, then extra functions will be added to the object. This will add:
            
            * .save(path: str) function - pickles the object and saves it to the given path
            * .load(path: str) function - Adds this function at the class level. Attempts to load and return a pickled
              object from the given path, checking to make sure it is the correct type
            * __setstate__(state) function - re-initializes this object with the given state information. This will attempt
              to initialize the new object with __init__ and using the args/kwargs present in __savedparams__['__init__']
              if present, then will fill in the rest of the __dict__ attributes as normal

        copy (Union[bool, str]): if True, will attempt to copy parameters by checking if they have a `.copy()` method
            and calling it if so to produce the object that is saved, that way any updates to objects during/after
            initialization will not affect the saved parameters. If False, then the original object will be used. Can
            also be the string 'deep' to perform a deep copy of each object.
    """
    # Do some error checking and cleaning
    if naming is not None and not_naming is not None:
        raise ValueError("Cannot pass both 'naming' and 'not_naming' to parameter saver")
    if ignore is not None and not_ignore is not None:
        raise ValueError("Cannot pass both 'ignore' and 'not_ignore' to parameter saver")
    if copy not in [True, False, 'deep']:
        raise ValueError("`copy` keyword must be True, False, or 'deep', not: %s" % repr(copy))
    
    def _clean(_iter):
        if _iter is None:
            return _iter
        name = [k for k, v in currentframe().f_back.f_locals.items() if _iter is v][0]  # Now we're thinking with portals
        _iter = [_iter] if isinstance(_iter, str) else list(_iter)
        for elem in _iter:
            if not isinstance(elem, str):
                raise TypeError("Elements in %s must all be strings, not: %s" % (repr(name), repr(type(elem).__name__)))
        return _iter
    
    naming, not_naming, ignore, not_ignore = _clean(naming), _clean(not_naming), _clean(ignore), _clean(not_ignore)

    def _is_positional(param):
        return param.default is EmptyDefaultParam
    def _is_kwarg(param):
        return not _is_positional(param)

    def new_obj_func(self, *args, **kwargs):
        """Fun fact: we don't have to do too much error checking here since we immediately call func() after, and any
           errors with args, positional/keyword-only arguments, passing multiple values for same parameter name, etc.
           will be caught by python when we call it
        """
        # Need access to the outer scope 'func' since it may be later modified by 'wrapped_func()'
        nonlocal func

        # Get the signature, and make sure all values in naming/not_naming/ignore/not_ignore are correct parameters
        sig = signature(func).parameters
        for _iter in [naming, not_naming, ignore, not_ignore]:
            if _iter is None:
                continue
            for elem in _iter:
                if elem not in sig:
                    raise ValueError("Unknown parameter: %s" % repr(elem))

        # Get the function signature (only keeping those names which are not to be ignored)
        # Pad args so we can zip them with paramspec later, just in case we have any args that spill over into kwargs
        # Also remove the 'self' argument, which is always the first one
        paramspec = OrderedDict()
        ret_ignore = set()
        rm_args_inds = []
        for i, (k, v) in enumerate(list(sig.items())[1:]):
            # We ignore anything in 'ignore' (while checking that it isn't an arg), or anything that is NOT in not_ignore
            #   (but only if it is a kwarg, we don't raise an error here if it is an arg - just add the arg silently)
            if (ignore is not None and k in ignore) or \
                (not_ignore is not None and k not in not_ignore and _is_kwarg(v)):
                if _is_positional(v):
                    raise ValueError("Cannot ignore positional argument: %s" % repr(k))
                
                # Also check if this is in naming (but don't worry about not_naming, that will just use everything leftover)
                if naming is not None and k in naming:
                    raise ValueError("Cannot ignore parameter %s as it is also in 'naming'" % repr(k))

                ret_ignore.add(k)
                rm_args_inds.append(i)
            
            # Otherwise we keep the parameter
            else:
                paramspec[k] = v
        
        # Determine which parameters will be used in naming
        ret_naming = list(paramspec.keys()) if (naming is None and not_naming is None) else list(naming) if naming is not None else \
            list(k for k in paramspec.keys() if k not in not_naming)
        
        # Pad the args so it zips correctly later, allowing us to make use of extra args that spill into kwargs
        # Also remove any args that would spill over into ignored kwargs
        _ARG_Padder = object()
        cleaned_args = tuple(a for i, a in enumerate(args) if i not in rm_args_inds)
        padded_args = cleaned_args + (_ARG_Padder,) * max(0, len(paramspec) - len(cleaned_args))

        def _copy(obj):
            # Copy the arg if using
            if copy == 'deep':
                return deepcopy(obj)
            elif copy and hasattr(obj, 'copy') and callable(obj.copy):
                return obj.copy()
            return obj

        # Use an OrderedDict so parameters are always in order. Order to choose kwargs is: arg from 'args', kwarg from
        #   'kwargs', default value
        ret_kwargs = OrderedDict()
        ret_args = OrderedDict()
        for arg, (key, param_obj) in zip(padded_args, paramspec.items()):
                
            # If this is an arg in the function definition
            if _is_positional(param_obj):
                ret_args[key] = _copy(arg)

            # If this is a kwarg in the function definition
            else:
                ret_kwargs[key] = _copy(arg if arg is not _ARG_Padder else kwargs[key] if key in kwargs else param_obj.default)

        # Check to make sure the __savedparams__ attribute is set on the object
        if not hasattr(self, '__savedparams__'):
            self.__savedparams__ = {}

        if func.__name__ not in self.__savedparams__:
            self.__savedparams__[func.__name__] = {'args': ret_args, 'kwargs': ret_kwargs, 'naming': ret_naming, 'ignore': ret_ignore}
        self.__paramspec_name__ = MethodType(paramspec_name, self)

        if insert_functions:
            self.save = MethodType(_paramspec_save, self)
            if not hasattr(type(self), 'load'):
                type(self).load = classmethod(_paramspec_load)
            self.__setstate__ = MethodType(_paramspec_setstate, self)
            self.__getstate__ = MethodType(_paramspec_getstate, self)
        
        return func(self, *args, **kwargs)
    
    def wrapped_func(_func):
        nonlocal func
        func = _func
        return new_obj_func
    
    return wrapped_func(func) if func is not None else wrapped_func


# Functions that will be added to paramspec objects if using
def _paramspec_save(self, path):
    with open(path, 'wb') as f:
        pickle.dump(self, f)
def _paramspec_load(cls, path):
    with open(path, 'rb') as f:
        return pickle.load(f)
_PARAMSPEC_SELF_CALLS = set()
def _paramspec_getstate(self):
    """Only keep the objects which can be pickled/unpickled, and assume everything else will be handled by __init__"""
    # We need to make sure we don't do infinite recursive calls to this function. If we do, just don't pickle it
    global _PARAMSPEC_SELF_CALLS
    if id(self) in _PARAMSPEC_SELF_CALLS:
        raise AttributeError
    _PARAMSPEC_SELF_CALLS.add(id(self))

    ret = {}
    for k, v in self.__dict__.items():
        if k == '__paramspec_name__':
            continue

        try:
            x = pickle.loads(pickle.dumps(v))
        except AttributeError:
            continue
        except Exception as e:
            print("Object cannot be pickled/unpickled: %s\nDue to Error: %s\n" % (v, e))
            raise e
        ret[k] = v
    
    _PARAMSPEC_SELF_CALLS.remove(id(self))
    return ret
def _paramspec_setstate(self, state):
    """Expects state to be a dictionary
    
    Will attempt to initialize self with __savedparams__['__init__'] if present, then will fill in the rest of the
    __dict__ info as normal
    """
    if '__savedparams__' in state and '__init__' in state['__savedparams__']:
        self.__init__(*state['__savedparams__']['__init__']['args'].values(), **state['__savedparams__']['__init__']['kwargs'])
    for k, v in state.items():
        setattr(self, k, v)


def paramspec_set_class_funcs(ret_cls):
    """Sets class functions for paramspec things on the given class"""
    ret_cls.__paramspec_name__ = paramspec_name
    ret_cls.save = _paramspec_save
    if not hasattr(ret_cls, 'load'):
        ret_cls.load = classmethod(_paramspec_load)
    ret_cls.__setstate__ = _paramspec_setstate
    ret_cls.__getstate__ = _paramspec_getstate
    return ret_cls


class ParameterSaver(type):
    """A metaclass used to add in parameter saving to the initialization function
    
    This allows you to wrap __init__ of a class without having to worry about blocking IDE's from seeing its args/kwargs,
    and will apply the parameter saving to all child classes as well. Will default to insert_functions=True
    """
    def __new__(cls, name, bases, dct):
        ret_cls = super().__new__(cls, name, bases, dct)
        ret_cls.__init__ = parameter_saver(ret_cls.__init__, insert_functions=False)
        return paramspec_set_class_funcs(ret_cls)
    

def split_by_metadata_key(metadata, set_splits, split_key, rng=None, subgroupings=None, final_sublist_size=1, eps=1e-8):
    """Splits data based on arbitrary keys in its metadata. Allows for subgroupings as well

    NOTE: This requires that all of the values for split_key in all metadata dictionaries (as well as those for any
    subgroupings being used) are hashable types.

    NOTE: make sure you include an 'INDEX' key in all of the metadata values if the order they appear in the metadata
    is not the order they should be interpreted to have in file. IE: if your 'INDEX' column in file does not match up
    with the index of datapoints within the file

    Args:
        metadata (List[Dict]): metadata for the data being split. A list of metadata dictionaries from all elements that could
            be loaded by the dataloader. If this has an 'INDEX' column, then that will be used to determine the 'indices'
            that are returned by this method. Otherwise, the indices will just be the order of datapoints as they appear.
            Assumes that if the 'INDEX' column is present in the first element, it will be present in all, and vice-versa
        set_splits (Dict[Any, float]): Dict mapping dataset name to float percent of the total dataset that should be 
            allocated to that dataset name. If an OrderedDict, then data will be assigned with priority to earlier datasets
            in the case of too few 'unique' datapoints (by `split_key`), or uneven class sizes. Otherwise, order is
            arbitrary.
        split_key (Optional[Any]): the metadata key to use to split data by. If None, will split just by the number of
            datapoints in metadata
        rng (Optional[Union[int, RNG]]): integer random state, or numpy RNG object to use for rng, or None to not randomly
            select elements and instead grab them in the order that they appear in metadata. This will gather elements
            first in order of the unique keys that appear, then in order of individual metadata elements.
        subgroupings (Optional[Iterable[Any]]): If None, then this will split normally by metadata key. Otherwise, this
            can be string/int or a list of subelements which will act as a key or keys in the metadata to subgroup
            data by. Each key will be grouped in order to apply 'subgroupings' to the data. For example, if you were
            to split by the 'problem_uid' key, then subgroup by the 'submission_id' key, this would return a list of
            lists of indices as the value for each set_split. The first list would be at the 'problem_uid' level where
            all indices with the same problem_uid would appear in the same outer list. Each sublist would contain all
            indices with the same 'submission_id' key value from those grouped into the outer 'problem_uid'-level list.
            Multiple subgrouping keys may be used at the same time to create deeper nested groupings. You may subgroup by
            the same key as the splitting key, which would ensure that, when loading data, all examples with the same value 
            for its splitting key would be prioritized to load together.

            NOTE: the current loading RNG implementation will randomly select subelements from each level of list deeper
            and deeper until reaching the final layer, at which time all values within that final list will be taken
            together. This means that if you were to say, split by 'problem_uid', and subgroup by both 'problem_uid'
            and 'submission_id' in order. You would then lose out on the prioritization of loading values with the
            same 'problem_uid' all together. To help with this, you may use the `final_sublist_size` argument which
            will make the final sublists contain that many 'unique' indices. In this the above example, it would ensure
            that there are `final_sublist_size` *unique* submission_id's within each final sublist, and that sublist
            would contain all indices with 1. a 'problem_uid' that is within that outer sublist and 2. a 'submission_id'
            that is within that inner sublist. This way, one could ensure the loading multiple examples from the same 
            problem_uid each selection, and make sure that all compilations of the same submission_id are loaded at
            the same time as well.
        final_sublist_size (int): the max size of the final sublist, in terms of number of 'unique' elements. See the 
            note above in subgroupings for more info. Only used if `subgroupings` is not None
        eps (float): small epsilon value to pass to split_list_by_sizes() using `set_splits`, see that func for more info
    
    Returns:
        Dict[Any, List[SplitIndElement]]: dictionary mapping each key in set_splits to its list of SplitIndElement
            objects. Each SplitIndElement can either be an integer index, or a list of SplitIndElement. This allows for
            nested groupings of elements to choose when loading data. 
    """
    rng = np.random.default_rng(seed=rng) if rng is not None and not isinstance(rng, np.random.Generator) else rng
    subgroupings = [subgroupings] if isinstance(subgroupings, (str, int, float, complex, bool)) else \
        [] if subgroupings is None else subgroupings
    
    # If we aren't splitting by a key at all, just split by length. RNG shuffle if needed
    if split_key is None:
        inds = list(range(len(metadata))) if 'INDEX' not in metadata[0] else [m['INDEX'] for m in metadata]
        if rng is not None: rng.shuffle(inds)
        return {k: vl for k, vl in zip(set_splits.keys(), split_list_by_sizes(inds, list(set_splits.values()), eps=eps))}

    # Group all of the indices based on their splitting metadata key
    # Wrap in try-catch in case user passes an unhashable type, not a list for `metadata`, etc
    try:
        split_mapping = OrderedDict()
        for i, md in enumerate(metadata):
            split_mapping.setdefault(md[split_key], []).append(i)
    except Exception as e:
        exc_type = type(e) if type(e) in [ValueError, TypeError, KeyError] else ValueError
        raise exc_type("Could not create inverse mapping for `metadata` dictionaries using `split_key` %s for reason:\n%s: %s"
                       % (repr(split_key), type(e).__name__, e))

    # Split up the unique split_key values based on the set_splits. Use RNG if `rng` is not None
    unique_vals = list(split_mapping.keys())
    if rng is not None: rng.shuffle(unique_vals)
    
    # Split up the unique values into all the sets and create a mapping from each value to its associated set
    # Then, grab sort all the indicies into their associated set (we already have them sorted by value in split_mapping)
    val_to_set = {v: k for k, vl in zip(set_splits.keys(), split_list_by_sizes(unique_vals, list(set_splits.values()), eps=eps)) for v in vl}
    set_inds = {k: [] for k in set_splits.keys()}
    for split_val, inds in split_mapping.items():
        set_inds[val_to_set[split_val]] += inds

    # Go through each group of elements applying subgroupings if needed, and randomizing final lists if doing that
    ret = {}
    for set_name, inds_list in set_inds.items():
        if len(subgroupings) > 0:
            inds_list = _apply_subgroupings(metadata, inds_list, subgroupings, final_sublist_size, rng)
        if rng is not None:
            rng.shuffle(inds_list)
        ret[set_name] = inds_list
    
    # Get the actual 'INDEX' values if present
    if 'INDEX' in metadata[0]:
        ret = {k: _rec_list_select_metadata_index(inds, metadata) for k, inds in ret.items()}
    return ret


def _rec_list_select_metadata_index(inds, metadata):
    """Returns a new list with the same structure as inds, just with each integer instead being converted to the correct
       'INDEX' value within the metadata"""
    return [metadata[i]['INDEX'] if isinstance(i, int) else _rec_list_select_metadata_index(i, metadata) for i in inds]


def _apply_subgroupings(metadata, set_inds, subgroupings, final_sublist_size, rng):
    """Applys subgroupings and whatnot to split_by_metadata_key indices
    
    Args:
        metadata (List[Dict]): list of dictionaries of metadata
        set_inds (List[int]): list of integer indices in metadata to subgroup
        subgroupings (List[Any]): list of keys in metadatas to group by. Will group by one key at a time recursively
        final_sublist_size (int): the number of 'unique' elements to have in the final sublist
        rng (Optional[RNG]): either None or numpy rng
    
    Returns:
        List[SetIndsElement]: list of integer set inds or other sublists of such elements
    """
    subgroup_key, subgroupings = subgroupings[0], subgroupings[1:]  # Need this, can't pop

    # Find all of the unique values for this subgroup, and which inds have those values
    try:
        subgroup_mapping = OrderedDict()
        for i in set_inds:
            subgroup_mapping.setdefault(metadata[i][subgroup_key], []).append(i)
    except Exception as e:
        exc_type = type(e) if type(e) in [ValueError, TypeError, KeyError] else ValueError
        raise exc_type("Could not create inverse mapping for `metadata` dictionaries using `subgroupings` key %s for reason:\n%s: %s"
                       % (repr(subgroup_key), type(e).__name__, e))

    # Find the unique subgroup_mapping values and RNG them if needed. This applies enough shuffling for this whole function
    unique_vals = list(subgroup_mapping.keys())
    if rng is not None: rng.shuffle(unique_vals)
    
    # Create a mapping from each unique value to a unique integer for the list it should go in. Then, create a bunch
    #   of lists to store all those values and insert all of the indices into their associated group
    unique_vals_mapping = {u: i for i, u in enumerate(unique_vals)}
    ret = [[] for _ in range(len(unique_vals))]
    for subgroup_val, inds in subgroup_mapping.items():
        ret[unique_vals_mapping[subgroup_val]] += inds
    
    # If there are more subgroupings, apply them. Otherwise, merge into final_sublist_size chunks if needed
    if len(subgroupings) > 0:
        ret = [_apply_subgroupings(metadata, l, subgroupings, final_sublist_size, rng) for l in ret]
    elif final_sublist_size > 1:
        merge_inds = np.array_split(np.arange(len(ret)), np.ceil(len(ret) / final_sublist_size))
        ret = [[v for aind in arr_inds for v in ret[aind]] for arr_inds in merge_inds]

    return ret

    
def split_list_by_sizes(l, sizes, eps=1e-8):
    """Splits the given list into len(sizes) different lists in order based on sizes
    
    Elements will be inserted into returned lists in order, prioritizing first having at least one element per list, then
    biasing any remaining elements into earlier lists.

    Args:
        l (Iterable[Any]): the list of elements to split
        sizes (Union[Iterable[float], Iterable[int]]): the different sizes to apply. Can either be an iterable of floats
            in which case each element is a percent of the total data to keep and all elements should be >=0 and <=1 and
            all elements should sum to 1. Or, can be an iterable of integers in which case all elements should be >=0
            and <= len(`l`) and all elements should sum to len(`l`)
        eps (float): the epsilon value used to determine if sum(`sizes`) (when `sizes` is a float) is equal to 1
    
    Returns:
        List[List[Any]]: a list of all sublists
    """
    l, sizes = list(l), list(sizes)

    if len(sizes) == 0:
        raise ValueError("Must pass at least one size in `sizes`")

    # Check for float vs int and get amount to have in each return list. If you pass only the int 1, then count is as 1.0
    if isinstance(sizes[0], float) or (isinstance(sizes[0], int) and sizes[0] == 1 and len(sizes) == 1):
        if abs(sum(sizes) - 1) > eps:
            raise ValueError("Sum of float `sizes` values must be 1.0, got: %f" % sum(sizes))
        
        # Split data by percent, and make sure we prioritize filling any empty lists
        sizes = [int(s * len(l)) for s in sizes]
        remaining = len(l) - sum(sizes)
        for i in range(len(sizes)):
            if remaining <= 0: break
            if sizes[i] == 0:
                sizes[i] += 1
                remaining -= 1
        
        # Any extra remaining can be inserted from start to end
        while remaining > 0:
            for i in range(len(sizes)):
                if remaining <= 0: break
                sizes[i] += 1
                remaining -= 1
        
    elif isinstance(sizes[0], (int, np.integer)):
        if sum(sizes) != len(l):
            raise ValueError("Sum of integer `sizes` values must be length of input list %d, got: %d" % (len(l), sum(sizes)))
    else:
        raise TypeError("Unknown `sizes` element type: %s" % repr(type(sizes[0]).__name__))
    
    # Now sizes should contain the number of elements to get for each list. Do a cumsum and gather all indices
    sums = np.concatenate(([0], np.cumsum(sizes)))
    return [l[sums[i]:sums[i+1]] for i in range(len(sizes))]


class _tqdm_like_iter:
    def __init__(self, iterable):
        self.iterable = iter(iterable)
    
    def __next__(self):
        return next(self.iterable)
    
    def __iter__(self):
        return self.iterable

    def update(self, num):
        next_val = None
        for i in range(num):
            next_val = next(self)
        return next_val


def _using_progressbar(iterable, *args, progress=True, **kwargs):
    """Allows one to call progressbar(iterable, progress) to determine use of progressbar automatically.
    
    Checks to see if we are in a python notebook or not to determine which progressbar we should use.
    Copied from: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    if not progress:
        return _tqdm_like_iter(iterable)

    global _IMPORT_PROGRESSBAR
    if _IMPORT_PROGRESSBAR is None:
        try:
            _tqdm_import = get_module('tqdm')
            _IMPORT_PROGRESSBAR = lambda *args, **kwargs: _tqdm_import.tqdm(*args, **kwargs)
        except ImportError:
            print("Could not import tqdm!")
            _IMPORT_PROGRESSBAR = _tqdm_like_iter
    
    return _IMPORT_PROGRESSBAR(iterable, *args, **kwargs)

progressbar = _using_progressbar
