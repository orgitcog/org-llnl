"""
Classes that define various compiler flag actions. You can create new ones by subclassing CFAction()
"""

import numpy as np
from bincfg.utils import hash_obj


class CFAction:
    """Base class for compiler flag actions"""
    def __call__(self, *args, return_combinations=None, rng=None, **kwargs):
        """Base class __call__ that should be overriden in subclasses
        
        __call__() should take in *args and **kwargs (only used to pass on to other CFActions by default), as well
        as the specific kwargs 'return_combinations' (a boolean) and 'rng' (either None, or a random number generator
        object). It should always return a string value to use for this flag (when return_combinations=None).

        Passing return_combinations='full' should make this action return an integer for the number of possible combinations
        this flag action has. EG: a CFRandomNoFlag would have 2 (for the plain and the 'no-' version of the flag), a
        CFConstant would have 1, a CFRange(0, 10) would have 10, etc.

        Passing return_combinations='simplified' should make this action return an integer for the 'simplified' number
        of possible combinations. This is purely a matter of opinion, but I chose to remove things like CFRange (IE:
        have them always return 1 no matter the range) since it can drastically blow up the number of possible
        combinations even though it may change very little.

        Passing a value to `rng` will use that random number generator while choosing random values. Otherwise if None,
        then the default numpy np.random RNG will be used.

        Args:
            args (Any): args to use, or pass to future CFActions
            return_combinations (Optional[str]): If None, then nothing changes. Otherwise can be a string 'all' or 
                'simplified' in which case this method should instead return the number of possible combinations for 
                this compiler flag action (either 'all' combinations, or a 'simplified' number of combinations)
            rng (Optional[RNG]): either a random number generator object, or None to use the default np.random RNG
        
        Returns:
            Union[str, int]: a string flag value if return_combinations is None, otherwise an integer number of combinations
        """
        raise NotImplementedError("Must override __call__ method in children of CFAction")
    
    def __repr__(self):
        return "%s()" % self.__class__.__name__
    
    def __str__(self):
        return repr(self)
    
    def __hash__(self):
        return super().__hash__()


class CFRandomNoFlag(CFAction):
    """Randomly switches between a flag and its 'no-' version
    
    The 'no-' version is the same as the original value, just with the string 'no-' inserted after the first character. 
    IE: finline -> fno-inline
    """
    def __init__(self, flag_name):
        self.flag_name = flag_name
        self.flag_name_no = self.flag_name[0] + 'no-' + self.flag_name[1:]

    def __call__(self, *args, return_combinations=None, rng=None, **kwargs):
        if return_combinations is not None:
            return 2
        return self.flag_name if (np.random.random() if rng is None else rng.random()) > 0.5 else self.flag_name_no
    
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self.flag_name))
    
    def __hash__(self):
        return hash_obj(self.flag_name, return_int=True)


class CFConstant(CFAction):
    """Always return the given value (as a string)"""
    def __init__(self, val):
        if not isinstance(val, (str, int)):
            raise TypeError("Cannot create CFConstant from value of type '%s', must be str or int" % type(val).__name__)
        self.val = str(val)
    
    def __call__(self, *args, return_combinations=None, **kwargs):
        if return_combinations is not None:
            return 1
        return self.val
    
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self.val))
    
    def __hash__(self):
        return hash_obj(self.val, return_int=True)


class CFChoice(CFAction):
    """Randomly choose uniformly from a list of values. Each value can be either a constant or another CFAction"""
    def __init__(self, vals):
        self.vals = [_good_val(v, self.__class__.__name__) for v in vals]
    
    def __call__(self, *args, return_combinations=None, rng=None, **kwargs):
        if return_combinations is not None:
            return sum(v(*args, return_combinations=return_combinations, rng=rng, **kwargs) for v in self.vals)
        
        return (rng.choice if rng is not None else np.random.choice)(self.vals)(*args, rng=rng, **kwargs)
    
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self.vals))
    
    def __hash__(self):
        return hash_obj(sum(hash_obj(v, return_int=True) for v in self.vals), return_int=True)  # Hash ignoring order


class CFRange(CFAction):
    """Randomly choose an integer value in the given range from start (inclusive) to end (exclusive)"""
    def __init__(self, start, end):
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Both start and end must be ints, not '%s' and '%s'" % (type(start).__name__, type(end).__name__))
        if end <= start:
            raise ValueError("End must be > start. Start: %d, end: %d" % (start, end))
        self.start, self.end = start, end
    
    def __call__(self, *args, return_combinations=None, rng=None, **kwargs):
        if return_combinations is not None:
            return 1 if return_combinations == 'simplified' else (self.end - self.start)
        return str((rng.randint if rng is not None else np.random.randint)(self.start, self.end))
    
    def __repr__(self):
        return "%s(%d, %d)" % (self.__class__.__name__, self.start, self.end)
    
    def __hash__(self):
        return hash_obj([self.start, self.end], return_int=True)  # Hash in order


class CFConcat(CFAction):
    """Evaluates all values in list/tuple and concatenates them (with optional separator)"""
    def __init__(self, vals, sep=''):
        """
        :param vals: a list/tuple of values
        :param sep: the string separator to use
        """
        if not isinstance(sep, str):
            raise TypeError("CFConcat 'sep' must be str, not '%s'" % type(sep).__name__)
        self.vals = [_good_val(v, self.__class__.__name__) for v in vals]
        self.sep = sep
    
    def __call__(self, *args, return_combinations=None, rng=None, **kwargs):
        if return_combinations is not None:
            ret = 1
            for v in self.vals:
                ret *= v(*args, return_combinations=return_combinations, rng=rng, **kwargs)
            return ret
        return self.sep.join([v(*args, rng=rng, **kwargs) for v in self.vals])
    
    def __repr__(self):
        return "%s(%s, sep=%s)" % (self.__class__.__name__, repr(self.vals), repr(self.sep))
    
    def __hash__(self):
        return hash_obj(self.vals, return_int=True)  # Hash in order


def _good_val(v, class_name):
    """Checks that v is a good type for a CFAction object. Converts str and int to CFConstant()"""
    if not isinstance(v, (str, int, CFAction)):
        raise TypeError("Cannot create %s() object with a value of type '%s': %s" % (class_name, type(v).__name__, v))
    return CFConstant(v) if not isinstance(v, CFAction) else v
