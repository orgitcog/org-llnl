"""
Objects to allow you to select which values will be used during a compilation process.

NOTE: these are only for selecting which values to use (IE: language, family, compiler, version, arch, and flags), NOT
the actual flag values themselves. Those would be CFAction()'s

One can create their own Selector() object by inheriting from Selector and overriding the __call__ and get_possible_choices
methods. It is recommended to also override the __hash__ method to work correctly in CAP.
"""

import numpy as np
import re
import sys
from bincfg.utils import hash_obj
from utils.version import Version


class Selector:
    def __call__(self, choices, rand=None):
        """Selects one choice from the list of available choices
        
        :param choices: a list of string choices
        :param rand: if not None, then an RNG object to use. Should match the numpy RNG methods
        """
        raise NotImplementedError
    
    def get_possible_choices(self, choices):
        """Returns a set of all elements in choices that could theoretically be chosen
        
        :param choices: an iterable of string elements to choose from
        """
        raise NotImplementedError
    
    def __hash__(self):
        raise NotImplementedError
    
    def __mem_usage__(self):
        raise NotImplementedError
    
    def __repr__(self):
        return "%s()" % self.__class__.__name__
    
    def __str__(self):
        return repr(self)


class SelectConstant(Selector):
    """Always returns a single value"""

    def __init__(self, val):
        """
        :param val: a string value to always return on calls to this object
        """
        self.val = val
    
    def get_possible_choices(self, choices):
        """Returns a set of all elements in choices that could theoretically be chosen
        
        :param choices: an iterable of string elements to choose from
        """
        return [c for c in choices if (isinstance(c, Version) and c.matches(self.val)) or c == self.val]
    
    def __call__(self, choices, rand=None):
        c = self.get_possible_choices(choices)
        return c[0] if len(c) > 0 else None
    
    def __hash__(self):
        return hash_obj("(%s) %s" % (self.__class__.__name__, self.val), return_int=True)
    
    def __mem_usage__(self):
        return sys.getsizeof(self.val) + sys.getsizeof(self)
    
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self.val))


class SelectRandom(Selector):
    """Randomly chooses one value when called
    
    Instantiating empty will select a random value from all available
    """

    def __init__(self, choices=None, regex_choices=None, select_not=False, method='uniform'):
        """
        :param choices: an iterable of string choices. All values not in this list will be ignored (unless they are
            a match in `regex_choices`).
        :param regex_choices: an iterable of string regex's (NOT compiled, will be compiled after initialization). All 
            values that do not match at least one of these will be ignored.
        :param select_not: if True, then the selection lists will be reversed and this will instead select randomly
            from all values that are NOT in any of these lists.
        :param method: string method for how to randomly choose. Can be:
            - 'uniform': uniform random choice over input space
        """
        # Check method is good
        _m = method.lower()
        if _m not in ['uniform']:
            raise ValueError("Unknown random selector method: %s" % repr(method))
        
        r_str = '|'.join(([] if choices is None else [str(c) for c in choices]) + ([] if regex_choices is None else list(regex_choices)))

        self.regex_match = re.compile(r_str if choices is not None or regex_choices is not None else '.*')
        self.select_not = select_not
        self.method = _m
    
    def get_possible_choices(self, choices):
        """Returns a set of all elements in choices that could theoretically be chosen
        
        :param choices: an iterable of string elements to choose from
        """
        if self.regex_match.pattern == '.*':
            return set(choices)
        if len(choices) > 0 and isinstance(choices[0], Version):
            return set(c for c in choices if any(c.matches(s) for s in self.regex_match.pattern.split('|')) ^ self.select_not)
        return set(c for c in choices if (self.regex_match.fullmatch(c) is not None) ^ self.select_not)
    
    def __call__(self, choices, rand=None):
        """Selects one choice from the list of available choices
        
        :param choices: a list of string choices
        :param rand: if not None, then an RNG object to use. Should match the numpy RNG methods
        """

        # Get all of the choices available to this object
        choices = list(self.get_possible_choices(choices))

        # Randomly select one depending on the method
        if self.method in ['uniform']:
            return (np.random if rand is None else rand).choice(choices)
        else:
            raise NotImplementedError("Did not implement random selector method: %s" % repr(self.method))
    
    def __hash__(self):
        return hash_obj("(%s) %s" % (self.__class__.__name__, hash_obj((self.regex_match, self.select_not, self.method))), return_int=True)
    
    def __repr__(self):
        return "%s(regex_choices=%s, select_not=%s, method=%s)" % (self.__class__.__name__, 
            repr(self.regex_match.pattern.split('|')), str(self.select_not), repr(self.method))
