"""
Objects that pick exactly which flags are used during a single compilation
"""

from bincfg.utils import hash_obj
import numpy as np


class ChooseNum:
    """Will attempt to choose a certain number of flags randomly, or all flags if there are less than `num`"""
    def __init__(self, num):
        """
        :param num: integer number of flags to attempt to choose
        """
        if num < 0:
            raise ValueError("num must be >= 0")

        self.num = num
    
    def __call__(self, flags, rand=None):
        return [] if self.num == 0 else flags if len(flags) <= self.num else (np.random if rand is None else rand).choice(flags, self.num, replace=False)

    def __hash__(self):
        return hash_obj("(%s) %d" % (self.__class__.__name__, self.num), return_int=True)


class ChooseGaussian:
    """Will attempt to choose a certain number of flags randomly, with the number of flags being a gaussian normal distribution
    
    Number of flags will always be >= 0.
    """
    def __init__(self, center=0, std=1, min_num=0, max_num=None):
        """
        :param center: the center of the normal distribution
        :param std: the standard deviation of the normal distribution
        :param min_num: the minimum number of flags to return
        :param max_num: the maximum number of flags to return. If None, then max will be the length of the flags list
        """
        if max_num is not None and max_num < 0:
            raise ValueError("max_num must be >= 0, or None")
        if max_num is not None and max_num < min_num:
            raise ValueError("max_num must be >= min_num, or None")
        if min_num is None:
            min_num = 0
        if not isinstance(min_num, int):
            raise TypeError("min_num must be None, or an integer")

        self.center, self.std, self.min_num, self.max_num = center, std, min_num, max_num
    
    def __call__(self, flags, rand=None):
        max_val = len(flags) if self.max_num is None else self.max_num
        val = int(abs((np.random if rand is None else rand).normal(self.center, self.std)))
        val = self.min_num if val < self.min_num else max_val if val > max_val else val
        return [] if val == 0 else flags if len(flags) <= val else (np.random if rand is None else rand).choice(flags, val, replace=False)

    def __hash__(self):
        return hash_obj("(%s) %s" % (self.__class__.__name__, ' '.join([hash_obj(o) for o in [self.center, self.std, self.min_num, self.max_num]])), return_int=True)
