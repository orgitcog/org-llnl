"""
A Version() object to keep track of and easily compare version numbers
"""

import copy
import re
from bincfg.utils import hash_obj


class Version:
    """Handles and compares version numbers/strings

    Exists mostly so I can more easily/confidently sort and compare version strings.

    Assumes version_str starts with an integer, and takes the form [VERSION][EXTRA], where [VERSION] is a version
        string composed of a '.'-delimited list of integers, and [EXTRA] is any extra non-whitespace characters.
    Determining operators like <, >, =, etc. will be done on the list of integer version numbers, then on a simple
        string comparison of [EXTRA] if it exists.

    NOTE: the version_str may optionally start with a 'v' which will be ignored
    """
    def __init__(self, version_str_or_Version):
        if isinstance(version_str_or_Version, Version):
            other = version_str_or_Version
            self.version_str = other.version_str
            self.version_nums = copy.deepcopy(other.version_nums)
            self.extra_characters = other.extra_characters

        elif isinstance(version_str_or_Version, (int, str)):
            version_str = str(version_str_or_Version)
            self.version_str = version_str.strip()

            if self.version_str.startswith('v'):
                self.version_str = self.version_str[1:]

            # Parse out the version string
            version = re.match(r'[0-9]+([.][0-9]+)*(?![.])', self.version_str)
            if version is None:
                raise ValueError("Could not parse version number from version string: '%s'" % version_str)
            version = version.group()
            
            # Get any extra characters, will be empty string '' if there are None
            self.extra_characters = self.version_str.partition(version)[-1]
            
            # Parse out the version numbers. There should always be at least one
            self.version_nums = []
            for n in version.split('.'):
                try:
                    self.version_nums.append(int(n))
                except ValueError:
                    raise ValueError("Could not parse version number '%s' into an integer in version string: '%s'"
                        % (n, self.version_str))
                        
        else:
            raise TypeError("Can only create Version() object from objects of type 'str' and 'Version', not '%s'" % 
                type(version_str_or_Version))
        
    
    def matches(self, other):
        """Returns true if this version 'matches' the other version.
        
        IE: this version is from the same version family as the other one. '5.5.0' would match '5' and '5.5' and 
        '5.5.0-latest', but not '5.4' nor '5.5.1')
        """
        if isinstance(other, str):
            return self.matches(Version(other))
        elif isinstance(other, Version):
            for s, o in zip(self.version_nums, other.version_nums):
                if s != o:
                    return False
            return True
        else:
            raise TypeError("matches() is only implemented for objects of type 'str' and 'Version', not '%s'" % type(other))
    
    def __lt__(self, other):
        if isinstance(other, str):
            return self < Version(other)
        elif isinstance(other, Version):
            # Check for immediate version number differences
            for vn_self, vn_other in zip(self.version_nums, other.version_nums):
                if vn_self < vn_other:
                    return True
                elif vn_self > vn_other:
                    return False
            
            # All are equal, check for version number length differences. Self will be inherently '<' if it has fewer
            #   version numbers than other, and '>' if it has more. If it's equal, then compare extra strings
            if len(self.version_nums) < len(other.version_nums):
                return True
            elif len(self.version_nums) > len(other.version_nums):
                return False
            
            # Finally, check for extra strings. This would also handle the case where they are both equal (including
            #   both empty)
            return self.extra_characters < other.extra_characters
        elif isinstance(other, int):
            return self.version_nums[0] < other
        else:
            raise TypeError("Less than operator is only defined for objects of type 'str', 'int', and 'Version', not '%s'" % type(other))
    
    def __le__(self, other):
        return self < other or self == other
    
    def __gt__(self, other):
        return not self < other and not self == other
    
    def __ge__(self, other):
        return self > other or self == other

    def __eq__(self, other):
        if isinstance(other, str):
            return self.version_str == other.strip()
        elif isinstance(other, Version):
            return self.version_str == other.version_str
        elif isinstance(other, int):
            return self.version_nums[0] == other
        else:
            raise TypeError("Equality operator is only defined for objects of type 'str', 'int', and 'Version', not '%s'" % type(other))
    
    def __hash__(self):
        return hash_obj("(" + self.__class__.__name__ + ") " + hash_obj([self.version_nums, self.version_str]), return_int=True)
    
    def __str__(self):
        return self.version_str
    
    def __repr__(self):
        return 'Version(%s)' % repr(self.version_str)
