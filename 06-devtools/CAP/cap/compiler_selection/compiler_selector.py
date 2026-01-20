"""
Allows one to do fine-grained selections of compiler setups for compile jobs.
"""
from bincfg.utils import hash_obj
from parsing.container_info import _LEVELS, _parse_flag
from compiler_selection.selectors import SelectRandom, SelectConstant, Selector
from compiler_selection.choose_flags import ChooseGaussian
from utils.version import Version


_SELECTORS = ['language', 'family', 'compiler', 'version', 'arch', 'flags']


class CompilerSelector:
    """Object that, when called, will return a set of compiler info for a single compilation."""

    def __init__(self, compile_method, compiler_info, prune_no_flags=True):
        """
        Object that, when called, will return a set of compiler info for a single compilation.
        :param compile_method: the compile method. Should be a dictionary with 0 or more of the following keys:
            'language', 'family', 'compiler', 'version', 'arch', 'flags', 'force_flags', 'update_flags', 'choose_flags'
            
            For the keys ['language', 'family', 'compiler', 'version', 'arch', 'flags'], values can be:
                - string/int: converted to string, name of singular value to use
                - list/tuple/set: will pick one of these values uniformly randomly. Any value in these collections that 
                    does not appear anywhere in the compiler_info will raise an error.
                - Selector() objects: will follow what that selector object does
            
            NOTE: by default, any key from above that was not passed in compile_method will randomly choose from all
            available values

            For 'force_flags', it should be a dictionary mapping string flag names to values. These will override
            any `force_flags` or `flags` from the compiler info

            All of the above keys are used to decide how to randomly choose values. If you wish to update what flag
            values are available for choosing (IE: overriding the flag values in compiler_info), you should use:
            
            'update_flags': a dictionary. Has keys of flag names, and flag-values are the same as those in parsing.compiler_info
            NOTE: These will apply both to flags in `flags` and in `force_flags`

            Finally, 'choose_flags' can be used to determine exactly how many and which flags will be chosen from `flags`
            each call to this object. It should be a callable that takes in a list of available flags and returns either
            a single flag string, or a list of flag strings that should be used this call

            compile_method may also contain keys/values that are kwargs to this function, which will override any kwargs
            passed to this method

        :param compiler_info: dictionary of compiler info. See parsing.compiler_info file for expected format
        :param prune_no_flags: by default, any compiler binaries that have no flags associated with them will be pruned
            (removed) from the tree, and thus would not appear as a possible compiler selection when this object is
            called. Setting `prune_no_flags=False` will keep the binaries and assume default flags
        """
        prune_no_flags = compile_method['prune_no_flags'] if 'prune_no_flags' in compile_method else prune_no_flags
        self.selectors, self.tree, self.flag_chooser, self.prune_no_flags = None, None, None, prune_no_flags
        self._build_tree(compiler_info)
        self._set_method(compile_method)
        self.combinations = self._count_combinations(rc='full')
        self.simplified_combinations = self._count_combinations(rc='simplified')
    
    def __hash__(self):
        return hash_obj("(%s) %s" % (self.__class__.__name__, \
            ' '.join([hash_obj(o) for o in [self.selectors, self.tree, self.flag_chooser, self.prune_no_flags]])), return_int=True)

    def __eq__(self, other):
        return isinstance(other, CompilerSelector) and hash(self) == hash(other)
    
    def __lt__(self, other):
        return isinstance(other, CompilerSelector) and hash(self) < hash(other)
    
    def has_path(self):
        """Returns True if there is at least one possible compiler configuration, false otherwise."""
        return len(self.tree) > 0
    
    def is_empty(self):
        return not self.has_path()
    
    def _build_tree(self, compiler_info):
        """Builds the compiler selection tree based on the given compiler_info

        Builds the tree from the compiler_info by doing the following:
            - languages get unraveled back to the beginning to start out the tree
            - keys will be converted into frozenset's of aliases
        
        The tree will be a dictionary shaped like:
        {
            language1: {
                family1: {
                    compiler1: {
                        version1: {
                            arch1: {
                                'binary_name': binary_name,
                                'container': container,
                                'flags': {
                                    flag1: val1,
                                    ...
                                },
                                'force_flags': {
                                    fflag1: fval1,
                                    ...
                                }
                            },
                            ...
                        },
                        ...
                    },
                    ...
                },
                ...
            },
            ...
        }

        NOTE: the force_flags and flags may have values in common at this step. This will be fixed by the _set_method() call
        """
        # Find the set of all languages
        langs = set(l for f, fd in compiler_info.items() for c, cd in fd[_LEVELS[1]].items() for v, vd in cd[_LEVELS[2]].items()\
            for a, ad in vd[_LEVELS[3]].items() for l in ad['supported_languages'])
        
        # Build the subtrees for each language. No need to remove empty trees just yet, they will be pruned later
        self.tree = {
            lang: {
                f: {
                    c: {
                        v: {
                            a: {ak: av for ak, av in ad.items() if ak not in ['supported_langauges']}
                            for a, ad in vd[_LEVELS[3]].items() if lang in ad['supported_languages']
                        } for v, vd in cd[_LEVELS[2]].items()
                    } for c, cd in fd[_LEVELS[1]].items()
                } for f, fd in compiler_info.items()
            } for lang in langs
        }
    
    def _set_method(self, compiler_method):
        """Constrains the tree to the given compiler_method, pruning any empty subtrees"""
        
        # Update all of the flag items
        for k in [k for k in ['update_flags', 'force_flags'] if k in compiler_method]:
            flag_dict = {fk: _parse_flag(fk, fv) for fk, fv in compiler_method[k].items()}
            for l, ld in self.tree.items():
                for f, fd in ld.items():
                    for c, cd in fd.items():
                        for v, vd in cd.items():
                            for a, ad in vd.items():
                                ad.setdefault('flags' if k == 'update_flags' else k, {}).update(flag_dict)

        # Get the selector items. Convert everything to a selector if possible
        self.selectors = []
        for k in _SELECTORS:
            ver = Version if k == 'version' else lambda x: x

            if k not in compiler_method:
                self.selectors.append(SelectRandom())
            elif isinstance(compiler_method[k], (str, int)):
                self.selectors.append(SelectConstant(ver(str(compiler_method[k]))))
            elif isinstance(compiler_method[k], (list, tuple, set, frozenset)):
                self.selectors.append(SelectRandom(choices=[ver(cm) for cm in compiler_method[k]]))
            elif isinstance(compiler_method[k], Selector):
                self.selectors.append(compiler_method[k])
            else:
                raise TypeError("Cannot build a compiler info selector object out of object of type '%s'" % type(compiler_method[k]).__name__)

        # Get the flag chooser if it is there
        self.flag_chooser = compiler_method['choose_flags'] if 'choose_flags' in compiler_method else ChooseGaussian()
        
        # Prune the tree
        self._prune_tree(self.tree, 0)
    
    def _prune_tree(self, tree, selector_idx):
        """Recursively prunes the tree, removing all of the impossible values"""
        all_vals = self.selectors[selector_idx].get_possible_choices(list(tree.keys()))
        dels = []

        for k, d in tree.items():
            if k not in all_vals:
                dels.append(k)
                continue
            
            # If each d is an arch_dict, we need to look at the force_flags and flags and update them
            if _SELECTORS[selector_idx] == 'arch':
                d.setdefault('flags', {})

                all_flags = self.selectors[selector_idx + 1].get_possible_choices(list(d['flags'].keys()))
                d['force_flags'] = {fk: fv for fk, fv in d['force_flags'].items() if fv is not None}
                d['flags'] = {fk: fv for fk, fv in d['flags'].items() if fv is not None and fk not in d['force_flags'] and fk in all_flags}
                
                # If there are no possible flags/force_flags, remove this leaf
                if len(d['flags']) == 0 and len(d['force_flags']) == 0 and self.prune_no_flags:
                    dels.append(k)
                    continue
            
            # Otherwise, normal recurse
            else:
                self._prune_tree(d, selector_idx + 1)

            if len(d) == 0:
                dels.append(k)
                continue
        
        for k in dels:
            del tree[k]

    def _count_combinations(self, rc='full'):
        """Counts the number of combinations, and simplified combinations in this CompilerSelector"""
        return sum(_prod(fv(return_combinations=rc) for fk, fv in ad['flags'].items()) for l, ld in self.tree.items() \
            for f, fd in ld.items() for c, cd in fd.items() for v, vd in cd.items() for a, ad in vd.items())
    
    def __call__(self, language, rand=None):
        """Returns a single compiler selection
        
        Selects a single dictionary of compiler info based on this object's tree and selectors.

        :param language: the language being used
        :param rand: if None, then use default np.random module, otherwise an RNG object similar to np.random
        :return: a dictionary with keys/values:
            {
                'container': string container path
                'binary_name': string binary name
                'language': string programming language (passed to this method)
                'family': string name of compiler family
                'compiler': string name of compiler
                'version': Version() object of the version
                'arch': string architecture name
                'flags': list of strings of compiler flags
            }
        """
        if not isinstance(language, str):
            raise TypeError("`language` must be a string, not %s" % type(language).__name__)
        
        if language not in self.tree:
            raise ValueError("language %s not found in tree. Available languages: %s" % (repr(language), list(self.tree.keys())))
        
        # Language
        ls = self.selectors[0]([language] if language is not None else list(self.tree.keys()))
        if ls is None:
            raise ValueError("Language selector did not select current language: %s" % repr(language))
        ld = self.tree[ls]

        # Family
        fs = self.selectors[1](list(ld.keys()))
        fd = ld[fs]

        # Compiler
        cs = self.selectors[2](list(fd.keys()))
        cd = fd[cs]
        
        # Version
        vs = self.selectors[3](list(cd.keys()))
        vd = cd[vs]

        # Architecture
        arch_s = self.selectors[4](list(vd.keys()))
        ad = vd[arch_s]

        # Flags
        # Get the forced flag values, choose the flag keys/values we will use, then get their values
        chosen_flags = self.flag_chooser(list(ad['flags'].keys()))
        flags = ['-' + fv(rand=rand) for fk, fv in ad['force_flags'].items()] + ['-' + ad['flags'][fk](rand=rand) for fk in chosen_flags]

        return {
            'container': ad['container'],
            'binary_name': ad['binary_name'],
            'language': language,
            'family': fs,
            'compiler': cs,
            'version': vs,
            'arch': arch_s,
            'flags': flags,
        }
    
    def __str__(self):
        return "CompilerSelector with %d possible simplified combinations (%d total)" % (self.simplified_combinations, self.combinations)
    
    def __repr__(self):
        return str(self)


def _prod(iter):
    prod = 1
    for i in iter:
        prod *= i
    return prod
