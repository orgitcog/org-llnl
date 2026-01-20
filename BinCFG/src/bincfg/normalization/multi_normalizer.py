"""Class that can use multiple normalization methods"""

from .base_tokenizer import _USE_DEFAULT_NT
from ..utils import hash_obj


class MultiNormalizer:
    """A normalizer that can work with multiple sub-normalizers based on architecture
    
    This does not inheret from BaseNormalizer, and thus you cannot modify or call most normalizer functions from this
    normalizer itself. It essentially just acts as a wrapper around multiple different normalizers.

    Parameters
    ----------
    normalizers: `BaseNormalizer`
        One or more normalizers to use together. May only use one per architecture.
    """
    def __init__(self, *normalizers):
        self._arch_to_norm = {}

        for n in normalizers:
            if n.tokenizer.ARCHITECTURE is None:
                raise ValueError("Cannot have a None architecture for a MultiNormalizer `normalizer`: %s" % repr(str(n)))
            elif n.tokenizer.ARCHITECTURE in self._arch_to_norm:
                raise ValueError("Found multiple tokenizers for the same architecture: %s" % repr(n.tokenizer.ARCHITECTURE))
            self._arch_to_norm[n.tokenizer.ARCHITECTURE] = n
    
    def normalize(self, *strings, cfg=None, block=None, newline_tup=_USE_DEFAULT_NT, match_instruction_address=True):
        """Normalizes the given iterable of strings.

        Args:
            strings (str): arbitrary number of strings to normalize
            cfg (Union[CFG, MemCFG], optional): either a ``CFG`` or ``MemCFG`` object that these lines occur 
                in. Used for determining function calls to self, internal functions, and external functions. If not 
                passed, then these will not be used. Defaults to None.
            block (Union[CFGBasicBlock, int], optional): either a ``CFGBasicBlock`` or integer block_idx in a ``MemCFG``
                object. Used for determining function calls to self, internal functions, and external functions. If not 
                passed, then these will not be used. Defaults to None.
            newline_tup (Tuple[str, str], optional): the tuple to insert inbetween each passed string, or None to not 
                insert anything. Defaults to self.tokenizer.DEFAULT_NEWLINE_TUPLE
            match_instruction_address (bool, optional): if True, will match instruction addresses. If there is an immediate
                value at the start of a line (IE: start of a string in `strings`, or immediately after a Tokens.NEWLINE
                or Tokens.INSTRUCTION_START [ignoring any Tokens.SPACING]), then that token will be converted into a
                Tokens.INSTRUCTION_ADDRESS token. If there is a Tokens.COLON immediately after that token (again, ignoring
                any Tokens.SPACING), then that first Tokens.COLON match will be appended (along with any inbetween Tokens.SPACING)
                to that Tokens.INSTRUCTION_ADDRESS token. For example, using the x86 tokenization scheme:

                    - "0x1234: add rax rax" -> [(Tokens.INSTRUCTION_ADDRESS, '0x1234:'), ...]
                    - "  0x1234     : add rax rax" -> [(Tokens.SPACING, '  '), (Tokens.INSTRUCTION_ADDRESS, '0x1234     :'), ...]
                    - "0x1234 add rax rax" -> [(Tokens.INSTRUCTION_ADDRESS, '0x1234'), ...]
                
            kwargs (Any): extra kwargs to pass along to tokenization method, and to store in normalizer state

        Returns:
            List[str]: a list of normalized string instruction lines
        """
        if cfg is None:
            raise ValueError("Must pass the cfg being used when calling the MultiNormalizer")
        
        return self._arch_to_norm[cfg.architecture](*strings, cfg=cfg, block=block, newline_tup=newline_tup, 
                                                  match_instruction_address=match_instruction_address)
    
    def __call__(self, *strings, cfg=None, block=None, newline_tup=_USE_DEFAULT_NT, match_instruction_address=True):
        """Normalizes the given iterable of strings.

        Args:
            strings (str): arbitrary number of strings to normalize
            cfg (Union[CFG, MemCFG], optional): either a ``CFG`` or ``MemCFG`` object that these lines occur 
                in. Used for determining function calls to self, internal functions, and external functions. If not 
                passed, then these will not be used. Defaults to None.
            block (Union[CFGBasicBlock, int], optional): either a ``CFGBasicBlock`` or integer block_idx in a ``MemCFG``
                object. Used for determining function calls to self, internal functions, and external functions. If not 
                passed, then these will not be used. Defaults to None.
            newline_tup (Tuple[str, str], optional): the tuple to insert inbetween each passed string, or None to not 
                insert anything. Defaults to self.tokenizer.DEFAULT_NEWLINE_TUPLE
            match_instruction_address (bool, optional): if True, will match instruction addresses. If there is an immediate
                value at the start of a line (IE: start of a string in `strings`, or immediately after a Tokens.NEWLINE
                or Tokens.INSTRUCTION_START [ignoring any Tokens.SPACING]), then that token will be converted into a
                Tokens.INSTRUCTION_ADDRESS token. If there is a Tokens.COLON immediately after that token (again, ignoring
                any Tokens.SPACING), then that first Tokens.COLON match will be appended (along with any inbetween Tokens.SPACING)
                to that Tokens.INSTRUCTION_ADDRESS token. For example, using the x86 tokenization scheme:

                    - "0x1234: add rax rax" -> [(Tokens.INSTRUCTION_ADDRESS, '0x1234:'), ...]
                    - "  0x1234     : add rax rax" -> [(Tokens.SPACING, '  '), (Tokens.INSTRUCTION_ADDRESS, '0x1234     :'), ...]
                    - "0x1234 add rax rax" -> [(Tokens.INSTRUCTION_ADDRESS, '0x1234'), ...]
                
            kwargs (Any): extra kwargs to pass along to tokenization method, and to store in normalizer state

        Returns:
            List[str]: a list of normalized string instruction lines
        """
        return self.normalize(*strings, cfg=cfg, block=block, newline_tup=newline_tup, match_instruction_address=match_instruction_address)
    
    def __eq__(self, other):
        return type(self) == type(other) and all(t1 == t2 for t1, t2 in zip(self._arch_to_norm.items(), other.normalizers.items()))
    
    def __hash__(self):
        return hash_obj(self._arch_to_norm, return_int=True)
    
    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, ', '.join([repr(n) for n in self._arch_to_norm.values()]))
    
    def __str__(self):
        return '-'.join([self.__class__.__name__.lower()] + [str(n) for n in self._arch_to_norm.values()])
