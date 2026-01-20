import numpy as np
import pickle
import traceback
from .cfg_dataset import CFG, CFGDataset
from .mem_cfg import MemCFG
from ..utils import check_for_normalizer, isinstance_with_iterables, eq_obj, hash_obj, AtomicTokenDict
from ..normalization import get_normalizer, normalize_cfg_data


class MemCFGDataset:
    """A CFGDataset that is more memory efficient
    
    Parameters
    ----------
    cfg_data: `Optional[Union[str, CFG, CFGDataset, MemCFG, MemCFGDataset, Iterable]]`
        the data to use. Can be None for an empty dataset, or a string (for input to CFG), `CFG`, `CFGDataset`, `MemCFG`, 
        `MemCFGDataset`, or iterable of those values to add that data to this dataset
    tokens: `Optional[Union[Dict[str, int], AtomicTokenDict]]`
        if passed, will initialize the token dictionary to this dictionary of tokens (will be copied). Can be an
        AtomicTokenDict to use an atomic file token dictionary
    normalizer: `Optional[Union[str, Normalizer]]`
        the normalizer to use, or None to default to the normalizer of the first added CFG/MemCFG
    metadata: `Optional[Dict]`
        a dictionary of metadata to attach to this MemCFGDataset
        NOTE: passed dictionary will be shallow copied
    add_data_kwargs: `Any`
        kwargs to pass to self.add_data() when adding the passed cfg_data
    """

    cfgs = None
    """The list of all memcfgs in this dataset"""

    normalizer = None
    """The normalizer used in this dataset, or None if there is no normalizer"""

    metadata = None
    """A dictionary of metadata associated with this ``MemCFGDataset``"""

    using_tokens = None
    """A dictionary mapping string tokens to their integer values
    
    Can be an AtomicTokenDict for atomic updates to tokens
    """

    def __init__(self, cfg_data=None, using_tokens=None, normalizer=None, metadata=None, **add_data_kwargs):
        self.cfgs = []
        self.tokens = {} if using_tokens is None else using_tokens
        self.normalizer = get_normalizer(normalizer) if normalizer is not None else None
        self.metadata = {} if metadata is None else metadata.copy()
        self._curr_cfg_memory_usage = 0

        if cfg_data is not None:
            self.add_data(cfg_data, **add_data_kwargs)
        elif len(add_data_kwargs) != 0:
            raise ValueError("Passed add_data_kwargs, but no data to add: %s" % add_data_kwargs)
    
    def add_data(self, *cfg_data, inplace=True, force_renormalize=False, progress=False):
        """Adds data to this dataset

        Args:
            cfg_data (Union[str, CFG, MemCFG, CFGDataset, MemCFGDataset, Iterable]): arbitrary amount of 
                str (CFG input)/CFG/MemCFG/CFGDataset/MemCFGDataset's, or iterables of them, to add to this dataset
            inplace (bool, optional): whether or not to normalize the incoming cfg_data inplace. Defaults to True.
            force_renormalize (bool, optional): by default, this method will only normalize cfg's whose 
                .normalizer != to this dataset's normalizer. However if `force_renormalize=True`, then all cfg's will be
                renormalized even if they have been previously normalized with the same normalizer. Defaults to False.
            mp (bool, optional): if True, will use multiprocessing to normalize cfgs. Defaults to False.
            progress (bool, optional): if True, will show a progressbar when adding multiple cfgs. Defaults to False.

        Raises:
            TypeError: if something other than a cfg/dataset is passed in `cfg_data`
        """
        # Check that all elements in cfg_data are CFG's/CFGDataset's/MemCFG's/MemCFGDataset's, or iterables of them
        temp = []
        for cfg in cfg_data:
            try:
                temp += isinstance_with_iterables(cfg, (str, CFG, CFGDataset, MemCFG, MemCFGDataset), recursive=False, ret_list=True)
            except:
                raise TypeError("Can only add str's/CFG's/CFGDataset's/MemCFG's/MemCFGDataset's, or iterables of them, to "
                    "CFGDataset, not '%s'" % type(cfg).__name__)
        cfg_data = temp

        # Check to make sure we have a normalizer to use
        if self.normalizer is None:
            check_for_normalizer(self, cfg_data)
        
        cfg_data = normalize_cfg_data(cfg_data, normalizer=self.normalizer, inplace=inplace, using_tokens=self.tokens,
            force_renormalize=force_renormalize, convert_to_mem=True, unpack_cfgs=True, progress=progress)
        
        # Set the cfg's normalizer and tokens attributes to this dataset's objects
        for cfg in cfg_data:
            cfg.normalizer, cfg.tokens = self.normalizer, self.tokens

        self.cfgs += cfg_data
    
    def normalize(self, normalizer=None, inplace=True, force_renormalize=False, progress=False):
        """Normalize this ``MemCFGDataset``.

        Args:
            normalizer (Union[str, Normalizer]): the normalizer to use. Can be a ``Normalizer`` object, or a 
                string, or None to use the default BaseNormalizer(). Defaults to None.
            inplace (bool, optional): by default, normalizes this dataset inplace (IE: without copying objects). Can set
                to False to return a copy. Defaults to True.
            force_renormalize (bool, optional): by default, this method will only normalize cfg's whose 
                .normalizer != to the passed normalizer. However if `force_renormalize=True`, then all cfg's will be 
                renormalized even if they have been previously normalized with the same normalizer.. Defaults to False.
            progress (bool, optional): if True, will show a progressbar while normalizing. Defaults to False.

        Returns:
            MemCFGDataset: this dataset normalized
        """
        # We don't want to pass using_tokens because we are normalizing with new tokens (most likely)
        return normalize_cfg_data(self, normalizer, inplace=inplace, force_renormalize=force_renormalize, progress=progress)
        
    def remove_cfg(self, cfg_or_idx):
        """Removes the given MemCFG (or index of MemCFG if cfg_or_idx is an integer) from this MemCFGDataset

        Args:
            cfg_or_idx (Union[MemCFG, int]): cfg or index to remove
        """
        if isinstance(cfg_or_idx, (int, np.integer)):
            cfg_or_idx = self.cfgs[cfg_or_idx]
        
        self.cfgs.remove(cfg_or_idx)
    
    @property
    def num_cfgs(self):
        return len(self.cfgs)

    @property
    def num_blocks(self):
        return sum(cfg.num_blocks for cfg in self.cfgs)
    
    @property
    def num_asm_lines(self):
        return sum(cfg.num_asm_lines for cfg in self.cfgs)
    
    @property
    def num_edges(self):
        return sum(cfg.num_edges for cfg in self.cfgs)
    
    @property
    def num_functions(self):
        return sum(cfg.num_functions for cfg in self.cfgs)

    def save(self, path, freeze_tokens=True):
        """Saves this MemCFGDataset to path
        
        Args:
            path (str): the filepath to save to
            freeze_tokens (bool): whether or not to 'freeze' the tokens in this MemCFGDataset. 'freezing' the tokens
                just means that, if an AtomicTokenDict is the current token dictionary for this MemCFGDataset, then
                its current data will be saved in the pickle file as a normal dict. This is useful for loading this
                data later so that the loading does not depend on being able to access the files for the AtomicTokenDict.
                Default: True. If the token dictionary is already a dict, then this has no effect
        """
        old_tokens = self.tokens
        self.tokens = self.tokens.data if isinstance(self.tokens, AtomicTokenDict) and freeze_tokens else self.tokens
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self.tokens = old_tokens
    
    def dumps(self):
        """Returns this object pickled with pickle.dumps()"""
        return pickle.dumps(self)
    
    @classmethod
    def load(cls, path):
        """Loads this MemCFGDataset from path"""
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise ValueError("Error reading pickle file: %s. Reason:\n%s: %s\nTraceback: %s" 
                             % (repr(path), type(e).__name__, e, traceback.format_exc()))
    
    def __getstate__(self):
        """State for pickling"""
        state = self.__dict__.copy()
        state['cfgs'] = [{k: v for k, v in cfg.__getstate__().items() if k not in ['normalizer', 'tokens']} for cfg in state['cfgs']]
        return state
    
    def __setstate__(self, state):
        """State for unpickling"""        
        # Setting attributes, and cfgs list
        self.cfgs = []
        for k, v in state.items():
            if k == 'cfgs':
                for cfg_dict in v:
                    new_cfg = MemCFG(None)
                    new_cfg.__setstate__(cfg_dict)
                    self.cfgs.append(new_cfg)
            else:
                setattr(self, k, v)
        
        # Updating the normalizer/tokens in the cfgs
        for cfg in self.cfgs:
            cfg.normalizer = self.normalizer
            cfg.tokens = self.tokens
    
    def __eq__(self, other):
        return isinstance(other, MemCFGDataset) and all(eq_obj(self, other, selector=s) for s in 
            ['normalizer', 'tokens', 'metadata', 'cfgs'])
    
    def __hash__(self):
        return sum(hash(c) for c in self.cfgs) * 11 + hash_obj([self.tokens, self.metadata], return_int=True) * 13

    def __str__(self):
        return "MemCFGDataset with %d cfg's and %s tokens" % (len(self.cfgs), len(self.tokens) if self.tokens is not None else 'UNKNOWN')
    
    def __str__(self):
        stat_names = ["CFG's", 'Functions', 'Edges', 'Basic Blocks', 'Assembly Lines']
        c = [self.num_cfgs, self.num_functions, self.num_edges, self.num_blocks, self.num_asm_lines]
        stats = _get_stats(stat_names, c)
        norm = ('with normalizer: %s' % self.normalizer) if self.normalizer is not None else 'with no normalizer'
        return "%s %s\nStats:\n%s" % (self.__class__.__name__, norm, stats)
    
    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        return len(self.cfgs)
    
    def __getitem__(self, idx):
        return self.cfgs[idx]
    
    def __iter__(self):
        return iter(self.cfgs)


def _get_stats(stat_names, counts):
    """Returns a nicely-printable set of statistics for the MemCFGDataset

    Args:
        stat_names (Iterable[str]): the names for each statistic
        counts (Iterable[int]): the values for each statistic
    
    Returns:
        str: the nicely formatted set of statistics for the MemCFGDataset
    """
    return '\n'.join([('\t' + name + ': ' + str(c)) for name, c in zip(stat_names, counts)])
