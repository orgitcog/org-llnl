"""Atomically update tokens"""
import os
import pickle
import warnings
import time
import datetime
import numpy as np
import bincfg
from .misc_utils import get_module


# Whether or not we warn about atomic data not being able to be loaded when unpickling
_WARN_ATOMIC_DATA = True

def _set_warn_atomic_data(val):
    global _WARN_ATOMIC_DATA
    _WARN_ATOMIC_DATA = val

_ATOMIC_READ_RAISE_ERR = object()

# The number of seconds to wait before attempting to aquire a lock after failing
AQUIRE_LOCK_FAIL_WAIT_TIME_SECONDS = 0.1


class AtomicData:
    """A class that allows for atomic reading/updating of the given data to a pickle file
    
    Parameters
    ----------
        init_data: `Any`
            Data to initialize the atomic file with. If the atomic file already exists, then that data will be loaded
        filepath: `Optional[str]`
            An optional filepath to store the dictionary, otherwise will be stored at './atomic_dict.pkl'
        lockpath: `Optional[str]`
            An optional filepath for the lock file to use to atomically update the dictionary, otherwise will be
                stored at './.[filepath].lock' where [filepath] is the given `filepath` parameter
        max_read_attempts: `Optional[int]`
            An optional integer specifying the maximum number of attempts to atomically read this dictionary before
                giving up and raising an error. Set to None to attempt indefinitely. Defaults to None
        delete_file: `bool`
            If True, then the file and lockfile will be deleted on initialization to start from scratch
    """

    def __init__(self, init_data, filepath=None, lockpath=None, max_read_attempts=None, delete_file=False):
        self._filepath = './atomic_data.pkl' if filepath is None else filepath
        self._lock_path = os.path.join(os.path.dirname(self._filepath), '.%s.lock' % os.path.basename(self._filepath)) if lockpath is None else lockpath
        self._lock = None

        if max_read_attempts is not None and max_read_attempts <= 0:
            raise ValueError("max_read_attempts must be > 0: %d" % max_read_attempts)
        self._max_read_attempts = 2**100 if max_read_attempts is None else max_read_attempts

        # Delete the files if starting from scratch
        if delete_file:
            self.delete_file(force=True)

        # Get the initial data
        self.atomic_read(default=init_data)
    
    def atomic_read(self, default=_ATOMIC_READ_RAISE_ERR):
        """Atomically reads the data from file, updating self.data
        
        Args:
            default (Optional[Any]): If this is passed and the file does not already exist, then this data will be saved
                to file and set to self.data
        """
        with _AquireLock(self._max_read_attempts, self._lock_path):

            # If the path doesn't exist, check if we need to raise an error, or update the file
            if not os.path.exists(self._filepath):
                if default is not _ATOMIC_READ_RAISE_ERR:
                    self.data = default
                    self._locked_write()
                else:
                    raise FileNotFoundError('Could not find inital atomic file to read from, and `default` data was not passed: %s' % self._filepath)
            
            # Otherwise it does exist, update self
            else:
                self.data = self._locked_read()
        
        return self.data
    
    def _locked_read(self):
        """Reads the data from file, assuming a lock has already been aquired"""
        with open(self._filepath, 'rb') as f:
            return pickle.load(f)
    
    def atomic_write(self):
        """Atomically writes the data at self.data to the pickle file"""
        with _AquireLock(self._max_read_attempts, self._lock_path):
            self._locked_write()
    
    def _locked_write(self):
        """Writes the data at self.data to file, assuming a lock has already been aquired"""
        with open(self._filepath, 'wb') as f:
            pickle.dump(self.data, f)
    
    def atomic_update(self, update_func, *update_args, **update_kwargs):
        """Atomically updates the data
        
        Will first aquire a lock on the data, read it in, then call `update_func(file_data, update_data)` where `file_data`
        is the data from the current atomic file, then write the data back to file and finally release the lock.

        NOTE: this will prevent any and all updates to the atomic file until update_func has completed

        NOTE: any errors within the update_func will be handled properly and will likely not mess up the atomic file

        Args:
            update_func (Callable): function that takes in: the data currently saved in file, the current data, then the 
                passed args and kwargs, and returns the updated data to write back to file
            update_args (Any): args to pass to update_func, after the current data saved in file
            update_kwargs (Any): kwargs to pass to update_func
        
        Returns:
            Any: the updated data
        """
        with _AquireLock(self._max_read_attempts, self._lock_path):
            self.data = update_func(self._locked_read(), self.data, *update_args, **update_kwargs)
            self._locked_write()
            return self.data
    
    def aquire_lock(self):
        """Aquires the lock needed to update data
        
        NOTE: this will prevent any and all updates to the atomic file until self.release_lock() is called. Make sure
        you call it quickly or other processes may hang!

        NOTE: if the lock has already been aquired, nothing will happen

        NOTE: it can be dangerous to attempt to aquire locks yourself, as any errors raised must be handled nicely and
        self.release_lock() must be called otherwise other processes may hang
        """
        if self._lock is None:
            self._lock = _AquireLock(self._max_read_attempts, self._lock_path).__enter__()
    
    def release_lock(self):
        """Releases the lock. Assumes it has already been aquired, otherwise an error will be raised"""
        if self._lock is None:
            raise ValueError("release_lock() was called, but the lock has not been aquired!")
        self._lock.__exit__()
        self._lock = None
    
    def delete_file(self, force=False):
        """Atomically deletes the file being used"""
        if force:
            if os.path.exists(self._filepath):
                os.remove(self._filepath)
            if os.path.exists(self._lock_path):
                os.remove(self._lock_path)
        else:
            with _AquireLock(self._max_read_attempts, self._lock_path):
                if os.path.exists(self._filepath):
                    os.remove(self._filepath)
    
    def __len__(self):
        """Gives length of current self.data"""
        return len(self.data)
    
    def __getstate__(self):
        """Doesn't send the actual data itself, that will be loaded"""
        ret = self.__dict__.copy()
        #del ret['data']
        return ret

    def __setstate__(self, state):
        """Set the state as normal, but read in the data when done"""
        for k, v in state.items():
            setattr(self, k, v)
        try:
            self._max_read_attempts, old = 50, self._max_read_attempts
            self.atomic_read()
            self._max_read_attempts = old
        except Exception as e:
            if _WARN_ATOMIC_DATA:
                warnings.warn("Could not load atomic data from file: %s, due to %s: %s. This data could be outdated!" % (self._filepath, type(e).__name__, e))


class _AquireLock:
    """Context manager to aquire a file lock, and remove it when done"""
    def __init__(self, max_attempts, lock_path):
        self._max_attempts, self._lock_path = max_attempts, lock_path
        get_module('atomicwrites', err_message='Package is required for atomic dictionary file!')
        from atomicwrites import atomic_write
        self.atomic_write = atomic_write
    
    def __enter__(self):
        rng = np.random.default_rng(seed=os.getpid() * int(time.time() * 1_000_000))
        for i in range(self._max_attempts):
            try:
                with self.atomic_write(self._lock_path, overwrite=False) as f:
                    f.write(datetime.datetime.now().isoformat())
                time.sleep(rng.random() * 0.2)  # Wait on average 0.1 seconds before trying again

                return self
            except FileExistsError:
                pass
            
            time.sleep(AQUIRE_LOCK_FAIL_WAIT_TIME_SECONDS)

        raise AquireLockError(self._max_attempts, self._lock_path)
            
    def __exit__(self, exc_type, exc_value, exc_tb):
        if os.path.exists(self._lock_path):
            os.remove(self._lock_path)


class AquireLockError(Exception):
    def __init__(self, attempts, lock_path):
        super().__init__("Could not aquire file lock from file after %d attempts using lock path: %s" % (attempts, lock_path))


class AtomicTokenDict:
    """Acts like a normal token dictionary, but allows for atomic operations
    
    Parameters
    ----------
        init_data: `Optional[Dict[str, int]]`
            Data to initialize the atomic token dict with. If the atomic file already exists, then that data will be loaded
        filepath: `Optional[str]`
            An optional filepath to store the dictionary, otherwise will be stored at './atomic_dict.pkl'
        lockpath: `Optional[str]`
            An optional filepath for the lock file to use to atomically update the dictionary, otherwise will be
                stored at './.[filepath].lock' where [filepath] is the given `filepath` parameter
        max_read_attempts: `Optional[int]`
            An optional integer specifying the maximum number of attempts to atomically read this dictionary before
                giving up and raising an error. Set to None to attempt indefinitely. Defaults to None
        delete_file: `bool`
            If True, then the file and lockfile will be deleted on initialization to start from scratch
    """

    def __init__(self, init_data=None, filepath=None, lockpath=None, max_read_attempts=None, delete_file=False):
        self._data = AtomicData(init_data={}, filepath=filepath, lockpath=lockpath, 
                                max_read_attempts=max_read_attempts, delete_file=delete_file)
        
        # Check to make sure init_data is a valid type, and there are no duplicate tokens
        if init_data is not None:
            if not isinstance(init_data, (dict, AtomicTokenDict)):
                raise TypeError("Can only initialize AtomicTokenDict with data of type 'dict' or 'AtomicTokenDict', not %s" 
                                % repr(type(init_data).__name__))
            
            found = {}
            for k, v in init_data.items():
                if v in found:
                    raise ValueError("Found tokens with duplicate values in init_data: %s" % [(found[v], v), (k, v)])
                if k in self and self[k] != v:
                    raise ValueError("Found token %s in init_data and in loaded atomic data with different values: %d != %d" % (repr(k), v, self[k]))
                found[v] = k
            
            self.update(init_data)
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __setitem__(self, key, value):
        if key in self.data:
            if self.data[key] != value:
                raise ValueError("Cannot set token key to a new value! key: %s, value: %s" % (repr(key), value))
            return
        
        self._atomic_update({key: value})
    
    def __contains__(self, key):
        return key in self.data
    
    def __len__(self):
        return len(self.data)
    
    def __str__(self):
        return repr(self)
    
    def __repr__(self):
        return repr(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    def update(self, tokens):
        """Updates this dictionary with the given tokens
        
        Args:
            tokens (Union[Dict[str, int], AtomicTokenDict]): dictionary mapping token strings to their integer values. 
                Any tokens in the dictionary that are not in this dictionary will be added, and any tokens that already 
                exist and have the same value will be ignored. If there are any tokens that already exist, but have a 
                different value, then an error will be raised
        """
        update_tokens = {}
        for k, v in tokens.items():
            if k in self.data:
                if self.data[k] != v:
                    raise ValueError("Cannot set token key to a new value! key: %s, value: %s" % (repr(k), v))
                continue
            update_tokens[k] = v
        
        if len(update_tokens) > 0:
            self._atomic_update(update_tokens)
    
    def items(self):
        return self.data.items()
    
    def values(self):
        return self.data.values()
    
    def keys(self):
        return self.data.keys()
    
    def get(self, key, default=None):
        return self.data.get(key, default=default)

    def setdefault(self, key, default=None):
        """If the key exists, return the value. Otherwise set the key to the given default (or len(self) if default=None)"""
        if key in self.data:
            return self.data[key]
        
        default = len(self) if default is None else default
        self._atomic_update({key: default})
        return default
    
    def addtokens(self, *tokens):
        """Adds the given tokens to this dictionary, ignoring any that already exist
        
        Args:
            tokens (str): arbitrary number of string tokens to add to this token dict
        """
        update_tokens = {}
        for t in tokens:
            if not isinstance(t, str):
                raise TypeError("Each token must be a string, not %s" % repr(type(t).__name__))
            if t not in self:
                update_tokens[t] = len(self) + len(update_tokens)
        
        if len(update_tokens) > 0:
            self._atomic_update(update_tokens)
    
    def _atomic_update(self, token_dict=None):
        """Atomically update the tokens from the given token_dict. Does no checks beforehand to see if there are any
        conflicts, duplicates, etc.
        
        Args:
            token_dict (Optional[Dict[str, int]]): token dictionary to update with, or None to just read in any updated
                tokens from file
        """
        self._data.atomic_update(bincfg.update_atomic_tokens, token_dict if token_dict is not None else {})
    
    def delete_file(self):
        "Deletes the atomic token dictinoary file"
        self._data.delete_file()

    @property
    def data(self):
        """Returns the token dictionary"""
        return self._data.data
    
    @property
    def inverse(self):
        """Return a new dict containing an inverse mapping of this current dictionary"""
        return {v: k for k, v in self.items()}
    
    @property
    def filepath(self):
        """Return the filepath being used to store the atomic data"""
        return self._data._filepath
    
    @property
    def lock_path(self):
        """Return the lock path being used to store the atomic data"""
        return self._data._lock_path
    
    def __hash__(self):
        import bincfg
        return bincfg.hash_obj(self.data)
