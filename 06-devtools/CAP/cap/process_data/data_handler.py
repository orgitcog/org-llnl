"""
Object to handle the updating/saving of CAP data
"""

import os
import re
import pickle
import pandas as pd
import datetime
import shutil
import traceback
from utils.logs import MPLogger
from utils.misc import CriticalError
from bincfg.normalization import get_normalizer
from bincfg import CFG, MemCFG, AtomicTokenDict
from hashlib import md5


# Time to wait before automatically saving
SAVE_TIME_MINUTES = 20

# Memory threshold in GB before saving a chunk
_MAX_MEM_USAGE_GB = 4.0

# The logger
LOGGER = MPLogger()


class CAPDataHandler():
    """A class to handle the data compilation process and hold/update all cap data.
    
    Will only generate the data necessary to save the specified data (EG: if you are only saving the binary/analyzer
    output, then CFG's/MemCFG's/stats will not be computed, however if you are saving only the 'stats' info, then all
    the data will be temporarily generated but only the stats will be saved)


    Parameters
    ----------
        paths: `Dict[str, str]`
            dictionary of paths
        exec_info: `Dict[str, Any]`
            dictionary of execution info. Should contain the keys/values:

                - 'execution_uid' (Union[str, int]): execution uid for this CAP process
                - 'task_id' (int): integer id of this task
                - 'postprocessing' (Optional[Union[str, List[str]]]): string or list of strings for the postprocessings 
                  to apply to analyzer outputs, or None to not apply any. Available strings:

                  * 'cfg': build a CFG() object, one for each of the normalizers in exec_info['normalizers']
                  * 'memcfg': build a MemCFG() object, one for each of the normalizers in exec_info['normalizers']
                  * 'stats': build a CFG() object and get the graph statistics with cfg.get_compressed_stats(), one for
                    each of the normalizers in exec_info['normalizers']
                
                  NOTE: these will be stored as pickled bytes() objects

                - 'drop_columns' (Optional[Union[str, List[str]]]): by default, all of the data generated is kept. This
                  if not None, can be a string or list of strings of the column or group of columns to drop. You may also
                  pass any columns that would appear in the metadata, and those will be dropped. Metadata columns to 
                  drop can be passed either as their original name, or with the prefix 'meta_' as they would appear in
                  the output data. Any columns that do not correspond to data being kept will raise an error, unless they 
                  start with the prefix 'meta_', in which case it is assume that that column is a possible metadata column 
                  which may or may not exist. Available non-metadata columns to drop: 
                  
                  'analyzer', 'binaries', 'analyzer_output', 'metadata', 'error', 'compile_stdout', 'compile_stderr', 
                  'analyzer_stdout', 'analyzer_stderr', 'compile_time', 'analysis_time', 'language_family', 'compiler_family',
                  'compiler', 'compiler_version', 'architecture', 'flags'

                  There are also a couple of special strings that will drop groups of columns including:

                    * 'metadata': drop any metadata that was passed in metadata dictionaries
                    * 'compile_info': drop all of the compilation info
                    * 'timings': drop all of the timing information ('compile_time', 'analysis_time')
                    * 'stdio': drop all of the stdio information ('compile_stderr', 'analyzer_stdout', etc.)
                    * 'stdout': drop all of the stdout information ('compile_stdout', 'analyzer_stdout')
                    * 'stderr': drop all of the stderr information ('compile_stderr', 'analyzer_stderr')

                  See the README.md for what all of these columns are.

                - 'normalizers': normalizers to use when building postprocessing CFG's. Will build those CFG's once for
                  each of the normalizers here. Can be:

                  * None: will not normalize at all (use raw input) for CFG's, will use 'unnormalized' normalization
                    (default BaseNormalizer()) for MemCFG's
                  * str: string name of a normalizer to use
                  * Normalizer: Normalizer-like object to use
                  * Iterable of any of the above: will do one of each normalization for each datapoint
        
        use_precomputed_tokens: `bool`
            whether or not to use any precomputed tokens. These should exist in the output_dir and be pickle files
            containing a 2-tuple of: (normalizer_used, token_dictionary), filenames starting with "precomputed_tokens".
            Any file starting with that string will be checked to see if its normalizer is the same as one of the
            normalizers we are using, and will prepopulate the token dictionaries if so.
    """

    def __init__(self, paths, exec_info, use_precomputed_tokens=True):

        self.paths = paths

        # Get the normalizers to use
        norms = exec_info['normalizers']
        self.normalizers = [get_normalizer(norms)] if isinstance(norms, str) or hasattr(norms, 'normalize') \
            else [None] if norms is None else [get_normalizer(n) for n in norms]
        
        # Get all the data we will be keeping
        self.postprocessing = set(s.lower() for s in exec_info['postprocessing'])
        for k in self.postprocessing:
            if k not in ['cfg', 'memcfg', 'stats']:
                raise ValueError("Unknown postprocessing string: %s" % repr(k))
        self.drop_columns = set(s.lower() for s in exec_info['drop_columns'])
        if 'binary' in self.drop_columns:
            self.drop_columns.remove('binary')
            self.drop_columns.add('binaries')

        # Get all the precomputed tokens if using
        self.precomputed_tokens = []
        if use_precomputed_tokens and ('memcfg' in self.postprocessing or 'stats' in self.postprocessing or (len(self.normalizers) > 0 and 'cfg' in self.postprocessing)):
            LOGGER.info("Searching for precomputed tokens...")
            for f in [os.path.join(paths['atomic_data'], f) for f in os.listdir(paths['atomic_data']) if f.startswith('precomputed_tokens')]:
                LOGGER.info("Found a precomputed tokens, loading...")
                with open(f, 'rb') as _f:
                    norm, tokens = pickle.load(_f)
                    self.precomputed_tokens.append((norm, tokens))
            LOGGER.info("done loading precomputed tokens!")
        else:
            LOGGER.info("Refusing to use any precomputed tokens!")

        LOGGER.info("Using postprocessings: %s" % self.postprocessing)
        LOGGER.info("Using normalizers: %s" % self.normalizers)
        LOGGER.info("Dropping columns: %s" % self.drop_columns)

        self.df_idx = 0
        self.exec_uid = exec_info['execution_uid']
        self.init_time = datetime.datetime.now()
        self.last_time = datetime.datetime.now()
        self.previous_compile_finish_time = datetime.datetime.now()
        self.num_capped = 0
        self.num_failed = 0
        self.task_id = exec_info['task_id']
        self.norm_tokens = []
        self.df = pd.DataFrame()

    def add_data(self, data_dict, empty_temp=False):
        """Adds the info for a single datapoint, or list of datapoints

        Args:
            data_dict (Dict[str, List[Any]]): dictionary of mapping column names to lists of new values to add. Should 
                contain at minimum the key: 'id'. If postprocessings are being performed, then their required columns 
                should appear here as well. All values should be lists with the same lengths. All keys and their data 
                will be added. Some special keys may be used:

                    - 'compile_info': list of either None or dictionaries of compile information. The compile information
                      will be unpacked into the columns: 'language_family', 'compiler_family', 'compiler', 'compiler_version',
                      'architecture', 'flags'
                    - 'binary_file_paths': list of either None or lists of string binary file paths. These will be read
                      in as bytes and saved as a list of bytes() object for each datapoint under the new column name
                      'binaries'
                    - 'analyzer_output_path': list of either None or string path to analyzer output file. This analyzer
                      output will be loaded as string text and stored under the new column name 'analyzer_output'
                    - 'metadata': list of either None or dictionaries for datapoints. The keys and values within these 
                      dictionaries will be unpacked into columns in the main output dataframe with the prefix 'meta_'
                      added to their column name. 

            empty_temp (bool): if True, will empty all files/folders in the temp folder after adding this data
        """
        if 'id' not in data_dict:
            raise ValueError("Did not find 'id' key in data_dict")
        for k, v in data_dict.items():
            if len(v) != len(data_dict['id']):
                raise ValueError("Could not add new datapoints since lists had different sizes. 'id': %s, %s: %d"
                                 % (len(data_dict['id']), repr(k), len(v)))
        
        LOGGER.debug("Adding %d datapoints" % len(data_dict['id']))

        # Get the files we should delete (at the end, only after adding all the data)
        delfiles = set()
        if 'analyzer_output_path' in data_dict:
            delfiles.update([v for v in data_dict['analyzer_output_path'] if v is not None])
        if 'binary_file_paths' in data_dict:
            for bin_paths in data_dict['binary_file_paths']:
                if bin_paths is not None:
                    delfiles.update(bin_paths)
        
        # Read in the analyzer output if it is not None, and if we are not dropping this column.
        # Read it in anyways if we are doing postprocessing on this analysis output
        if 'analyzer_output_path' in data_dict and ('analyzer_output' not in self.drop_columns or len(self.postprocessing) != 0):
            analyzer_outputs = []
            for apath in data_dict['analyzer_output_path']:
                if apath is not None:
                    with open(apath, 'r') as f:
                        analyzer_outputs.append(f.read())
                else:
                    analyzer_outputs.append(None)
            data_dict['analyzer_output'] = analyzer_outputs
        
        # Get the compile information if it is not None
        if 'compile_info' in data_dict and 'compile_info' not in self.drop_columns:
            data_dict.update({
                'language_family': [v['language'] if v is not None else None for v in data_dict['compile_info']], 
                'compiler_family': [v['family'] if v is not None else None for v in data_dict['compile_info']],
                'compiler': [v['compiler'] if v is not None else None for v in data_dict['compile_info']],
                'compiler_version': [str(v['version']) if v is not None else None for v in data_dict['compile_info']],
                'architecture': [v['arch'] if v is not None else None for v in data_dict['compile_info']],
                'flags': [v['flags'] if v is not None else None for v in data_dict['compile_info']],
            })
            opt_levels = []
            for v in data_dict['compile_info']:
                if v is None:
                    opt_levels.append(None)
                    continue

                for f in v['flags']:
                    if re.fullmatch(r'-O(?:[0123sgz]|fast)', f) is not None:
                        opt_levels.append(f)
                        break
                else:
                    if v['compiler'] in ['gcc', 'g++']:
                        opt_levels.append('-O0')  # Apply the default -O0 if there wasn't one in the flags for GCC
                    else:
                        opt_levels.append('Unknown Compiler: %s' % repr(v['compiler']))
            data_dict['optimization_level'] = opt_levels
            
        # Read in the binaries if the paths are not None and we are not dropping this column
        if 'binary_file_paths' in data_dict and 'binaries' not in self.drop_columns:
            binaries = []
            for bin_list in data_dict['binary_file_paths']:
                if bin_list is None:
                    binaries.append(None)
                    continue

                next_list = []
                for bfp in bin_list:
                    with open(bfp, 'rb') as f:
                        next_list.append(f.read())
                binaries.append(next_list)
            data_dict['binaries'] = binaries

            # Some fun trickery here to make it a one-liner, just cause I can. I wish the writers of hashlib made their functions better...
            if 'binary_md5_hashes' not in self.drop_columns:
                data_dict['binary_md5_hashes'] = [([(lambda f, a: f.update(a) or f)(md5(), b).hexdigest().upper() for b in bl] \
                                                   if bl is not None else None) for bl in binaries]
            
            if 'total_size_mb' not in self.drop_columns:
                data_dict['total_size_mb'] = [(float(sum(len(b) for b in bl) / 2 ** 20) if bl is not None else None) for bl in binaries]
        
        # If we are saving cfg/memcfg/stats info, we do this once for each normalization method used, make sure we
        #   were able to read in the analyzer output
        if any(k in self.postprocessing for k in ['cfg', 'memcfg', 'stats']) and 'analyzer_output' in data_dict:
            
            # For all of these normalizers, use the prebuilt tokens for that normalizer, checking for precomputed
            #   tokens. This is all handled by self._get_norm_tokens
            for n_idx, norm in enumerate(self.normalizers):
                temp_cfgs = None

                # Build the cfg's for this normalizer if using
                if 'cfg' in self.postprocessing:
                    norm = get_normalizer(norm)
                    temp_cfgs = [_capture_err(CFG, ao, normalizer=norm, using_tokens=self._get_norm_tokens(norm, n_idx))
                                 if ao is not None else None for ao in data_dict['analyzer_output']]
                    data_dict['cfg-%s-%d' % (str(norm), n_idx)] = [(cfg.dumps() if cfg is not None else None) for cfg in temp_cfgs]
                
                # If we are doing a memcfg or stats postprocessing, then convert any None normalizers to 'unnormalized',
                #   and build the CFG's we need. If they were already built with this normalizer/norm_idx, use those,
                #   otherwise attempt to build new ones with this normalizer.
                if any(k in self.postprocessing for k in ['memcfg', 'stats']):
                    norm = get_normalizer('unnormalized') if norm is None else norm
                    norm_tokens = self._get_norm_tokens(norm, n_idx)

                    # Replace all CAPPostprocessingExceptions with None if it exists
                    if temp_cfgs is not None:
                        temp_cfgs = [(None if isinstance(t, Exception) else t) for t in temp_cfgs]
                    
                    # Get the temp_cfgs if it does not already exist, or if they are using a wrong normalizer
                    if temp_cfgs is None or len(temp_cfgs) == 0 or any(cfg is not None and cfg.normalizer != norm for cfg in temp_cfgs):
                        temp_cfgs = [_capture_err(CFG, ao, normalizer=norm, using_tokens=norm_tokens, return_none=True)
                                     if ao is not None else None for ao in data_dict['analyzer_output']]
                    
                    # Make the memcfg's and stats. We can build memcfg's directly since the cfg's should have already been normalized
                    if 'memcfg' in self.postprocessing:
                        data_dict['memcfg-%s-%d' % (str(norm), n_idx)] = [_capture_err(MemCFG, cfg, using_tokens=norm_tokens).drop_tokens().dumps() 
                                                                          if cfg is not None else None for cfg in temp_cfgs]
                    if 'stats' in self.postprocessing:
                        data_dict['stats-%s-%d' % (str(norm), n_idx)] = [_capture_err(cfg.get_compressed_stats, norm_tokens, ret_pickled=True)
                                                                         if cfg is not None else None for cfg in temp_cfgs]

        # Add in metadata. Find all keys used by these metadatas, and add them all as columns. Iterate through them adding
        #   key values if they exist, or None if they don't
        if 'metadata' in data_dict and 'metadata' not in self.drop_columns:
            total_keys = set(k for md in [m for m in data_dict['metadata'] if m is not None] for k in md)
            data_dict.update({'meta_%s' % k: [] for k in total_keys})
            for md in data_dict['metadata']:
                for k in total_keys:
                    data_dict['meta_%s' % k].append(md[k] if md is not None and k in md else None)
        
        def _dropdd(vals, raise_err=False):
            nonlocal data_dict
            for k in ([vals] if isinstance(vals, str) else vals):
                if k not in data_dict:
                    if raise_err:
                        raise KeyError("Could not find key to drop in data: %s" % repr(k))
                else:
                    del data_dict[k]
        
        # Delete any of the special columns if they exist
        _dropdd(['compile_info', 'binary_file_paths', 'analyzer_output_path', 'metadata'], raise_err=False)
        
        # Delete any of the drop columns
        for k in self.drop_columns:
            # Columns that may have already been dropped, so it's fine if they aren't there
            if k in ['compile_info', 'binary_file_paths', 'analyzer_output_path', 'metadata', 'language_family', 'compiler_family', 
                        'compiler', 'compiler_version', 'architecture', 'flags', 'optimization_level', 'binaries', 'analyzer_output']:
                _dropdd(k, raise_err=False)
            
            # Columns that drop special things, fine if they aren't already there
            elif k in ['timing', 'timings']:
                _dropdd(['compile_time', 'analysis_time'], raise_err=False)
            elif k in ['stdio']:
                _dropdd(['compile_stdout', 'compile_stderr', 'analyzer_stdout', 'analyzer_stderr'], raise_err=False)
            elif k in ['stdout']:
                _dropdd(['compile_stdout', 'analyzer_stdout'], raise_err=False)
            elif k in ['stderr']:
                _dropdd(['compile_stderr', 'analyzer_stderr'], raise_err=False)
            
            # If this is a metadata column, it's fine if the column doesn't exist
            elif k.startswith('meta_'):
                _dropdd(k, raise_err=False)
            
            # Otherwise, assume this column should exist
            else:
                # Check if this is a metadata column that doesn't have the 'meta_' in front
                try:
                    _dropdd('meta_' + k, raise_err=True)
                except:
                    # Final attempt: check if it is a normal column
                    _dropdd(k, raise_err=True)
            
        # Add this data to the dataframe. This is a slow way of doing it, but I figure it ought to be far faster
        #   than it takes to compile/analyze something, and plus it allows us to easily call df.memory_usage(),
        #   so it should be fine
        self.df = pd.concat((self.df, pd.DataFrame(data_dict)))

        # Delete files/folders if needed
        if empty_temp:
            for dir_key in ['temp']:
                shutil.rmtree(self.paths[dir_key])
                os.mkdir(self.paths[dir_key])
        
        # Check if we are passed memory limit and should save
        df_usage = self.df.memory_usage(deep=True).sum() / 2 ** 30
        if df_usage > _MAX_MEM_USAGE_GB:
            LOGGER.info("Memory usage threshold passed! Saving data for task_id: %d, df_idx: %d..." % (self.task_id, self.df_idx))
            self.save(empty=True)
            self.last_time = datetime.datetime.now()

        # Check if it's been a while and we should save
        elif (datetime.datetime.now() - self.last_time) > datetime.timedelta(minutes=SAVE_TIME_MINUTES):
            LOGGER.info("Enough time has passed! Saving data for task_id: %d, df_idx: %d..." % (self.task_id, self.df_idx))
            self.save(empty=False)
            self.last_time = datetime.datetime.now()

        total_time = (datetime.datetime.now() - self.init_time).total_seconds()
        this_time = (datetime.datetime.now() - self.previous_compile_finish_time).total_seconds()
        self.previous_compile_finish_time = datetime.datetime.now()
        self.num_capped += len(data_dict['id'])
        LOGGER.debug("This compilation took: %.4f seconds. Current average time: %.4f seconds/compilation. Current memory usage: %.4f GB" 
                     % (this_time, total_time / (self.num_capped + self.num_failed), df_usage))
    
    def _get_norm_tokens(self, norm, norm_idx):
        """Returns the tokens associated with that normalizer, loading in preloded tokens"""
        if norm is None:
            return norm
        
        # Check if we've already loaded this normalizer
        for n, tokens in self.norm_tokens:
            if n == norm:
                return tokens
        
        # Otherwise, make a new tokens, check if they're in the preloaded tokens
        tokens = AtomicTokenDict(filepath=os.path.join(self.paths['atomic_data'], "%s-%s-%d-tokens.pkl" % (self.exec_uid, norm, norm_idx)))
        for n, preloaded in self.precomputed_tokens:
            if n == norm:
                LOGGER.debug("Found precomputed tokens with %d tokens for normalizer: %s" % (len(preloaded), repr(str(norm))))
                tokens.update(preloaded)
                break
        else:
            LOGGER.debug("No precomputed tokens found for normalizer: %s" % repr(str(norm)))
        self.norm_tokens.append((norm, tokens))

        return tokens

    def save(self, empty=False):
        """Saves the data to a parquet file
        
        Args:
            empty (bool): if True, then the data will be emptied after saving
        """
        # If, for whatever reason, this fails, raise a big error cause we can't trust our ability to save
        try:
            if len(self.df) == 0:
                LOGGER.info("Dataset empty!")
                return

            LOGGER.info("Saving data for task_id: %d, df_idx: %d, %d datapoints currently taking up %.4f GB of memory..." 
                        % (self.task_id, self.df_idx, len(self.df), self.df.memory_usage(deep=True).sum() / 2 ** 30))

            save_path = os.path.join(self.paths['output'], "%s-%d-%d.parquet" % (self.exec_uid, self.task_id, self.df_idx))
            self.df.to_parquet(save_path)
            
            # Empty the data if needed, update the df_idx
            if empty:
                self.df = pd.DataFrame()
                self.df_idx += 1

            LOGGER.info("Saving complete! Saved to path: %s" % repr(save_path))
        except Exception as e:
            raise CriticalError(e)


def _capture_err(func, *args, return_none=False, err_message='', **kwargs):
    """Captures error, returning a new exception. Can optionally return None if return_none=True on error"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if return_none:
            return None
        return CAPPostprocessingException(func, args, kwargs, err_message)


class CAPPostprocessingException(Exception):
    """For an error during postprocessing. Adds a .dumps() method that returns bytes of the string traceback of this exception,
       and a .drop_tokens() method that just returns self"""
    def __init__(self, func, args, kwargs, err_message):
        super().__init__("Error during CAP postprocessing in data handler when calling function %s. Message: %s" 
                         % (repr(func.__name__), repr(err_message)))
        self._traceback = traceback.format_exc()
    
    def dumps(self):
        return ("%s: %s\nTraceback: %s" % (type(self).__name__, str(self), self._traceback)).encode('utf-8')
    
    def drop_tokens(self):
        return self
