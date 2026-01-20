"""
Parsing/loading a container_info yaml file.
"""

import os
import yaml
import copy
import traceback
import subprocess
from functools import lru_cache
from compiler_selection.cf_actions import *
from utils.version import Version
from utils.misc import CriticalError
from collections import OrderedDict


_LEVELS = ['families', 'compilers', 'versions', 'architectures']

# The current types of information that can be stored in the container_info yaml file
CONTAINER_TYPES = ['compiler', 'analyzer']

# The directory path to mount within a container
MOUNTING_DIR = '/mounted'

# Dictionary mapping 'platform_name' to info dictionaries about each container platform. Each container platform info
#   dictionary should contain the keys: 'detection_command', 'bind_command', 'execution_command'. Values are:
#
#   - platform_name (str): string name of the container platform
#   - bind_command (str): command used to bind one path from host machine to another in container. Should take two format
#     kwargs strings: 'host_path' and 'container_path'
#   - detection_command (str): a command that will execute correctly iff that container platform is available for use, 
#     and fail otherwise
#   - execution_command (str): string used to execute a command within the container. Each should take formatting args:
#
#       * 'dir_bind': the bind command to bind the main working dir from the host to the container
#       * 'extra_args': any extra string args to pass on the command line while using container
#       * 'temp_dir': the temporary directory that houses the current data needed for compilation/analysis
#       * 'container': the path/name of the container being used
#       * 'command': the command being executed
#     
#     These should mount an outside directory to '/mounted' within the container
CONTAINER_PLATFORMS = {
    'singularity':  {'detection_command': 'singularity --version', 
                     'bind_command': '--bind {host_path}:{container_path} --containall',
                     'execution_command': 'singularity exec {dir_bind} {extra_args} {container} /bin/bash -c "{command}"'}, 
    'docker':       {'detection_command': 'docker --version', 
                     'bind_command': '-v {host_path}:{container_path}',
                     'execution_command': 'docker run --rm {dir_bind} {extra_args} {container} /bin/bash -c "{command}"'},
}


def read_container_info_yaml(path, ret_type=None):
    """Loads the yaml file, checks to make sure each outer object is good
    
    Args:
        path (str): path to yaml file to load
        ret_type (Optional[Union[str, Iterable[str]]]): the type of data to load. If None, will return all types of data found.
            Otherwise can be a string or iterable of strings specifying which type(s) of data to return. Available strings
            are those found in CONTAINER_TYPES: 'compiler', 'analyzer'
    """
    ret_type = CONTAINER_TYPES.copy() if ret_type is None else ret_type
    ret_type = [ret_type] if isinstance(ret_type, str) else list(ret_type)
    for t in ret_type:
        if t not in CONTAINER_TYPES:
            raise ValueError("Unknown `type` argument passed: %s\nAvailable types: %s" % (repr(t), repr(CONTAINER_TYPES)))
    
    if not os.path.exists(path):
        raise FileNotFoundError("Could not find configuration file at: %s" % repr(path))
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, dict):
        raise TypeError("Initial data loaded from container_info file must be a dict, not %s, in file: %s"
                        % (repr(type(data).__name__), repr(path)))
    
    # Pick out only the types we are returning, and check all inputs are valid
    ret = {}
    for k, v in data.items():
        if not isinstance(v, dict):
            raise TypeError("Top-level objects in the container_info file must be dictionaries. Got an object of "
                            "type %s for key: %s, in file: %s" % (repr(type(v).__name__), repr(k), repr(path)))
        if 'type' not in v:
            raise ValueError("Could not find 'type' key in dictionary for container_info object %s, in file: %s"
                             % (repr(k), repr(path)))
        if v['type'] not in CONTAINER_TYPES:
            raise ValueError("Unknown 'type' value %s found for container_info file object %s, in file: %s"
                             % (repr(v['type']), repr(k), repr(path)))
        
        if v['type'] in ret_type:
            ret[k] = v
    
    return ret


def load_analyzer_info(path):
    """Loads the analyzer info from the given container_info file
    
    Args:
        path (str): the path to the file to load
    """
    data = read_container_info_yaml(path, ret_type='analyzer')
    
    for k, v in data.items():
        for key in ['container', 'analysis_cmd']:
            if key not in v:
                raise ValueError("Could not find %s key in analyzer %s, in file: %s" % (repr(key), repr(k), repr(path)))
        v['name'] = k

        # Make sure the 'container' is valid
        if not isinstance(v['container'], dict):
            if not isinstance(v['container'], str):
                raise ValueError("Expected 'container' key's value to be str or dict, not %s, in analyzer %s, in file: %s"
                                 % (repr(type(v['container']).__name__), repr(k), repr(path)))
            v['container'] = {k: v['container'] for k in CONTAINER_PLATFORMS}
        for key, str_val in v['container'].items():
            if not isinstance(str_val, str):
                raise ValueError("Expected 'container' dictionary value for key %s to be str, not %s, in analyzer %s, in file: %s"
                                 % (repr(key), repr(type(str_val).__name__), repr(k), repr(path)))
    
    return data


def get_analyzer_methods(analyzer_names, analyzer_info):
    """Returns a list of dictionaries of analyzer info, one for each unique analyzer being used. Case-insensitive

    Values will be returned in the order they appear in analyzer_names. If there are no analyzers matching a given name,
    or there are multiple matching the case-insensitive name, an error will be raised.
    
    Args:
        analyzer_names (Union[str, Iterable[str]]): string name or Iterable of string names of analyzers to use. These
            should match (case-insensitive) the names of analyzers in the compiler_info file
        analyzer_info (List[Dict[str, Any]]): list of dictionaries of analyzer information. Each dict should at least
            container the key 'name' with the name of that analyzer
    
    Returns:
        List[Dict[str, Any]]: list of dictionaries of unique analyzers being used in the order that they appear in `analyzer_names`
    """
    analyzer_names = [analyzer_names.lower()] if isinstance(analyzer_names, str) else [n.lower() for n in analyzer_names]
    analyzer_names = OrderedDict(zip(analyzer_names, range(len(analyzer_names)))).keys()
    ret = []
    for an in analyzer_names:
        infos = [ai for k, ai in analyzer_info.items() if ai['name'].lower() == an]
        if len(infos) == 0:
            raise KeyError("Could not find analyzer with name: %s" % repr(an))
        elif len(infos) >= 2:
            raise ValueError("Fould multiple analyzers with the same case-insensitive name %s: %s" % (repr(an), [ai['name'] for ai in infos]))
        ret.append(infos[0])
    return ret


def load_compiler_info(path):
    """Loads the compiler info from the given container_info file

    Args:
        path (str): the path to the file to load
    
    Returns:
        dict: a dictionary of the compiler info. Will look like -

        {
            family1: {
                'compilers': {
                    
                    compiler1: {
                        'versions': {

                            version1: {
                                'architectures': {

                                    arch1: {

                                        'container': container_path,
                                        'binary_name': binary_path,
                                        'supported_languages': set(lang1, lang2, ...),
                                        'flags': {
                                            flag1: flag_val1,
                                            ...
                                        },
                                        'force_flags': {
                                            fflag1: fflag_val1,
                                            ...
                                        },

                                    },

                                }

                            },

                        }
                    },

                }
            },
        }
    """
    data = read_container_info_yaml(path, ret_type='compiler')
    
    # Load in all of the compiler families
    ret = {}
    for k in data:
        try:
            _load_level(k, ret, data, set(), 0, {})
        except Exception as e:
            tb = traceback.format_exc()
            raise ValueError("Error loading compiler family '%s' from path: %s\n\nError: %s: %s\nTraceback:\n%s" % (k, path, type(e).__name__, e, tb))
    
    # Do a final pass to clean things
    return _final_pass(ret)


def _final_pass(ret):
    """Does a final pass through the return dictionary cleaning things
    
    Currently:
        * Removes all flags that are set to None
        * Inserts string values as needed into paths and things
        * Changes all version string keys to Version() objects
    """
    for family, family_dict in ret.items():
        for compiler, compiler_dict in family_dict[_LEVELS[1]].items():
            for version, version_dict in compiler_dict[_LEVELS[2]].items():
                for arch, arch_dict in version_dict[_LEVELS[3]].items():

                    # Make sure there is a flag dict, and remove None flags
                    for k in ['flags', 'force_flags']:
                        arch_dict[k] = {} if k not in arch_dict else {fk: fv for fk, fv in arch_dict[k].items() if fv is not None}
                    
                    # Format strings
                    v = Version(version)
                    form_str = lambda s: s.format(family=family, compiler=compiler, version=version, arch=arch, 
                        architecture=arch, v=v, **{("v%d" % i): str(x) for i, x in enumerate(v.version_nums)})
                    arch_dict['binary_name'] = form_str(arch_dict['binary_name'])
                    arch_dict['container'] = {k: form_str(v) for k, v in arch_dict['container'].items()}
            
            compiler_dict[_LEVELS[2]] = {Version(k): v for k, v in compiler_dict[_LEVELS[2]].items()}
    return ret
    

def _load_level(name, loaded, raw_data, loading, level, metadata):
    """Recursively load the given level of data from a compiler info file

    For each level, we want to:
        1. Add this name to loading first to check for circular dependencies later
        2. Create the empty loaded data dictionary if it doesn't exist, and get the data to load if it does exist
        3. Check for a use_previous. If so:
            a. find the previous version to use
            b. check to make sure it is not already being loaded (circular dependency)
            c. load it if needed, copy it if already loaded
        4. Check for metadata. Can come in various types:
            a. metadata at this location from file should always override any from above, and any from use_previous
            b. metadata from use_previous will override that from above for only 'flags' and 'force_flags' keywords
            c. all other metadata from above will override that from use_previous
            d. Will also:
                * do some type checking and input sanitizing
    
    :param name: the name of this current value being loaded
    :param loaded: the data at the same level as to_load that has already been loaded
    :param raw_data: the raw data that contains to_load
    :param loading: a set of names that are currently being loaded (to track circular dependancies)
    :param level: the current level in the hierarchy
    :param metadata: the current metadata
    """
    # Add this name to loading
    loading.add(name)
    
    # Create the loaded dictionary for this name if it doesn't exist. Get the data to load if it does exist
    loaded.setdefault(name, {})
    to_load = raw_data[name] if name in raw_data else {}

    if to_load is None and _LEVELS[level] == 'versions':
        to_load = {'use_previous': True}

    # Check for a use_previous
    previous = _load_previous(name, to_load['use_previous'], loaded, raw_data, level, loading, metadata) if 'use_previous' in to_load else {}
    
    # Get the metadata being used at this level
    metadata = _load_metadata(name, loaded, to_load, level, metadata, previous)
    
    # Load those values that are lower down the hierarchy if needed
    if level + 1 < len(_LEVELS):
        _load_sublevel(name, loaded, to_load, level, metadata, previous)

    # Otherwise, set the metadata at the final level. Check to make sure we have all the required fields
    else:
        loaded[name].update(metadata)

        for k in ['container', 'binary_name', 'supported_languages']:
            if k not in loaded[name]:
                raise ValueError("Value '%s' at level '%s' did not contain necessary key '%s'" % (name, _LEVELS[level], k))


def _load_sublevel(name, loaded, to_load, level, metadata, previous):
    """Loads the next sublevel"""
    next_level_name = _LEVELS[level + 1]    
    to_load.setdefault(next_level_name, {})
    loaded[name].setdefault(next_level_name, {})

    loaded[name][next_level_name].update(previous.setdefault(next_level_name, {}))

    # Go through all lower levels, loading them from file if they exist, otherwise passing metadata downwards.
    for k in set(to_load[next_level_name].keys()).union(loaded[name][next_level_name].keys()):
        try:
            _load_level(k, loaded[name][next_level_name], to_load[next_level_name], set(), level + 1, metadata)
        except:
            raise ValueError("Error loading name '%s' which is at level '%s'" % (k, next_level_name))


def _load_metadata(name, loaded, to_load, level, metadata, previous):
    """Loads the current metadata
    
    Metadata can come in various types:
        a. metadata at this location from file should always override any from above, and any from use_previous
        b. metadata from use_previous will override that from above for only 'flags' and 'force_flags' keywords
        c. all other metadata from above will override that from use_previous
    Will also:
        * do some type checking and input sanitizing
    """
    metadata = copy.deepcopy(metadata)

    # Check for metadata from current location in file. 'Flags' and 'force_flags' will override previous, since those
    #   keys will later override that which is in metadata (and we want loaded data to take precedence). All keys will 
    #   override metadata since metadata should override previous for those keys.
    for k, v in to_load.items():

        # If the key is the next level, or is 'use_previous', ignore it
        if (level + 1 < len(_LEVELS) and k == _LEVELS[level + 1]) or k in ['use_previous']:
            continue

        elif k in ['flags', 'force_flags']:
            if not isinstance(v, dict):
                raise TypeError("%s value must be dictionary, not '%s'" % (repr(k), type(v).__name__))

            try:
                previous.setdefault(k, {})
                for fk, fv in to_load[k].items():
                    previous[k][fk] = _parse_flag(fk, fv)
            except:
                raise ValueError("Error updating flag '%s' to value: %s" % (k, v))

        elif k in ['container']:
            if isinstance(v, str):
                metadata[k] = {k: v for k in CONTAINER_PLATFORMS.keys()}
            elif isinstance(v, dict):
                metadata[k] = v
            else:
                raise TypeError("%s value must be a string or dictionary, not '%s'" % (repr(k), type(v).__name__))
            
        elif k in ['binary_name']:
            if isinstance(v, str):
                metadata[k] = v
            else:
                raise TypeError("%s value must be a string, not '%s'" % (repr(k), type(v).__name__))
        
        elif k in ['supported_languages']:
            if not isinstance(v, (str, list, tuple, set)):
                raise TypeError("%s value must be string, or list/tuple/set of string, not '%s'" % (repr(k), type(v).__name__))
            if isinstance(v, str):
                v = set([v])
            else:
                for o in v:
                    if not isinstance(o, str):
                        raise TypeError("Elements in %s must all be string, not %s" % (repr(k), type(o).__name__))
                v = set(v)
            
            metadata[k] = v
        
        # If this is a level-0 key (IE: at the top of the container definition) and a known key, continue
        elif level == 0 and k in ['type']:
            continue
        
        # Otherwise this is an unknown key, show an error
        else:
            raise ValueError("Unknown key %s" % repr(k))

    # Check for 'flags' and 'force_flags' to update into metadata
    for k in [k for k in ['flags', 'force_flags'] if k in previous]:
        metadata.setdefault(k, {}).update(previous[k])

    # All other flags in previous should only be written to metadata iff its name isn't already there
    for k in [k for k in previous if k not in metadata and k not in _LEVELS]:
        metadata[k] = previous[k]
    
    return metadata


def _load_previous(name, prev, loaded, raw_data, level, loading, metadata):
    """Check if a previous version has already been loaded and return a copy of it, otherwise load it then return the copy
    
    Will:
        * find the previous version to use
        * check to make sure it is not already being loaded (circular dependency)
        * load it if needed, copy it if already loaded
    """
    # If it's a string, we can just check to make sure it exists as something to load
    if isinstance(prev, str):
        if prev in raw_data:
            next_load = prev
        else:
            raise ValueError("Could not find use_previous to load: '%s'" % prev)

    # Otherwise it should be a boolean and only on 'version' level to find the most previous version
    elif isinstance(prev, bool):
        if _LEVELS[level] != 'versions':
            raise ValueError("Cannot parse boolean type 'use_previous' while not at 'version' level")
        
        all_versions = sorted([Version(k) for k in raw_data])
        idx = all_versions.index(Version(name))

        if idx == 0:
            raise ValueError("Cannot find an earlier version than '%s'" % name)

        nl_version = all_versions[idx - 1]
        next_load = [k for k in raw_data if Version(k) == nl_version][0]

    else:
        raise TypeError("Cannot parse 'use_previous' of type '%s'" % type(prev).__name__)
    
    # Check to see if it needs to be loaded
    if next_load not in loaded:
    
        # Check for a circular dependency
        if next_load in loading:
            raise CircularDependencyError("Found circular dependency chain loading values: %s" % repr(loading))
        
        # Otherwise we need to load it ourselves, then return a copy
        _load_level(next_load, loaded, raw_data, loading, level, metadata)

    # Return a copy of the data
    return {k: copy.deepcopy(v) for k, v in loaded[next_load].items()}


def _parse_flag(flag_name, flag_val):
    """Parses a single flag
    
    :param flag_name: the name of the flag, or None to not allow for bool 'no-' values
    :param flag_val: the flag value to parse. Can be int/string/list/tuple/bool/None
    """
    if flag_val is None:
        return None
    elif isinstance(flag_val, bool):  # Have to check bool's before int's because apparently, isinstance(True, int) is True (which is very dumb)
        if flag_name is None:
            raise TypeError("Attempting to parse flag list, found a bool value")
        return CFRandomNoFlag(flag_name) if flag_val else CFConstant(flag_name)
    elif isinstance(flag_val, (int, float)):
        return CFConcat([flag_name, flag_val])
    elif isinstance(flag_val, (list, tuple)):
        if len(flag_val) <= 1:
            raise ValueError("List length too small, should have at least 2 elements, found %d: %s" % (len(flag_val), flag_val))
        return CFConcat([flag_name, flag_val[0], CFChoice([_parse_flag('', v) for v in flag_val[1:]])])
    elif isinstance(flag_val, str):
        # Parse out the evaluated strings. If there are an even number of splits, then there is a mismatched '$$'
        if '$$' in flag_val:
            splits = flag_val.split('$$')
            if len(splits) % 2 == 0:
                raise ValueError("Mismatched '$$' in flag value: %s" % repr(flag_val))
            
            return CFConcat([flag_name] + [(eval(s) if i % 2 == 1 else s) for i, s in enumerate(splits)])
        else:
            return CFConcat([flag_name, flag_val])
    elif isinstance(flag_val, CFAction):
        return CFConcat([flag_name, flag_val])
    else:
        raise TypeError("Cannot parse flag of type '%s'" % type(flag_val).__name__)


class CircularDependencyError(Exception):
    pass


@lru_cache(maxsize=30)
def get_container_platform(container_platform):
    """cp can either be None to auto-detect, or a string. Returns a lowercase string"""
    if container_platform is None:
        for test_container_platform, (cmd_dict) in CONTAINER_PLATFORMS.items():
            proc = subprocess.Popen(cmd_dict['detection_command'], shell=True, stderr=subprocess.PIPE)
            _, cmd_out = proc.communicate()
            cmd_out = cmd_out.decode('utf-8')
            if len(cmd_out) == 0:
                container_platform = test_container_platform
                break
        
        else:
            raise CriticalError(ValueError("Could not automatically determine the job scheduler on this machine!"))

    if not isinstance(container_platform, str):
        raise CriticalError(ValueError("Expected `str` for container_platform not `%s`: %s" 
                                       % (type(container_platform).__name__, repr(container_platform))))
    container_platform = container_platform.lower()
    
    known_platforms = list(CONTAINER_PLATFORMS.keys())
    if container_platform not in known_platforms:
        raise CriticalError(ValueError("Unknown container platform: %s\nAvailable platforms: %s" % (container_platform, known_platforms)))
    
    return container_platform


def get_container_path(paths, obj_info, container_platform):
    """Returns the container path being used
    
    If using singularity, this will first check if the exact file exists, then it will check if the filepath + '.sif'
    exists (assuming it doesn't already end in '.sif')

    Otherwise if using docker, it will just assume the path name is the docker image name and return it

    Args:
        paths (Dict[str, Any]): the paths, we will use 'containers' here
        obj_info (Dict[str, Any]): the analyzer/compiler info. Should contain a 'container' key with the value being a
            dictionary, within which are key/values being the container platform name and container path respectively.
            Should also contain a 'name' key with that copmiler's name
        container_platform (str): the container platform name being used. If 'singularity', then an extra '.sif' will
            be added to the end of the returned container path if it doesn't already exist
    """
    if container_platform not in obj_info['container']:
        raise CriticalError(ValueError("Could not find the container path for the %s container platform for analyzer: %s"
                                       % (repr(container_platform), repr(obj_info['name']))))
    path = obj_info['container'][container_platform]

    if container_platform == 'singularity':
        path = os.path.join(paths['containers'], path)
        if not os.path.exists(path) or not os.path.isfile(path):
            if path.endswith('.sif'):
                raise CriticalError(ValueError("Cound not find singularity sif file at %s" % repr(path)))
            if not os.path.exists(path + '.sif'):
                raise CriticalError(ValueError("Could not find singularity sif file at %s, nor %s" % (repr(path), repr(path + '.sif'))))
            path = path + '.sif'
    
    return path
