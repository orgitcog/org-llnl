"""Build/test manually checked CFG objects

All manual CFG function files should be in the same directory as this file, should be python files that start with 'cfg',
and should have a 'get_manual_cfg()' function that is used to build the CFG. See cfg.py for function description.
"""
import sys
import os
import importlib
sys.path.append(os.path.dirname(__file__))


__FOUND_CFG_FUNCS = None
def get_all_manual_cfg_functions():
    """Returns a list of all functions that can be used to build manual cfg's"""
    global __FOUND_CFG_FUNCS
    if __FOUND_CFG_FUNCS is None:
        __FOUND_CFG_FUNCS = []
        for f in [f for f in os.listdir(os.path.dirname(__file__)) if f.startswith('cfg') and f.endswith('.py')]:
            try:
                imp = importlib.import_module("tests.cfg.manual_cfgs." + f.replace('.py', ''))
                if hasattr(imp, 'get_manual_cfg') and callable(imp.get_manual_cfg):
                    __FOUND_CFG_FUNCS.append(imp.get_manual_cfg)
            except ImportError:
                pass
        print("\nFound %d manual cfg functions to test" % len(__FOUND_CFG_FUNCS))
    return __FOUND_CFG_FUNCS
