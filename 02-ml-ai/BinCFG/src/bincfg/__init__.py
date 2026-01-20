"""A Python tool designed to parse binary analyzer outputs to produce call flow graphs (CFG), tokenize and normalize the assembly lines within those CFGs, and convert that data into ML-ready formats."""

__version__ = '1.0.2'  # Change here, in setup.cfg, and in CITATION
__author__ = 'Justin Allen'

from .utils import *  # Keep this first!
from .cfg import *
from .normalization import *
