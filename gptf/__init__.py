"""Provides a GPflow / GPy like interface for distributed GPs."""
# allow some core modules to be accessed using their names
from .core import transforms, densities, likelihoods, meanfunctions, kernels
from .core import tfhacks   # hopefully someday we can be rid of these

# fetch important objects from core modules
from .core.models import Model, GPModel
from .core.params import DataHolder, Param, Parameterized, \
        ParamAttributes, ParamList, autoflow
from .core.wrappedtf import tf_method

# make submodules visible
from . import gpr, distributed

__author__ = "Blaine Rogers <br1314@ic.ac.uk>"
__version__ = "0"
