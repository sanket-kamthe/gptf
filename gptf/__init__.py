"""Provides a GPflow / GPy like interface for distributed GPs."""
# allow some core modules to be accessed using their names
from .core import transforms, densities, likelihoods, meanfunctions, tfhacks

# fetch important objects from core modules
from .core.models import Model, GPModel
from .core.params import DataHolder, Param, Parameterised, ParamList, autoflow
from .core.wrappedtf import tf_method

__author__ = "Blaine Rogers <br1314@ic.ac.uk>"
__version__ = "0"
