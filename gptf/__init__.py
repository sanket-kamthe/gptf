"""Provides a GPflow / GPy like interface for distributed GPs."""
from . import params, wrappedtf, parentable, transforms
__author__ = "Blaine Rogers <br1314@ic.ac.uk>"
__version__ = "0"

# in aid of doctest discovery
__test__ = \
        { 'params': params
        , 'wrappedtf': wrappedtf
        , 'parentable': parentable
        , 'transforms': transforms
        }
