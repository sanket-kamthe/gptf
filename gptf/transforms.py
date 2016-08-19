"""Provides a number of transforms that can be used to constrain `Param`s."""
from builtins import object
import inspect

import numpy as np
import tensorflow as tf

class Transform(object):
    """A transform, used to constrain the optimisation of a `Param`.

    NB: Unless otherwise stated, attributes of transforms are read-only
    and should be treated as such. If you wish to change the transform
    of a `Param`, assign a _new_ `Transform` to its `.transform` attribute.

    """
    def tf_forward(self, x):
        """Map from the free space to the variable space using TensorFlow."""
        NotImplemented

    #def tf_backward(self, x):
    #    """Map from the variable space to the free space using TensorFlow."""
    #    NotImplemented

    #def np_forward(self, x):
    #    """Map from the free space to the variable space using NumPy."""
    #    NotImplemented

    def np_backward(self, x):
        """Map from the variable to the free space using NumPy."""
        NotImplemented
    
class Identity(Transform):
    @staticmethod
    def tf_forward(x):
        return tf.identity(x)

    #@staticmethod
    #def tf_backward(x):
    #    return tf.identity(x)

    #@staticmethod
    #def np_forward(x):
    #    return x

    @staticmethod
    def np_backward(x):
        return x

    def __repr__(self):
        return "{}.Identity()".format(__name__)

class Exp(Transform):
    """An exponential transform.

    value = exp(free_state) + lower
    
    """
    def __init__(self, lower=1e-6):
        """Initialises the transform.

        Args:
            lower (float): The minimum value that this transform can take.
                Defaults to `1e-06`. This helps stability during optimisation,
                because some aggressive optimizers can take overly long steps
                which can lead to 0 in the transformed variable, causing an
                error.

        """
        self._lower = lower

    def tf_forward(self, x):
        return tf.exp(x) + self._lower

    #def tf_backward(self, x):
    #    return tf.log(x - self._lower)

    #def np_forward(self, x):
    #    return np.exp(x) + self._lower

    def np_backward(self, y):
        return np.log(y - self._lower)

    def __repr__(self):
        return "{}.Exp(lower={})".format(__name__, self._lower)
