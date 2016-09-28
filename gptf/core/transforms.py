"""Provides a number of transforms that can be used to constrain `Param`\ s."""
from builtins import object
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod
import inspect

import numpy as np
import tensorflow as tf
from overrides import overrides

class Transform(with_metaclass(ABCMeta, object)):
    """A transform, used to constrain the optimisation of a `Param`.

    NB: Unless otherwise stated, attributes of transforms are read-only
    and should be treated as such. If you wish to change the transform
    of a `Param`, assign a *new* `Transform` to its `.transform` attribute.

    """
    @abstractmethod
    def tf_forward(self, x):
        """Map from the free space to the variable space using TensorFlow."""
        NotImplemented

    #@abstractmethod
    #def tf_backward(self, x):
    #    """Map from the variable space to the free space using TensorFlow."""
    #    NotImplemented

    #@abstractmethod
    #def np_forward(self, x):
    #    """Map from the free space to the variable space using NumPy."""
    #    NotImplemented

    @abstractmethod
    def np_backward(self, x):
        """Map from the variable to the free space using NumPy."""
        NotImplemented
    
class Identity(Transform):
    @staticmethod
    @overrides
    def tf_forward(x):
        return tf.identity(x)

    #@overrides
    #@staticmethod
    #def tf_backward(x):
    #    return tf.identity(x)

    #@overrides
    #@staticmethod
    #def np_forward(x):
    #    return x

    @staticmethod
    @overrides
    def np_backward(x):
        return x

    def __str__(self):
        return "identity"

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

    @overrides
    def tf_forward(self, x):
        return tf.exp(x) + self._lower

    #@overrides
    #def tf_backward(self, x):
    #    return tf.log(x - self._lower)

    #@overrides
    #def np_forward(self, x):
    #    return np.exp(x) + self._lower

    @overrides
    def np_backward(self, y):
        return np.log(y - self._lower)

    def __str__(self):
        return "+ve (Exp)"

    def __repr__(self):
        return "{}.Exp(lower={})".format(__name__, self._lower)
