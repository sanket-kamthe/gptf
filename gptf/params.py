from builtins import super

import tensorflow as tf

from .wrappedtf import WrappedTF

#class UninitialisedParameterError(Exception):
#    """Raised when an attempt is made to use an uninitialised parameter."""

class Param(WrappedTF):
    """A parameter of a model.

    Attributes:
        value (np.ndarray): The value of the parameter.
        tensor (tf.Tensor): A tensor representation of the parameter, 
            suitable for passing to TensorFlow ops.
        free_state (tf.Variable): The free state form of the parameter, that
            can be freely optimised.
        transform (.transforms.Transform): The transform used to move the
            variable into a free state where it can be optimised.

    Examples:
        Getting and setting values
        --------------------------

        You can get and set the (numpy) value of a `Param` using its 
        `value` attribute:
        >>> p = Param(1.0)
        >>> p.value
        1.0
        >>> p.value = 2.0
        >>> p.value
        2.0

        A tensor representing the paremeter can be acquired using the 
        `tensor` attribute.
        >>> isinstance(p.tensor, tf.Tensor)
        True

        Behind the scenes, this creates a `tf.Variable`, accessible using the
        `free_state` attribute. We can use this variable in a session like so:
        >>> with p.get_session() as sess:
        ...     print(sess.run(p.free_state))
        ...     # assigning to p.value changes p.free_state
        ...     p.value += 1.0
        ...     print(sess.run(p.free_state))
        ...     # assigning to p.free_state changes p.value
        ...     _ = sess.run(p.free_state.assign_add(1.0))
        ...     print(p.value)
        2.0
        3.0
        4.0

        The session returned by `p.get_session()` maintains the value of 
        `p.free_state` across uses:
        >>> with p.get_session() as sess:
        ...     print(sess.run(p.free_state))
        4.0

        Note also that we did not have to run an initializer for `p.free_state`.
        If we use `tf.Session()` instead of `p.get_session()`, we get a 
        different story entirely.

        >>> p.tensor = tf.Variable(1)
        Traceback (most recent call last):
            ...
        AttributeError: can't set attribute
        `p.get_session()`


        Transforms
        ----------

        Constraints can be applied to a parameter in the form of `Transform`s.
        A `Transform` is used to transform the parameter into a free state,
        where it can then be optimized. The free state can be acquired using
        the `free_state` attribute.

    """
    def __init__(self, initial_value):
        super().__init__()
        self._numpy_value = initial_value
        
    @property
    def value(self):
        if "variable" in self.cache:
            with self.get_session() as sess:
                return sess.run(self.cache["variable"])
        else:
            return self._numpy_value

    @value.setter
    def value(self, value):
        if "variable" in self.cache:
            with self.get_session() as sess:
                sess.run(self.cache["variable"].assign(value))
        else:
            self._numpy_value = value

    def on_session_birth(self):
        self._ensure_variable()
        with self.get_session() as sess:
            sess.run(self.cache["variable"].initializer)
            sess.run(self.cache["variable"].assign(self._numpy_value))

    def on_session_death(self):
        assert 'variable' in self.cache
        with self.get_session() as sess:
            self._numpy_value = sess.run(self.cache["variable"])

    def clear_cache(self):
        #TODO: Overwrite this method to save the value of "variable" to
        # the numpy value before clearing. be sensible.
        NotImplemented
            
    @WrappedTF.tf_method
    def _ensure_variable(self):
        """Creates a variable if necessary."""
        if "variable" not in self.cache:
            self.cache["variable"] = tf.Variable(self._numpy_value)
    
    @property
    @WrappedTF.tf_method
    def tensor(self):
        self._ensure_variable()
        return tf.identity(self.cache["variable"])

    @property
    @WrappedTF.tf_method
    def free_state(self):
        self._ensure_variable()
        return self.cache["variable"]
