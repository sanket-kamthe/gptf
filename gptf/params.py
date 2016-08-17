from builtins import super
from .wrappedtf import WrappedTF

class UninitialisedParameterError(Exception):
    """Raised when an attempt is made to use an uninitialised parameter."""

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
        Initialisation
        --------------

        A `Param` must be initialised before it can be used. This should
        happen after it is installed in a model and the correct device
        placement and session target is set.

        >>> p = Param(1.0)
        >>> p.value
        Traceback (most recent call last):
            ...
        gptf.params.UninitialisedParameterError: ...
        >>> p.initialise()
        >>> p.value
        1.0

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

        A tensor value can be acquired using the `tensor` attribute.
        >>> isinstance(p.tensor, tf.Tensor)
        True

        The variable cannot be set, however.
        >>> p.tensor = tf.Variable(1)
        Traceback (most recent call last):
            ...
        AttributeError: can't set attribute

        Transforms
        ----------

        Constraints can be applied to a parameter in the form of `Transform`s.
        A `Transform` is used to transform the parameter into a free state,
        where it can then be optimized. The free state can be acquired using
        the `free_state` attribute.

    """
    def __init__(self, initial_value):
        super().__init__()
        self._variable = None
        self._value = initial_value
        
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def _assert_variable(self):
        """Asserts that the variable has been created.
        
        Raises:
            UninitialisedParameterError: if no variable exists.

        """
        if self._variable is None:
            raise UninitialisedParameterError('paremeter {long_name} must be \
initialised before use'.format(long_name=self.long_name))

    @WrappedTF.tf_method
    def get_variable(self):
        """Initialises the parameter."""
        #TODO: add object caching to WrappedTF
        # then store the variable in the WrappedTF cache
        # now WrappedTF can deal with clearing the cache if the 
        # architecture changes etc.
        self._variable = tf.Variable(self._value)

    @WrappedTF.tf_method
    def initialiser(self):
        self._assert_variable()
        return self._variable.initializer
    
    @property
    @WrappedTF.tf_method
    def tensor(self):
        return tf.Identity(self._variable)

    @property
    @WrappedTF.tf_method
    def free_state(self):
        return self._variable
