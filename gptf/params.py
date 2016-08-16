from .wrappedtf import WrappedTF

class Param(WrappedTF):
    """A parameter of a model.

    Attributes:
        value (np.ndarray): The value of the parameter.
        tensor (tf.Variable | tf.Tensor): A tensor (like) representation
            of the parameter, suitable for passing to TensorFlow ops.

        free_state (tf.Variable): The free state form of the parameter, that
            can be freely optimised.
        transform (.transforms.Transform): The transform used to move the
            variable into a free state where it can be optimised.

    Examples:
        Getting and setting values
        --------------------------

        The current value of the `Param` is stored internally as a 
        `tf.Variable` instance. When a numpy value is requested or set, 
        `WrappedTF.get_session()` is used to evaluate the fetch / set op.

        You can get and set the (numpy) value of a `Param` using its 
        `value` attribute:
        >>> p = Param(1.0)
        >>> p.value
        1.0
        >>> p.value = 2.0
        >>> p.value
        2.0

        The variable can be accessed using the `tensor` attribute:
        >>> p.tensor
        <tensorflow.python.ops.variables.Variable object at ...>

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

