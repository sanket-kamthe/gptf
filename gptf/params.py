from builtins import super

import tensorflow as tf

from .transforms import Transform, Identity
from .wrappedtf import WrappedTF

class FixedParameterError(Exception):
    """Raised when the free state of a fixed parameter is accessed."""

class Param(WrappedTF):
    """A parameter of a model.

    Attributes:
        value (np.ndarray): The value of the parameter.
        tensor (tf.Tensor): A tensor representation of the parameter, 
            suitable for passing to TensorFlow ops.
        free_state (tf.Variable): The free state form of the parameter, that
            can be freely optimised.
        fixed (bool): A flag indicating whether or not the variable is fixed.
            Fixed parameters will not be optimised.
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
        >>> with p.get_session() as sess:
        ...     print(sess.run(p.tensor))
        2.0

        Behind the scenes, this creates a `tf.Variable`, accessible using the
        `free_state` attribute. We can use this variable in a session like so:
        >>> with p.get_session() as sess:
        ...     print(".free_state: {}".format(sess.run(p.free_state)))
        ...     print(".value: {}".format(p.value))
        ...     # assigning to p.value changes p.free_state
        ...     p.value += 1.0
        ...     print(".free_state: {}".format(sess.run(p.free_state)))
        ...     print(".value: {}".format(p.value))
        ...     # assigning to p.free_state changes p.value
        ...     _ = sess.run(p.free_state.assign_add(1.0))
        ...     print(".free_state: {}".format(sess.run(p.free_state)))
        ...     print(".value: {}".format(p.value))
        .free_state: 2.0
        .value: 2.0
        .free_state: 3.0
        .value: 3.0
        .free_state: 4.0
        .value: 4.0

        The session returned by `p.get_session()` maintains the value of 
        `p.free_state` across uses:
        >>> with p.get_session() as sess:
        ...     print(sess.run(p.free_state))
        4.0

        If we have multiple `Param`s, as long as they are in the same
        tree of `WrappedTF`, each `Param`'s `.get_session()` will return
        the same session, and every `Param` in the tree will have its free
        state maintained in that session.
        >>> w = WrappedTF()
        >>> w.p = Param(1.0)
        >>> w.q = Param(2.0)
        >>> with w.p.get_session() as sess:
        ...     print(sess.run(w.p.free_state))
        ...     print(sess.run(w.q.free_state))
        1.0
        2.0

        It is possible to use `tf.Session()` instead of `p.get_session()`.
        In that case, we must run `p.on_session_birth()` after the session
        has been installed as the default session and `p.on_session_death()`
        just before the session closes.
        >>> p = Param(4.0)
        >>> with tf.Session() as sess:
        ...     p.on_session_birth()
        ...     print(".free_state: {}".format(sess.run(p.free_state)))
        ...     print(".value: {}".format(p.value))
        ...     # assiging to p.value changes p.free_state
        ...     p.value = .0
        ...     print(".free_state: {}".format(sess.run(p.free_state)))
        ...     print(".value: {}".format(p.value))
        ...     # assigning to p.free_state does not change p.value
        ...     _ = sess.run(p.free_state.assign_add(1.0))
        ...     print(".free_state: {}".format(sess.run(p.free_state)))
        ...     print(".value: {}".format(p.value))
        ...     p.on_session_death()
        .free_state: 4.0
        .value: 4.0
        .free_state: 0.0
        .value: 0.0
        .free_state: 1.0
        .value: 1.0

        We advise the reader to use `p.get_session()`.

        Attempting to set the tensor or free_state paremeters results in an
        error:
        >>> p.tensor = tf.constant(1)
        Traceback (most recent call last):
            ...
        AttributeError: can't set attribute
        >>> p.free_state = tf.Variable(1)
        Traceback (most recent call last):
            ...
        AttributeError: can't set attribute

        Fixing parameters
        -----------------

        A parameter can be fixed by setting the `.fixed` attribute to `True`.
        A fixed parameter should not be optimised. Attempting to
        access the `.free_state` attribute of a fixed parameter will result
        in a `FixedParameterError`:
        >>> p = Param(2.0)
        >>> p.fixed = True
        >>> p.free_state
        Traceback (most recent call last):
            ...
        gptf.params.FixedParameterError: message

        Ultimately, however, it is the responsibility of the optimiser to 
        respect this flag. See the `Parameterised` and `Model` classes for
        more details.

        Transforms
        ----------

        Constraints can be applied to a parameter in the form of `Transform`s.
        A `Transform` is used to transform the parameter into a free state,
        where it can then be optimized. The transform can be set either by
        specifying `transform` paramater of the constructor or after creation
        using the `.transform` attribute. The default transform is
        `gptf.transforms.Identity`.
        
        >>> from gptf.transforms import Exp, Identity
        >>> Param(1.0).transform
        gptf.transforms.Identity()
        >>> p = Param(1.0, transform=Exp())
        >>> p.transform
        gptf.transforms.Exp(lower=1e-06)
        >>> p.transform = Identity()
        >>> p.transform
        gptf.transforms.Identity()

        The associated free state can be obtained using the `.free_state`
        parameter.
        >>> p = Param(1.0, transform=Exp())
        >>> with p.get_session() as sess:
        ...     print(p.value)
        ...     print(sess.run(p.free_state))
        1.0
        -1e-06

        The free state can then be freely optimised, and `p.value` and 
        `p.tensor` will remain constrained by the transform.
        >>> p = Param(1.0, transform=Exp())  # p.value > 0
        >>> with p.get_session() as sess:
        ...     _ = sess.run(p.free_state.assign(-100))
        ...     print(p.value == sess.run(p.tensor))
        ...     print(p.value > 0)
        True
        True

        Tensorflow will take the transform into account when calculating
        the derivative of `p.tensor` w.r.t. its free state:
        >>> from math import e
        >>> p = Param(e)
        >>> grad_identity = tf.gradients([p.tensor], [p.free_state])[0]
        >>> with p.get_session() as sess:
        ...     print(sess.run(grad_identity))
        1.0
        >>> p.transform = Exp()
        >>> grad_exp = tf.gradients([p.tensor], [p.free_state])[0]
        >>> with p.get_session() as sess:
        ...     print("{:.3f}".format(sess.run(grad_exp)))
        2.718

    """
    def __init__(self, initial_value, transform=Identity()):
        super().__init__()
        self._numpy_value = initial_value
        self.fixed = False
        self._transform = transform
        
    @property
    def value(self):
        if self._variable:
            sess = self.get_session()
            return sess.run(self.tensor)
        else:
            return self._numpy_value

    @value.setter
    def value(self, value):
        if self._variable:
            sess = self.get_session()
            free_state = self.transform.np_backward(value)
            self._numpy_value = sess.run(self._variable.assign(free_state))
            return self._numpy_value
        else:
            self._numpy_value = value

    @property
    @WrappedTF.tf_method
    def tensor(self):
        """Returns a tensor representing the value of the parameter.
        
        Returns:
            (tf.Tensor) The forward transform of the parameter applied to its
            free state.
            
        """
        self._ensure_variable()
        return self.transform.tf_forward(self._variable)

    @property
    @WrappedTF.tf_method
    def free_state(self):
        """Returns a variable that maps to the free state of the parameter."""
        if not self.fixed:
            self._ensure_variable()
            return self._variable
        else:
            raise FixedParameterError("cannot access free state of fixed Param")

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, value):
        if self._variable:
            # anything that caches anything that relies on self.tensor needs
            # to clear its cache.
            self.clear_ancestor_caches()
            old_value = self.value
            self._transform = value
            self.value = old_value
        else:
            self._transform = value

    def on_session_birth(self):
        self._ensure_variable()
        sess = self.get_session()
        sess.run(self.initializer)
        super().on_session_birth()

    def on_session_death(self):
        assert self._variable
        self._numpy_value = self.value
        super().on_session_death()

    def clear_cache(self):
        """Save the variable value before it is cleared from the cache."""
        if self._variable:
            self._numpy_value = self.value
        super().clear_cache()

    @property
    @WrappedTF.tf_method
    def initializer(self):
        """Initialises the internal `tf.Variable` to the correct value.
        
        This op is automatically run for sessions obtained using 
        `.get_session()`.
        
        """
        self._ensure_variable()
        with tf.control_dependencies([self._variable.initializer]):
            free_state = self.transform.np_backward(self._numpy_value)
            return self._variable.assign(free_state)
            
    @WrappedTF.tf_method
    def _ensure_variable(self):
        """Creates a variable if necessary."""
        if not self._variable:
            self._variable = tf.Variable(self._numpy_value)
    
    @property
    def _variable(self):
        """Get the variable if there is one.
        
        Returns `None` (falsity) if there is no variable, or a `tf.Variable`
        (truth) if there is one. Hence variable existence may be checked using
        `if self._variable`.

        """
        return self.cache.get("_Param__variable", None)

    @_variable.setter
    def _variable(self, value):
        """Sets the variable."""
        self.cache["_Param__variable"] = value
