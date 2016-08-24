from builtins import super
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod
from functools import partial
import os

import numpy as np
import tensorflow as tf
from overrides import overrides

from .trees import Leaf
from .transforms import Transform, Identity
from .wrappedtf import WrappedTF
from .utils import flip, isclassof, isattrof, construct_table, prefix_lines

class FixedParameterError(Exception):
    """Raised when the free state of a fixed parameter is accessed."""

class ShapeChangeError(Exception):
    """Raised when the shape of a Param or DataHolder changes."""

class ShapeChangeError(Exception):
    """Raised when the shape of a Param or DataHolder changes."""


class WrappedValue(with_metaclass(ABCMeta, WrappedTF)):
    """A class with a wrapped NumPy value, accessible through `.value`.

    Attributes:
        value (np.ndarray): The value of the parameter.
        on_shape_change ('raise' | 'pass' | 'recompile'): The action to take
            when the shape of the data changes; see the setter for `.value`.
        on_shape_change ('raise' | 'pass' | 'recompile'): The action to take
            when the shape of the data changes; see the setter for `.value`.

    """
    __ON_SHAPE_CHANGE_VALUES = ('raise', 'pass', 'recompile')
    __ON_DTYPE_CHANGE_VALUES = ('raise', 'pass', 'recompile')

    def __init__(self, on_shape_change='raise', on_dtype_change='raise'):
        """Initialiser.

        Args:
            on_shape_change ('raise' | 'pass' | 'recompile'): The initial
                value for `.on_shape_change`. Defaults to 'raise'.
            on_dtype_change ('raise' | 'pass' | 'recompile'): The initial
                value for `.on_dtype_change`. Defaults to 'raise'.

        """
        super().__init__()
        self.on_shape_change = on_shape_change
        self.on_dtype_change = on_dtype_change

    @abstractmethod
    def _get_value(self):
        """Gets (a copy of) the hidden numpy value."""
        NotImplemented

    @abstractmethod
    def _set_value(self, value):
        """Sets the hidden numpy value."""
        NotImplemented

    @property
    def value(self):
        return self._get_value()

    @value.setter
    def value(self, value):
        """Sets the value of the data.

        If the shape of the data changes, take one of the following actions
        depending on the value of `self.on_shape_change`:
          - on 'raise', raise a `gptf.params.ShapeChangeError`.
          - on 'recompile', clear the cache of everything higher in the tree.
          - on 'pass', do nothing.

        Examples:
            >>> class Example(WrappedValue):
            ...     def __init__(self, initial_value, **kwargs):
            ...         super().__init__(**kwargs)
            ...         self._numpy_value = initial_value
            ...     def _get_value(self):
            ...         return self._numpy_value.copy()
            ...     def _set_value(self, value):
            ...         self._numpy_value[...] = value

            Shape changes
            -------------

            On 'raise', we raise an error on shape change:
            >>> a = np.array([1,2,3])
            >>> b = np.array([1,2])
            >>> e = Example(a, on_shape_change='raise')
            >>> e.value = b
            Traceback (most recent call last):
                ...
            gptf.params.ShapeChangeError: message

            On 'recompile', we clear the compiled function cache of everything
            higher in the tree:
            >>> w = WrappedTF()
            >>> w.e = e
            >>> w.cache[0] = 123
            >>> w.e.on_shape_change = 'recompile'
            >>> w.e.value = b
            >>> 0 in w.cache
            False

            On 'pass', we do nothing and assign the new value anyway.
            >>> w.cache[0] = 123
            >>> w.e.value = a
            >>> 0 in w.cache
            True

        """
        self._new_shape_action(value)
        self._new_dtype_action(value)
        self._set_value(value)

    def _new_shape_action(self, value):
        """Performs the appropriate action given a new shape for `.value`."""
        if self._numpy_value.shape == np.shape(value):
            pass
        elif self.on_shape_change == 'raise':
            raise ShapeChangeError("cannot change shape of {}"\
                    .format(self.long_name))
        elif self.on_shape_change == 'recompile':
            self.clear_ancestor_caches()
            self.clear_cache()
        elif self.on_shape_change == 'pass':
            self.clear_cache()  # our variable has the wrong shape
        else:
            # this is more of a sanity check than anything else.
            raise ValueError("bad value of on_shape_change in value.setter???")

    def _new_dtype_action(self, value):
        """Performs the appropriate action given a new dtype for `.value`."""
        try:
            dtype = value.dtype
        except AttributeError as e:
            try:
                dtype = np.dtype(type(value))
            except TypeError:
                raise e
        if self._numpy_value.dtype == dtype:
            pass
        elif self.on_dtype_change == 'raise':
            raise DTypeChangeError("cannot change shape of {}"\
                    .format(self.long_name))
        elif self.on_dtype_change == 'recompile':
            self.clear_ancestor_caches()
            self.clear_cache()
        elif self.on_dtype_change == 'pass':
            self.clear_cache()  # our tensorflow object has the wrong shape
        else:
            # this is more of a sanity check than anything else.
            raise ValueError("bad value of on_shape_change in value.setter???")

    @property
    def on_shape_change(self):
        return self._on_shape_change

    @on_shape_change.setter
    def on_shape_change(self, value):
        """Checks that the new value of on_shape_change is valid."""
        if value not in self.__ON_SHAPE_CHANGE_VALUES:
            valuestr = ', '.join(map(repr, self.__ON_SHAPE_CHANGE_VALUES))
            raise ValueError("on_shape_change must be one of {}"
                    .format(valuestr))
        self._on_shape_change = value

    @property
    def on_dtype_change(self):
        return self._on_dtype_change

    @on_dtype_change.setter
    def on_dtype_change(self, value):
        """Checks that the new value of on_dtype_change is valid."""
        if value not in self.__ON_DTYPE_CHANGE_VALUES:
            valuestr = ', '.join(map(repr, self.__ON_DTYPE_CHANGE_VALUES))
            raise ValueError("on_shape_change must be one of {}"
                    .format(valuestr))
        self._on_dtype_change = value


class Param(WrappedValue, Leaf):
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
        on_shape_change ('raise' | 'pass' | 'recompile'): The action to take
            when the shape of the data changes; see the setter for `.value`.

    Examples:
        Getting and setting values
        --------------------------

        You can get and set the (numpy) value of a `Param` using its 
        `value` attribute:
        >>> p = Param(1.0)
        >>> p.value
        array(1.0)
        >>> p.value = 2.0
        >>> p.value
        array(2.0)

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
        ...     print("{:.3e}".format(sess.run(p.free_state)))
        1.0
        -1.000e-06

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
    def __init__(self, initial_value, transform=Identity(), **kwargs):
        """Initialiser.

        Args:
            initial_value (np.array_like): The initial value of the parameter.
            transform (gptf.transforms.Transform): The transform of the
                paramter. Defaults to `gptf.transforms.Identity()`.
            **kwargs: passed through to the constructor of WrappedValue.

        """
        super().__init__(**kwargs)
        self._numpy_value = np.array(initial_value)
        self.fixed = False
        self._transform = transform
        
    @overrides
    def _get_value(self):
        if self._variable:
            sess = self.get_session()
            return np.array(sess.run(self.tensor))
        else:
            return self._numpy_value.copy()

    @overrides
    def _set_value(self, value):
        if self._variable:
            sess = self.get_session()
            free_state = self.transform.np_backward(value)
            self._numpy_value[...] = sess.run(self._variable.assign(free_state))
        else:
            self._numpy_value[...] = value

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
        """The transform between the free space and value space."""
        return self._transform

    @transform.setter
    def transform(self, value):
        """Sets the transform of the parameter.

        If the parameter is in a variable, clears the cache of anything
        higher in the tree that might rely on `self.tensor`.
        
        Examples:
            >>> from .transforms import Identity, Exp
            >>> w = WrappedTF()
            >>> w.p = Param(1.)
            >>> w.cache[0] = 123
            >>> w.p.transform = Exp()
            >>> 0 in w.cache  # p has no variable, so no cache is cleared
            True
            >>> w.p.free_state  # force p to move into variable
            >>> w.p.transform = Identity
            >>> 0 in w.cache  # cache is cleared
            False

        """
        if self._variable:
            # anything that caches anything that relies on self.tensor needs
            # to clear its cache.
            self.clear_ancestor_caches()
            old_value = self.value
            self._transform = value
            self.value = old_value
        else:
            self._transform = value

    @overrides
    def on_session_birth(self):
        self._ensure_variable()
        sess = self.get_session()
        sess.run(self.initializer)
        super().on_session_birth()

    @overrides
    def on_session_death(self):
        assert self._variable
        self._numpy_value[...] = self.value
        super().on_session_death()

    def clear_cache(self):
        """Save the variable value before it is cleared from the cache."""
        if self._variable:
            self._numpy_value[...] = self.value
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


class DataHolder(WrappedValue, Leaf):
    """Holds data to be fed into TensorFlow for computation.

    Attributes:
        value (np.ndarray): The value of the data.
        placeholder (tf.placeholder): A placeholder for the data, 
            suitable for passing to TensorFlow ops.
        feed_dict (Dict[tf.placeholder, np.array_like]): A feed dictionary
            that feeds the value of the data into the placeholder op.
        on_shape_change ('raise' | 'pass' | 'recompile'): The action to take
            when the shape of the data changes; see the setter for `.value`.

    Examples:
        Getting and setting values
        --------------------------

        To get and set the value of the data, use the `.value` property:
        >>> a = np.array([1,2,3])
        >>> b = np.array([4,5,6])
        >>> d = DataHolder(a)
        >>> d.value
        array([1, 2, 3])
        >>> d.value = b
        >>> d.value
        array([4, 5, 6])

        See the docs for `gptf.params.WrappedValue` for more info.
        
        To access the value from TensorFlow, first build an op that relies
        on the `.placeholder` attribute:
        >>> d = DataHolder(a)
        >>> op = tf.add(d.placeholder, 1)

        Then evaluate the op in a session, passing in the feed dictionary
        to `tf.Session.run()`:
        >>> with d.get_session() as sess:
        ...     sess.run(op, feed_dict=d.feed_dict)
        array([2, 3, 4])
        
    """

    def __init__(self, initial_value, **kwargs):
        """Initialiser.

        Args:
            initial_value (np.array_like): The initial value of the data.
            **kwargs: passed through to the constructor of WrappedValue.

        """
        super().__init__(**kwargs)
        self._numpy_value = np.array(initial_value)
        self._placeholder = None

    @overrides
    def _get_value(self):
        return self._numpy_value.copy()

    @overrides
    def _set_value(self, value):
        self._numpy_value[...] = value

    @property
    def placeholder(self):
        self._assert_placeholder()
        return self._placeholder

    @property
    def feed_dict(self):
        self._assert_placeholder()
        return { self._placeholder: self._numpy_value }

    @property
    def _placeholder(self):
        return self.cache.get('_DataHolder__placeholder', None)

    @_placeholder.setter
    def _placeholder(self, value):
        self.cache['_DataHolder__placeholder'] = value

    def _assert_placeholder(self):
        if self._placeholder is None:
            dtype = tf.as_dtype(self._numpy_value.dtype)
            self._placeholder = tf.placeholder(dtype) #self._numpy_value.shape)


class Parameterized(WrappedTF):
    """An object that contains parameters and data.

    This object is designed to be part of a tree, with `Param`s and 
    `DataHolder`s at the leaves.

    Attributes:
        fixed (bool): A flag indicating whether or not any child `Param`s 
            should be fixed. Setting this attribute also sets the `.fixed` 
            attribute of anything lower in the tree.
        feed_dict (Dict[tf.placeholder, np.array_like]): A feed dictionary
            that feeds the values of DataHolders into their placeholder ops.
        params (List[Param]): A list of all the `Param`s lower in the tree,
            sorted by their long name.
        data_holders (List[Param]): A list of all the `DataHolder`s lower in 
            the tree, sorted by their long name.

    Examples:

    """
    def __init__(self):
        super().__init__()
        self._fixed = False

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, value):
        for node in filter(isattrof('fixed'), self):
            node.fixed = value

    @property
    def feed_dict(self):
        result = {}
        for node in self.data_holders:
            result.update(node.feed_dict)
        return result

    @property
    def params(self):
        """A sorted list of the `Param`s lower in the tree."""
        l = list(filter(isclassof(Param), self))
        l.sort(key=lambda x: x.long_name)
        return l

    @property
    def data_holders(self):
        """A sorted list of the `DataHolder`s lower in the tree."""
        l = list(filter(isclassof(DataHolder), self))
        l.sort(key=lambda x: x.long_name)
        return l

    @property
    def html(self):
        """Returns an html table representing the object."""

    def summary(self, array_len=5, fmt="fancy"):
        """A string table summarizing `self`.
        
        Args:
            array_len (int): The maximum number of elements to display
                of each array value.
            fmt ('fancy', 'plain', 'html'): The format for the table.
                `'fancy'` returns a table with fancy formatting using box
                drawing characters and ANSI terminal escape codes.
                `'plain'` returns a table using only ascii characters.
                `'html'` returns an html table.

        Returns:
            (str): A summary of this object's parameters and data.

        """
        lines = ["Parameterized object {}".format(self.long_name)]
        params = self.param_summary(fmt)
        if params:
            lines.extend(["", "Params:"])
            lines.append(prefix_lines("    ", params))
        data = self.data_summary(fmt)
        if data:
            lines.extend(["", "Data:"])
            lines.append(prefix_lines("    ", data))
        return os.linesep.join(lines)

    def param_summary(self, fmt="fancy"):
        """A string table summarizing the parameters of `self`.
        
        Args:
            fmt ('fancy', 'plain', 'html'): The format for the table.
                `'fancy'` returns a table with fancy formatting using box
                drawing characters and ANSI terminal escape codes.
                `'plain'` returns a table using only ascii characters.
                `'html'` returns an html table.

        Returns:
            (str): A string containing a table specifying the name,
            value, transform and prior of each parameter.

        """
        params = self.params
        if not params:
            return ""

        names = tuple(self._name_str(p.long_name) for p in params)
        values = tuple(self._value_str(p.value) for p in params)
        transforms = tuple(str(p.transform) for p in params)
        priors = tuple("nyi" for p in params)

        columns = (names, values, transforms, priors)
        headers = ("name", "value", "transform", "prior")

    def data_summary(self, array_len=5, fmt="fancy"):
        """A string table summarizing the data of `self`.
        
        Args:
            array_len (int): The maximum number of elements to display
                of each array value.
            fmt ('fancy', 'plain', 'html'): The format for the table.
                `'fancy'` returns a table with fancy formatting using box
                drawing characters and ANSI terminal escape codes.
                `'plain'` returns a table using only ascii characters.
                `'html'` returns an html table.

        Returns:
            (str): A string containing a table specifying the name,
            value, transform and prior of each parameter.

        """
        data = self.data_holders
        if not data:
            return ""

        names = tuple(self._name_str(d.long_name) for d in data)
        values = tuple(self._value_str(d.value) for d in data)

        columns = (names, values)
        headers = ("name", "value")

        return construct_table(columns, headers, fmt=fmt)

    @staticmethod
    def _name_str(name, fmt):
        if fmt == "fancy" and os.name != "nt":
            parts = name.split('.')
            parts[-1] = "\033[1m" + parts[-1] + "\033[0m"
            return '.'.join(parts)
        else:
            return name

    @staticmethod
    def _value_str(value, fmt):
        """Constructs a string representation of a numpy value."""
        if len(value.shape) > 1:
            return "<np.ndarray>"
        elif len(value.shape) == 1:
            if value.shape[0] > array_len:
                return "[" + ", ".join(map(str, value[:array_len])) \
                        + ", ...]"
            else:
                return "[" + ", ".join(map(str, value)) + "]"
        else:
            return str(value)

    def __str__(self):
        return self.summary()
