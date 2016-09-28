# -*- encoding: utf-8 -*-
"""Provides classes that deal with the fetching as setting of parameters."""
from builtins import super, object, filter, map, range
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from functools import wraps
import os
import gc
from weakref import WeakSet
from contextlib import contextmanager

import numpy as np
import tensorflow as tf
from overrides import overrides

from .trees import AttributeTree, Leaf, ListTree, Tree
from .transforms import Transform, Identity
from .wrappedtf import WrappedTF, tf_method
from .utils import isclassof, isattrof, is_array_like
from .utils import construct_table, combine_fancy_tables, prefix_lines


#TODO: Implement priors.

class FixedParameterError(Exception):
    """Raised when the free state of a fixed parameter is accessed."""

class ShapeChangeError(Exception):
    """Raised when the shape of a Param or DataHolder changes."""

class DTypeChangeError(Exception):
    """Raised when the dtype of a Param or DataHolder changes."""

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
          - on 'raise', raise a `gptf.core.params.ShapeChangeError`.
          - on 'recompile', clear the cache of everything higher in the tree.
          - on 'pass', do nothing.

        Examples:
            >>> class Example(WrappedValue, AttributeTree):
            ...     def __init__(self, initial_value, **kwargs):
            ...         super().__init__(**kwargs)
            ...         self._numpy_value = initial_value
            ...     def _get_value(self):
            ...         return self._numpy_value.copy()
            ...     def _set_value(self, value):
            ...         self._numpy_value[...] = value

            .. rubric:: Shape changes

            On 'raise', we raise an error on shape change:

            >>> a = np.array([1,2,3])
            >>> b = np.array([1,2])
            >>> e = Example(a, on_shape_change='raise')
            >>> e.value = b
            Traceback (most recent call last):
                ...
            gptf.core.params.ShapeChangeError: message

            On 'recompile', we clear the compiled function cache of everything
            higher in the tree:

            >>> w = Example()
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
        value = np.asanyarray(value)
        self._new_shape_action(value)
        self._new_dtype_action(value)
        self._set_value(value)

    def _new_shape_action(self, value):
        """Performs the appropriate action given a new shape for `.value`."""
        if self.value.shape == value.shape:
            pass
        elif self.on_shape_change == 'raise':
            raise ShapeChangeError("cannot change shape of {}"\
                    .format(self.long_name))
        elif self.on_shape_change == 'recompile':
            self.clear_ancestor_caches()
            self.clear_cache()
        elif self.on_shape_change == 'pass':
            pass 
        else:
            # this is more of a sanity check than anything else.
            raise ValueError("bad value of on_shape_change in value.setter???")

    def _new_dtype_action(self, value):
        """Performs the appropriate action given a new dtype for `.value`."""
        if self.value.dtype == value.dtype:
            pass
        elif self.on_dtype_change == 'raise':
            raise DTypeChangeError("cannot change dtype of {}"\
                    .format(self.long_name))
        elif self.on_dtype_change == 'recompile':
            self.clear_ancestor_caches()
            self.clear_cache()
        elif self.on_dtype_change == 'pass':
            self.clear_cache()  # our tensorflow object has the wrong shape
        else:
            # this is more of a sanity check than anything else.
            raise ValueError("bad value of on_dtype_change in value.setter???")

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

class Proxy(Tree):
    """Places important state in an object shared between copies."""  
    class Shared(object):
        pass

    def __init__(self):
        super().__init__()
        self._shared = Proxy.Shared()
        self._shared.copies = WeakSet([self])

    @overrides
    def copy(self):
        copy = super().copy()
        assert copy._shared is self._shared
        self._shared.copies.add(copy)
        return copy

@contextmanager
def no_gc():
    gc.disable()
    yield
    gc.enable()
        
def share_properties(*properties):
    """Shares properties of a `Proxy` subclass between all copies.

    Args:
        *properties (Tuple[str]): The (string) names of the properties
            to override.
        
    Examples:
        Here we set up an example class with a property and an
        inheriting class that overrides that property.

        >>> @share_properties('a')
        ... class Example(Proxy, Leaf):
        ...     @property
        ...     def a(self):
        ...         print('getter called')
        ...         return self._a
        ...     @a.setter
        ...     def a(self, value):
        ...         print('setter called')
        ...         self._a = value
        ...     @a.deleter
        ...     def a(self):
        ...         print('deleter called')
        ...         del self._a

        The inheriting class shares the property between the copies
        using the property's getter/setter/deleter methods.

        >>> e = Example()
        >>> copies = [e.copy() for _ in range(3)]
        >>> e.a = 1
        setter called
        setter called
        setter called
        setter called
        >>> all(copy.a == 1 for copy in copies)
        getter called
        getter called
        getter called
        True
        >>> del e.a
        deleter called
        deleter called
        deleter called
        deleter called

    """
    def wrapper(class_):
        for name in properties:
            def get_property(name=name):
                # closures!
                wrappedproperty = getattr(class_, name)
                def getprop(self):
                    return wrappedproperty.fget(self)
                def setprop(self, value):
                    with no_gc():
                        for copy in self._shared.copies:
                            wrappedproperty.fset(copy, value)
                def delprop(self):
                    with no_gc():
                        for copy in self._shared.copies:
                            wrappedproperty.fdel(copy)
                doc = wrappedproperty.__doc__
                prop = property(getprop, setprop, delprop, doc)
                return prop
            setattr(class_, name, get_property())
        return class_
    return wrapper

@share_properties('tf_device', 'tf_graph', 'tf_session_target',
                  'on_shape_change', 'on_dtype_change')
class ProxyWrappedValue(WrappedValue, Proxy):
    """Sync various useful properties between copies.
    
    Shares one cache between all copies, and shares the `.tf_device`, 
    `.tf_graph`, `.tf_session_target`, `.on_shape_change` and
    `.on_dtype_change` parameters.
    
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shared.cache = {}

    @property
    def cache(self):
        return self._shared.cache

    @cache.setter
    def cache(self, value):
        self._shared.cache = value

    @cache.deleter
    def cache(self):
        del self._shared.cache

    def clear_all_ancestor_caches(self):
        """Clears the caches of the ancestors of all copies."""
        with no_gc():
            for copy in self._shared.copies:
                copy.clear_ancestor_caches()

class Param(ProxyWrappedValue, Leaf):
    """A parameter of a model.

    Instances have all attributes of `WrappedValue` and the following:

    Attributes:
        value (np.ndarray): The value of the parameter.
        tensor (tf.Tensor): A tensor representation of the parameter, 
            suitable for passing to TensorFlow ops.
        feed_dict (Dict): Currently an empty dictionary.
        free_state (tf.Variable): The free state form of the parameter, that
            can be freely optimised.
        fixed (bool): A flag indicating whether or not the variable is fixed.
            Fixed parameters will not be optimised.
        transform (.transforms.Transform): The transform used to move the
            variable into a free state where it can be optimised.

    Examples:
        .. rubric:: Getting and setting values

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

        >>> class AttributeWrappedTF(WrappedTF, AttributeTree):
        ...     pass
        >>> w = AttributeWrappedTF()
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
        ...     p.on_session_birth(sess)
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
        ...     p.on_session_death(sess)
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

        .. rubric:: Fixing parameters

        A parameter can be fixed by setting the `.fixed` attribute to `True`.
        A fixed parameter should not be optimised. Attempting to
        access the `.free_state` attribute of a fixed parameter will result
        in a `FixedParameterError`:

        >>> p = Param(2.0)
        >>> p.fixed = True
        >>> p.free_state
        Traceback (most recent call last):
            ...
        gptf.core.params.FixedParameterError: message

        Ultimately, however, it is the responsibility of the optimiser to 
        respect this flag. See the `Parameterised` and `Model` classes for
        more details.

        .. rubric:: Transforms

        Constraints can be applied to a parameter in the form of `Transform`s.
        A `Transform` is used to transform the parameter into a free state,
        where it can then be optimized. The transform can be set either by
        specifying `transform` paramater of the constructor or after creation
        using the `.transform` attribute. The default transform is
        `gptf.transforms.Identity`.
        
        >>> from gptf import transforms
        >>> Param(1.0).transform
        gptf.core.transforms.Identity()
        >>> p = Param(1.0, transform=transforms.Exp())
        >>> p.transform
        gptf.core.transforms.Exp(lower=1e-06)
        >>> p.transform = Identity()
        >>> p.transform
        gptf.core.transforms.Identity()

        The associated free state can be obtained using the `.free_state`
        parameter.

        >>> p = Param(1.0, transform=transforms.Exp())
        >>> with p.get_session() as sess:
        ...     print(p.value)
        ...     print("{:.3e}".format(sess.run(p.free_state)))
        1.0
        -1.000e-06

        The free state can then be freely optimised, and `p.value` and 
        `p.tensor` will remain constrained by the transform.

        >>> p = Param(1.0, transform=transforms.Exp())  # p.value > 0
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
        >>> p.transform = transforms.Exp()
        >>> grad_exp = tf.gradients([p.tensor], [p.free_state])[0]
        >>> with p.get_session() as sess:
        ...     print("{:.3f}".format(sess.run(grad_exp)))
        2.718

        .. rubric:: Copies

        You can create a copy of a `Param` using `Param.copy()`. The
        new copy represents the same parameter, so the value and
        various items of state (see below) are the same.

        >>> p = Param(1.0, transform=transforms.Exp())
        >>> copy = p.copy()

        Copies have the same value as the original, and updating the
        value of the copy updates the value of the original.

        >>> np.array_equal(p.value, copy.value)
        True
        >>> p.value = 5.0
        >>> copy.value
        array(5.0)

        Copies have the same transform, and setting the transform
        of a copy sets the transform of the original.

        >>> copy.transform is p.transform
        True
        >>> copy.transform = transforms.Identity()
        >>> p.transform
        gptf.core.transforms.Identity()

        Copies have the same "fixed" flag. Fixing a copy fixes the
        original and vice-versa.

        >>> p.fixed = True
        >>> copy.fixed
        True
        >>> copy.fixed = False
        >>> p.fixed
        False

        Copies have the same free state.

        >>> p.free_state is copy.free_state
        True

        This means that it is *very important* that every copy of of a
        `Param` has the same device context and graph, or odd things
        happen.

        >>> class AttributeWrappedTF(AttributeTree, WrappedTF):
        ...     pass
        >>> treeparam = Param(1.0)
        >>> w = AttributeWrappedTF()
        >>> w.param = treeparam
        >>> w.child = AttributeWrappedTF()
        >>> w.child.param = treeparam  # creates a copy of the param
        >>> w.tf_graph = tf.Graph()
        >>> w.tf_device = '/job:spoon/task:0'
        >>> w.child.tf_device = '/job:knife/task:0'
        
        In the above example, where should we place
        `treeparam.free_state`? Should it be on `'/job:spoon/task:0'`
        where `w.param` is, or on `'/job:knife/task:0'` where
        `w.child.param` is? Currently, there is no system in place
        to resolve this. In this situation, one should define the 
        device context of `treeparam` explicitly:

        >>> treeparam.tf_device = '/job:spoon'

        To create a new param with the same value, pass the value to
        `Param.__init__()`.

        >>> notacopy = Param(p.value, transform=p.transform)

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
        self._shared.numpy_value = np.array(initial_value)
        self._shared.transform = transform
        self._shared.fixed = False
        self._session = None

    @tf_method(cache=False)
    @overrides
    def _get_value(self):
        if self._variable and self._session:
            return self._session.run(self.tensor)
        else:
            return self._shared.numpy_value.copy()

    @tf_method(cache=False)
    @overrides
    def _set_value(self, value):
        if self._variable and self._session:
            sess = self._session
            free_state = self.transform.np_backward(value)
            self._shared.numpy_value[...] = \
                    sess.run(self._variable.assign(free_state))
        else:
            self._shared.numpy_value[...] = value

    @property
    @tf_method(cache=False, rename_output=False)
    def tensor(self):
        """Returns a tensor representing the value of the parameter.
        
        Returns:
            (tf.Tensor) The forward transform of the parameter applied to its
            free state.
            
        """
        self._ensure_variable()
        if '_Param__tensor' not in self._shared.cache:
            self._shared.cache['_Param__tensor'] =\
                    self.transform.tf_forward(self._variable)
        return self._shared.cache['_Param__tensor']
        #return self.transform.tf_forward(self._variable)

    @property
    def feed_dict(self):
        return {}

    @property
    @tf_method(cache=False, rename_output=False)
    def free_state(self):
        """Returns a variable that maps to the free state of the parameter."""
        if not self.fixed:
            self._ensure_variable()
            return self._variable
        else:
            raise FixedParameterError("cannot access free state of fixed Param")

    @property
    def fixed(self):
        return self._shared.fixed

    @fixed.setter
    def fixed(self, value):
        self._shared.fixed = value

    @property
    def transform(self):
        """The transform between the free space and value space."""
        return self._shared.transform

    @transform.setter
    def transform(self, value):
        """Sets the transform of the parameter.

        If the parameter is in a variable, clears the cache of anything
        higher in the tree that might rely on `self.tensor`.
        
        Examples:
            >>> from .transforms import Identity, transforms.Exp
            >>> class AttributeWrappedTF(WrappedTF, AttributeTree):
            ...     pass
            >>> w = AttributeWrappedTF()
            >>> w.p = Param(1.)
            >>> w.cache[0] = 123
            >>> w.p.transform = transforms.Exp()
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
            self.clear_all_ancestor_caches()
            old_value = self.value
            if '_Param__tensor' in self._shared.cache:
                del self._shared.cache['_Param__tensor']
            self._shared.transform = value
            self.value = old_value
        else:
            self._shared.transform = value

    @overrides
    def on_session_birth(self, session):
        if self._session:
            raise RuntimeError(
                "{} is already initialised in another session, {}."
                .format(self.long_name, self._session)
            )
        self._session = session
        self._ensure_variable()
        session.run(self.initializer)
        super().on_session_birth(session)

    @overrides
    def on_session_death(self, session):
        if self._session is not session:
            raise RuntimeError(
                "{} is not initialised in {}"
                .format(self.long_name, session)
            )
        self._session = None
        if self._variable:
            self._shared.numpy_value[...] = session.run(self.tensor)
        super().on_session_death(session)

    @overrides
    def clear_cache(self):
        """Save the variable value before it is cleared from the cache."""
        if self._variable:
            self._shared.numpy_value[...] = self.value
        self._shared.cache.clear()
        super().clear_cache()

    @property
    @tf_method(cache=False, rename_output=False)
    def initializer(self):
        """Initialises the internal `tf.Variable` to the correct value.
        
        This op is automatically run for sessions obtained using 
        `.get_session()`.
        
        """
        self._ensure_variable()
        with tf.control_dependencies([self._variable.initializer]):
            free_state = self.transform.np_backward(self._shared.numpy_value)
            return self._variable.assign(free_state)
            
    @tf_method(cache=False)
    def _ensure_variable(self):
        """Creates a variable if necessary."""
        if not self._variable:
            self._variable = tf.Variable(self._shared.numpy_value)
    
    @property
    def _variable(self):
        """Get the variable if there is one.
        
        Returns `None` (falsity) if there is no variable, or a `tf.Variable`
        (truth) if there is one. Hence variable existence may be checked using
        `if self._variable`.

        """
        return self._shared.cache.get('_Param__variable', None)

    @_variable.setter
    def _variable(self, value):
        """Sets the variable."""
        self._shared.cache['_Param__variable'] = value

class DataHolder(ProxyWrappedValue, Leaf):
    """Holds data to be fed into TensorFlow for computation.

    Attributes:
        value (np.ndarray): The value of the data.
        tensor (tf.placeholder): A placeholder for the data, 
            suitable for passing to TensorFlow ops.
        feed_dict (Dict[tf.placeholder, np.array_like]): A feed dictionary
            that feeds the value of the data into the placeholder op.
        on_shape_change ('raise' | 'pass' | 'recompile'): The action to take
            when the shape of the data changes; see the setter for `.value`.

    Examples:
        .. rubric:: Getting and setting values

        To get and set the value of the data, use the `.value` property:

        >>> a = np.array([1,2,3])
        >>> b = np.array([4,5,6])
        >>> d = DataHolder(a)
        >>> d.value
        array([1, 2, 3])
        >>> d.value = b
        >>> d.value
        array([4, 5, 6])

        See the docs for `gptf.core.params.WrappedValue` for more info.
        
        To access the value from TensorFlow, first build an op that relies
        on the `.tensor` attribute:

        >>> d = DataHolder(a)
        >>> op = tf.add(d.tensor, 1)

        Then evaluate the op in a session, passing in the feed dictionary
        to `tf.Session.run()`:

        >>> with d.get_session() as sess:
        ...     sess.run(op, feed_dict=d.feed_dict)
        array([2, 3, 4])

        .. rubric:: Copies

        You can create a copy of a `DataHolder` using `.copy()`. The
        new copy represents the same data, so the value and
        various items of state (see below) are the same.

        >>> d = DataHolder([1., 1., 1.])
        >>> copy = d.copy()

        Copies have the same value as the original, and updating the
        value of the copy updates the value of the original.

        >>> np.array_equal(d.value, copy.value)
        True
        >>> d.value = [5., 4., 3.]
        >>> copy.value
        array([ 5., 4., 3.])

        Copies have the same placeholder.

        >>> d.tensor is copy.tensor
        True

        This means that it, just in the case of `Param`s, it is 
        *very important* that every copy of a `DataHolder` has the
        same device context and graph, or odd things happen.

        >>> class AttributeWrappedTF(AttributeTree, WrappedTF):
        ...     pass
        >>> treedata = DataHolder(1.0)
        >>> w = AttributeWrappedTF()
        >>> w.param = treedata
        >>> w.child = AttributeWrappedTF()
        >>> w.child.param = treedata  # creates a copy of the data
        >>> w.tf_graph = tf.Graph()
        >>> w.tf_device = '/job:spoon/task:0'
        >>> w.child.tf_device = '/job:knife/task:0'
        
        In circumstances like the above, where copies have
        conflicting parental device contexts, set the device
        context explicitly.

        >>> treedata.tf_device = '/job:spoon'
        
        To create a new dataholder with the same value, pass the value
        to `DataHolder.__init__()`.

        >>> notacopy = DataHolder(d.value)
        
    """

    def __init__(self, initial_value, **kwargs):
        """Initialiser.

        Args:
            initial_value (np.array_like): The initial value of the data.
            **kwargs: passed through to the constructor of WrappedValue.

        """
        super().__init__(**kwargs)
        self._shared.numpy_value = np.array(initial_value)
        self._placeholder = None

    @overrides
    def _get_value(self):
        return self._shared.numpy_value.copy()

    @overrides
    def _set_value(self, value):
        self._shared.numpy_value[...] = value

    @property
    def tensor(self):
        self._assert_placeholder()
        return self._placeholder

    @property
    def feed_dict(self):
        self._assert_placeholder()
        return { self._placeholder: self._shared.numpy_value }

    @property
    def _placeholder(self):
        return self._shared.cache.get('_DataHolder__placeholder', None)

    @_placeholder.setter
    def _placeholder(self, value):
        self._shared.cache['_DataHolder__placeholder'] = value

    @tf_method(cache=False)
    def _assert_placeholder(self):
        if self._placeholder is None:
            dtype = tf.as_dtype(self._shared.numpy_value.dtype)
            self._placeholder = tf.placeholder(dtype) #self._shared.numpy_value.shape)

class Parameterized(WrappedTF):
    """An object that contains parameters and data.

    This object is designed to be part of a tree, with `Param`\ s and 
    `DataHolder`\ s at the leaves.

    Attributes:
        fixed (bool): A flag indicating whether or not any child `Param`\ s 
            should be fixed. Setting this attribute also sets the `.fixed` 
            attribute of anything lower in the tree.
        feed_dict (Dict[tf.placeholder, np.array_like]): A feed dictionary
            that feeds the values of DataHolders into their placeholder ops.
        params (List[Param]): A list of all the `Param`\ s lower in the tree,
            sorted by their long name.
        data_holders (List[Param]): A list of all the `DataHolder`\ s lower in 
            the tree, sorted by their long name.

    Examples:
        >>> from gptf import transforms
        >>> class Parameterized(Parameterized, AttributeTree):
        ...     pass
        >>> m = Parameterized()
        >>> m.param = Param(1.)
        >>> m.child = Parameterized()
        >>> m.child.a = Param([2., 3., 4.], transform=transforms.Exp())
        >>> m.child.b = Param([[1.0]])
        >>> m.X = DataHolder([1., 2., 3., 4., 5.])
        >>> m.Y = DataHolder([10., 23.3, 3., 42., .1])
        >>> m.child.data = DataHolder(23)

        The `.params` and `.data` attributes return all instances of their
        associated types lower in the tree.

        >>> set(m.params) == {m.param, m.child.a, m.child.b}
        True
        >>> set(m.data_holders) == {m.X, m.Y, m.child.data}
        True

    """
    ARRAY_DISPLAY_LENGTH = 5

    def __init__(self):
        super().__init__()
        self._fixed = False

    @property
    def fixed(self):
        return self._fixed

    @fixed.setter
    def fixed(self, value):
        iterator = iter(self)
        next(iterator)  # skip self to prevent recursion bugs
        for node in filter(isattrof('fixed'), iterator):
            node.fixed = value

    @property
    def feed_dict(self):
        """The union of the `.feed_dict`\ s of objects lower in the tree."""
        result = {}
        iterator = iter(self)
        next(iterator)  # skip self to prevent recursion bugs
        for node in filter(isattrof('feed_dict'), iterator):
            result.update(node.feed_dict)
        return result

    @property
    def params(self):
        """A sorted list of the `Param`\ s lower in the tree."""
        l = list(filter(isclassof(Param), self))
        l.sort(key=lambda x: x.long_name)
        return l

    @property
    def data_holders(self):
        """A sorted list of the `DataHolder`\ s lower in the tree."""
        l = list(filter(isclassof(DataHolder), self))
        l.sort(key=lambda x: x.long_name)
        return l

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
        heading = "Parameterized object {}".format(self.long_name)
        params = self.param_summary(array_len, fmt)
        data = self.data_summary(array_len, fmt)

        if fmt == "fancy":
            headinglength = len(heading)
            if os.name != "nt":
                heading = "\033[1m" + heading + "\033[0m"
            if not (params or data):
                return heading
            headingtable = "┍━" + "━" * headinglength + "━┑\n"
            headingtable += "│ " + heading + " │\n"
            headingtable += "┕━" + "━" * headinglength + "━┙"
            tables = [headingtable]
            if params:
                tables.append("┌─────────┐" + os.linesep\
                            + "│ Params: │" + os.linesep\
                            + "└─────────┘")
                tables.append(params)
            if data:
                tables.append("┌───────┐" + os.linesep\
                            + "│ Data: │" + os.linesep\
                            + "└───────┘")
                tables.append(data)
            return combine_fancy_tables(*tables)
        elif fmt == "plain":
            lines = [heading]
            if params:
                lines.extend(["", "Params:"])
                lines.append(prefix_lines("    ", params))
            if data:
                lines.extend(["", "Data:"])
                lines.append(prefix_lines("    ", data))
            if params or data: lines.append("")
            return os.linesep.join(lines)
        elif fmt == "html":
            lines = ["<table id='parameterized' width=100%>",
                     "  <tr><th>{}</th></tr>".format(heading)]
            if params:
                lines.append("  <tr><td><b>Params</b></td></tr>")
                lines.append("  <tr><td>{}</td></tr>".format(params))
            if data:
                lines.append("  <tr><td><b>Data</b></td></tr>")
                lines.append("  <tr><td>{}</td></tr>".format(data))
            lines.append("</table>")
            return os.linesep.join(lines)

    def param_summary(self, array_len=ARRAY_DISPLAY_LENGTH, fmt="fancy"):
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

        Examples:
            >>> from gptf import transforms
            >>> class Parameterized(Parameterized, AttributeTree):
            ...     pass
            >>> p = Parameterized()
            >>> p.fallback_name = 'p'
            >>> p.child = Parameterized()
            >>> p.param = Param(1.)
            >>> p.child.a = Param([1., 2., 3.])
            >>> p.child.b = Param([[1.]], transform=transforms.Exp())
            >>> print(p.param_summary(fmt='plain'))  # doctest:-NORMALIZE_WHITESPACE
            name      | value                 | transform | prior
            ----------+-----------------------+-----------+------
            p.child.a | [1.000, 2.000, 3.000] | identity  | nyi  
            p.child.b | <np.ndarray>          | +ve (Exp) | nyi  
            p.param   | 1.000                 | identity  | nyi  
            
        """
        params = self.params
        if not params:
            return ""

        names = tuple(self._name_str(p.long_name, fmt) for p in params)
        values = tuple(self._value_str(p.value, array_len, fmt) for p in params)
        transforms = tuple(str(p.transform) for p in params)
        priors = tuple("nyi" for p in params)

        columns = (names, values, transforms, priors)
        headers = ("name", "value", "transform", "prior")

        return construct_table(columns, headers, fmt=fmt)

    def data_summary(self, array_len=ARRAY_DISPLAY_LENGTH, fmt="fancy"):
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

        Examples:
            >>> class Parameterized(Parameterized, AttributeTree):
            ...     pass
            >>> p = Parameterized()
            >>> p.fallback_name = 'p'
            >>> p.child = Parameterized()
            >>> p.data = DataHolder(1.)
            >>> p.child.a = DataHolder([1., 2., 3.])
            >>> p.child.b = DataHolder([[1.]])
            >>> print(p.data_summary(fmt='plain'))  # doctest:-NORMALIZE_WHITESPACE
            name      | value                
            ----------+----------------------
            p.child.a | [1.000, 2.000, 3.000]
            p.child.b | <np.ndarray>         
            p.data    | 1.000                

        """
        data = self.data_holders
        if not data:
            return ""

        names = tuple(self._name_str(d.long_name, fmt) for d in data)
        values = tuple(self._value_str(d.value, array_len, fmt) for d in data)

        columns = (names, values)
        headers = ("name", "value")

        return construct_table(columns, headers, fmt=fmt)

    @staticmethod
    def _name_str(name, fmt):
        """Maybe add ANSI escape codes to prettify a name.
        
        Args:
            name (str): The name to prettify.
            fmt ('fancy' | 'plain' | 'html'): The style to format with.
                If 'fancy' and not Windows, emboldens the last part of the
                name.
                
        """
        if fmt == "fancy" and os.name != "nt":
            parts = name.split(".")
            parts[-1] = "\033[1m" + parts[-1] + "\033[0m"
            return ".".join(parts)
        elif fmt == "html":
            parts = name.split(".")
            parts[-1] = "<b>" + parts[-1] + "</b>"
            return ".".join(parts)
        else:
            return name

    @staticmethod
    def _value_str(value, max_len, fmt):
        """Constructs a string representation of a numpy value.

        Args:
            value (np.ndarray): The array to represent.
            max_len (np.ndarray): The maximum length a 1d array can be before
                it is truncated.
            fmt ('fancy' | 'plain' | 'html'): The style to format with.
                Currently ignored.
        
        Examples:
            >>> a = np.array(1.)
            >>> b = np.arange(5)
            >>> c = np.arange(6)
            >>> d = np.array([[1.]])

            A zero-dimensional array produces a scalar representation:

            >>> Parameterized._value_str(a, 5, '')
            '1.000'

            One dimensional arrays are repreduced up to `max_len`:

            >>> Parameterized._value_str(b, 5, '')
            '[0, 1, 2, 3, 4]'
            >>> Parameterized._value_str(c, 5, '')
            '[0, 1, 2, 3, ...]'

            Multidemensional arrays appears as a placeholder value.

            >>> Parameterized._value_str(d, 5, '')
            '<np.ndarray>'
          
        """
        def tostring(arr):
            arr = np.asscalar(arr)
            if isinstance(arr, float):
                return "{:.3f}".format(arr)
            else:
                return str(arr)
        if len(value.shape) > 1:
            return "<np.ndarray>"
        elif len(value.shape) == 1:
            if value.shape[0] > max_len:
                return "[" + ", ".join(map(tostring, value[:max_len - 1])) \
                        + ", ...]"
            else:
                return "[" + ", ".join(map(tostring, value)) + "]"
        else:
            return tostring(value)

# en-gb compatibility patch
Parameterised = Parameterized

class ParamAttributes(Parameterized, AttributeTree):
    """Parameters are accessed using attributes."""
    @overrides
    def __setattr__(self, name, value):
        """Set the value of `Param`s and `DataHolder`s on assignment.
        
        Args:
            name (str): The name of the attribute.
            value: The value to set.
        
        Examples:
            Assigning a numerical, numpy or string value to a `Param` or 
            `DataHolder` child assigns to that child's value instead.

            >>> p = ParamAttributes()
            >>> p.param = Param(1.0)
            >>> p.param = 2.0
            >>> isinstance(p.param, Param)
            True
            >>> p.param.value
            array(2.0)
            >>> p.data = DataHolder(1.0)
            >>> p.data = 2.0
            >>> isinstance(p.data, DataHolder)
            True
            >>> p.data.value
            array(2.0)

            Assigning anything else overwrites the attribute:

            >>> class Example():
            ...     pass
            >>> p.param = Example()
            >>> isinstance(p.param, Example)
            True
            >>> p.data = Example()
            >>> isinstance(p.data, Example)
            True

            Children still know who their parents are.

            >>> p.param = Param(1.0)
            >>> p.param.parent is p
            True

        """
        if hasattr(self, name):
            curr = getattr(self, name) 
            if ((isinstance(curr, Param) or isinstance(curr, DataHolder)) and 
                is_array_like(value)):
                # okay to assign to curr.value
                curr.value = value
                return
        super().__setattr__(name, value)

class ParamList(Parameterized, ListTree):
    """A list of `Param` or `DataHolder` objects.

    Examples:
        You can set the value of children by assigning to their index:

        >>> p = ParamList()
        >>> p.append(Param(1.))
        >>> p.append(DataHolder(1.))
        >>> p[0] = 3.
        >>> isinstance(p[0], Param)
        True
        >>> p[0].value
        array(3.0)
        >>> p[1] = 5.
        >>> isinstance(p[1], DataHolder)
        True
        >>> p[1].value
        array(5.0)

        You can still overwrite parameters etc with new ones:

        >>> p[0] = ParamAttributes()
        >>> isinstance(p[0], Param)
        False

    """
    def __init__(self, initial_values=()):
        """Initialiser.

        Args:
            initial_values (Sequence[Tree], optional): The initial 
                value for the children of this paramlist.
        
        """
        super().__init__()
        super(ListTree, self).extend(initial_values)

    @overrides
    def __setitem__(self, key, value):
        """If `self[key]` is `Param` or `DataHolder` and value is 
        `np.array_like`, set `self[key].value` instead."""
        curr = self[key]
        if ((isinstance(curr, Param) or isinstance(curr, DataHolder)) and 
            is_array_like(value)):
            # okay to assign to curr.value
            curr.value = value
        else:
            super().__setitem__(key, value)

def autoflow(*placeholder_specs):
    """Wraps up a TensorFlow method so that it takes NumPy and gives NumPy.

    When an autoflowed method is called, we construct placeholders to 
    represent the passed arguments, apply `tf_method` 
    to the wrapped method and evaluate it on the placeholders to produce 
    a TensorFlow op. We then evaluate the op in a session, passing in the 
    appropriate feed dictionaries, and return the resulting NumPy array.

    We also cache the op, so that multiple calls to the function construct
    the op only once. This cache is cleared when device contexts change,
    when transforms on `Param`\ s change, and any number of other similar
    circumstances.

    Args:
        *placeholder_specs: some tuples that specify how the placeholders 
            for the arguments of the decorated method should be 
            constructed. Each tuple will be used as the arguments to a 
            call to `tf.placeholder()`. The first tuple will be used to
            construct the placeholder for the first argument of the
            decorated function, the second for the second, and so on.

    Returns:
        A decorator that autoflows the decorated method.

    Examples:
        The decorator syntax looks like this:

        >>> class MyClass(Parameterized, AttributeTree):
        ...     @autoflow((tf.float64,), (tf.float64,))
        ...     def tf_add(self, a, b):
        ...         return tf.add(a, b)
        ...
        ...     @autoflow((tf.float64, [3, None]))
        ...     def tf_reduce_sum(self, a):
        ...         return tf.reduce_sum(a, 1)

        Now we can leverage the mighty power of tensorflow without ever
        having to get our hands dirty:

        >>> m = MyClass()
        >>> m.tf_add(5, 9)
        14.0
        >>> a = np.array([1., 2., 3.])
        >>> b = np.array([4., 5., 6.])
        >>> m.tf_add(a, b)
        array([ 5., 7., 9.])

        `MyClass.tf_reduce_sum` will only allow arguments that match the
        shape `[1, None]`; that is, rank 2 tensors whose first dimension
        is `5`.

        >>> # shape is (3, 2), compatible with [3, None]
        >>> m.tf_reduce_sum([[1., 2.], [2., 3.], [3., 4.]])
        array([ 3., 5., 7.])
        >>> # shape is (2, 1), not compatible with [3, None]
        >>> m.tf_reduce_sum([[1.], [2.]])
        Traceback (most recent call last):
            ...
        ValueError: Cannot feed value of shape (2, 1)...

        Autoflowed methods still work if the device context changes:

        >>> # set up a distributed execution environment
        >>> clusterdict = \\
        ...     { 'worker': ['localhost:2224']
        ...     , 'master': ['localhost:2225']
        ...     }
        >>> spec = tf.train.ClusterSpec(clusterdict)
        >>> worker = tf.train.Server(spec, job_name='worker', task_index=0)
        >>> worker.start()
        >>> master = tf.train.Server(spec, job_name='master', task_index=0)
        >>> # change m's device context
        >>> # we're about to do weird things with op placement, and we
        >>> # don't want it in the default graph where it can mess with
        >>> # other doctests, so change m's tf_graph as well.
        >>> m.tf_graph = tf.Graph()
        >>> m.tf_device = '/job:worker/task:0'
        >>> m.tf_session_target = master.target
        >>> # autoflow
        >>> m.tf_add(3, 2)
        5.0
        >>> m.tf_reduce_sum([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        array([  6., 15., 24.])

    """
    def decorator(method):
        """A decorator that autoflows a method.
        
        Args:
            method (Callable[[Parameterized, ...], tf.Tensor]): A method 
                of a `Parameterized` that returns a TensorFlow op.

        Returns:
            (Callable[[Parameterized, ...], np.ndarray]): A method that
            returns a NumPy array through autoflow magic.

        """
        @tf_method(cache=False)
        @wraps(method)
        def wrapper(instance, *np_args):
            name = '_Parameterized__autoflow__{}'.format(method.__name__)
            if name in instance.cache:
                storage = instance.cache[name]
            else:
                tf_args = [tf.placeholder(*s) for s in placeholder_specs]
                storage =\
                        { 'op': method(instance, *tf_args)
                        , 'tf_args': tf_args
                        }
                instance.cache[name] = storage
            feed_dict = dict(zip(storage['tf_args'], np_args))
            feed_dict.update(instance.feed_dict)
            
            with instance.get_session() as sess:
                return sess.run(storage['op'], feed_dict=feed_dict)
        return wrapper
    return decorator

