# standard library
from builtins import super, object
from functools import wraps
try:  # in case of rogue Python 2.7, use collections instead of collections.abc
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
try:  # in case of rogue Python 2.7, use contextlib2 instead of contextlib
    from contextlib import contextmanager, ExitStack
except ImportError:
    from contextlib2 import contextmanager, ExitStack
import re

# nonstandard library
import tensorflow as tf
from overrides import overrides

# local
from .trees import TreeWithCache, cache_method


INVALID_NAME_SCOPE_CHAR = re.compile(r"[^\w.\\\-/]")

class WrappedTFSession(tf.Session):
    @wraps(tf.Session.__init__)
    def __init__(self, wrappedtf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wrappedtf = wrappedtf
        for node in self.wrappedtf:
            node.on_session_birth(self)
        self.dead = False

    @wraps(tf.Session.__enter__)
    def __enter__(self):
        return super().__enter__()

    @wraps(tf.Session.__exit__)
    def __exit__(self, type_, value, traceback):
        self._maybe_death()
        return super().__exit__(type_, value, traceback)

    @wraps(tf.Session.close)
    def close(self):
        try:  # we might be being called from __del__
            self._maybe_death()
        finally:
            return super().close()

    def _maybe_death(self):
        if not self.dead:
            for node in self.wrappedtf:
                node.on_session_death(self)
            self.dead = True

class NullContextWrapper(object):
    """Wraps an object so that its context does nothing.

    Attributes:
        _NullContextWrapper__wrapped: the wrapped context manager.
    
    Examples:
        >>> class Example(object):
        ...     def __enter__(self):
        ...         print("enter called")
        ...         return self
        ...     def __exit__(self, *_):
        ...         print("exit called")

        Wrapping a context manager causes it to do nothing.

        >>> e = Example()
        >>> with e:
        ...     print("with body")
        enter called
        with body
        exit called
        >>> with NullContextWrapper(e) as e_:
        ...     assert e is e_
        ...     print("with body")
        with body

        You can access other attributes of the context manager
        through the wrapper.

        >>> e.attribute = "horse"
        >>> NullContextWrapper(e).attribute
        'horse'
        
    """
    def __init__(self, wrapped):
        self.__wrapped = wrapped

    def __enter__(self):
        return self.__wrapped

    def __exit__(self, type_, value, traceback):
        pass

    # delegate everything but enter/exit to self.__wrapped
    def __getattribute__(self, name):
        if name not in {'__enter__', '__exit__', 
                '_NullContextWrapper__wrapped'}:
            return getattr(self.__wrapped, name)
        else:
            return super(NullContextWrapper, self).__getattribute__(name)

    def __setattr__(self, name, value):
        if name not in {'__enter__', '__exit__',
                '_NullContextWrapper__wrapped'}:
            setattr(self.__wrapped, name, value)
        else:
            #super().__setattr__(name, value)
            super(NullContextWrapper, self).__setattr__(name, value)

    def __delattr__(self, name):
        if name not in {'__enter__', '__exit__',
                '_NullContextWrapper__wrapped'}:
            delattr(self.__wrapped, name, value)
        else:
            super(NullContextWrapper, self).__delattr__(name, value)

class WrappedTF(TreeWithCache):
    """Provides facilities for keeping TensorFlow behind the scenes.

    WARNING: `WrappedTF` assumes that its parent, and indeed all things
    higher than it in the tree, are also `WrappedTF`. Make sure that the
    root of the tree has implemented `.get_session()`, and that
    the direct parent has implemented `.op_placement_context()`.

    Attributes:
        NO_DEVICE (object): A class-level constant, used to specify an
            empty op placement context.
        tf_device (str | Callable[[tf.Operation], str] | tf.DeviceSpec | WrappedTF.NO_DEVICE | None):
            The device context onto which this object's ops should be pinned.
            Device contexts are applied hierarchically, starting from the
            highest parent. See `.op_placement_context()`.

            This will be passed as the sole argument to `tf.device()`. 
            `WrappedTF.NO_DEVICE` indicates that `None` will be
            passed to `tf.device`, whereas `None` indicates that `tf.device()`
            will not be called. Otherwise, see the documentation for
            `tf.device()`.

            Defaults to `None`.
        tf_graph (tf.Graph | None): The graph to place ops in. If `None`,
            the graph an op is placed in is the lowest defined `.tf_graph` 
            above this one in the tree. If all objects in the `Tree` have
            `.tf_graph` set to `None`, the default graph is used.
            Defaults to `None`.
        tf_session_target (str | dict | None): The target under which 
            sessions should run. If this is `None`, no arguments will be 
            passed to `tf.session()`. If this is a dictionary, then its 
            contents will be used as keyword arguments for `tf.session()`. 
            Else, this will be the sole argument for
            `tf.session()`. See `.get_session()`.

    """
    _NO_DEVICE = object()

    def __init__(self):
        super().__init__()
        self._tf_graph = None
        self._tf_device = None
        self._tf_session_target = None

    # This is an attempt to guard against assignment to _NO_DEVICE
    @property
    def NO_DEVICE(self):
        """When `.tf_device` is set to `.NO_DEVICE`, `None` will be passed
        to `tf.device()`"""
        return self._NO_DEVICE

    @property
    def tf_device(self):
        """The device context onto which this object's ops should be pinned."""
        return self._tf_device

    @tf_device.setter
    def tf_device(self, value):
        self._on_op_placement_context_change()
        self._tf_device = value

    @property
    def tf_graph(self):
        """The graph to place ops in."""
        return self._tf_graph

    @tf_graph.setter
    def tf_graph(self, value):
        self._on_op_placement_context_change()
        self._tf_graph = value

    @property
    def tf_session_target(self):
        """The target under which sessions should be run."""
        return self._tf_session_target

    @tf_session_target.setter
    def tf_session_target(self, value):
        self._tf_session_target = value

    @contextmanager
    def op_placement_context(self, name_scope=True):
        """Applies op placement rules based on the object hierarchy.

        Examples:
            >>> from gptf.core.trees import AttributeTree
            >>> class Example(WrappedTF, AttributeTree):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.tf_graph = tf.Graph()

            Choose the op placement context by assigning to `.tf_device`:

            >>> a, b, c, d, e = [Example() for _ in range(5)]
            >>> a.tf_device = '/job:worker'
            >>> b.tf_device = tf.DeviceSpec(device_type='GPU', device_index=0)
            >>> c.tf_device = None
            >>> d.tf_device = d.NO_DEVICE
            >>> e.tf_device = '/job:spoon'

            Device contexts are combined, starting from the context of the
            root of the tree. `c.tf_device` is `None`, so it uses the context
            of its parent.

            >>> a.child = c
            >>> with a.op_placement_context():
            ...     print(tf.constant(0).device)
            /job:worker
            >>> with c.op_placement_context():
            ...     print(tf.constant(0).device)
            /job:worker

            `d.tf_device` is `WrappedTF.NO_DEVICE`, so it resets the device
            context.

            >>> a.child = d
            >>> with d.op_placement_context():
            ...     print(tf.constant(0).device)
            <BLANKLINE>

            Other device contexts combine the way you would expect them to.

            >>> a.child = b
            >>> b.child = e
            >>> with b.op_placement_context():
            ...     # get job from a
            ...     print(tf.constant(0).device)
            /job:worker/device:GPU:0
            >>> with e.op_placement_context():
            ...     # get device from b, overwrite job from a
            ...     print(tf.constant(0).device)
            /job:spoon/device:GPU:0

            The root node of the tree may define a `.tf_graph`. Child ops will
            be placed in the `.tf_graph` of their highest parent.

            >>> a.tf_graph = tf.Graph()
            >>> b.tf_graph = tf.Graph()
            >>> with a.op_placement_context():
            ...     tf.constant(0).graph is a.tf_graph
            True
            >>> with b.op_placement_context():
            ...     tf.constant(0).graph is a.tf_graph
            True
            >>> with e.op_placement_context():
            ...     tf.constant(0).graph is a.tf_graph
            True

            In addition, a name scope is opened that matches the object
            hierachy:

            >>> with a.op_placement_context():
            ...     print(tf.constant(0).name)
            unnamed/Const...
            >>> with b.op_placement_context():
            ...     print(tf.constant(0).name)
            unnamed.child/Const...
            >>> with e.op_placement_context():
            ...     print(tf.constant(0).name)
            unnamed.child.child/Const...

        """
        with ExitStack() as stack:
            if self.parent is None and self.tf_graph is not None:
                stack.enter_context(self.tf_graph.as_default())
            elif self.parent is not None:
                parent_context = self.parent.op_placement_context(name_scope)
                stack.enter_context(parent_context)

            if self.tf_device is not None:
                dev = self.tf_device
                if dev is self.NO_DEVICE:
                    dev = None
                stack.enter_context(tf.device(dev))

            if name_scope:
                # enter "absolute" name scope by appending "/"
                scope = self.long_name + "/"
                scope = INVALID_NAME_SCOPE_CHAR.sub("", scope)
                stack.enter_context(tf.name_scope(scope))

            yield

    def get_session(self):
        """Gets a TensorFlow session in which ops can be run.
        
        Returns the default session if there is one. Else, returns a 
        new session using the session target of the highest parent.

        Returns:
            (tf.Session): The default session if there is one, else a 
            session matching the session target of the highest parent.

        Examples:
            >>> from gptf.core.trees import AttributeTree
            >>> class AttributeWrappedTF(WrappedTF, AttributeTree):
            ...     pass

            >>> w = AttributeWrappedTF()

            If there is already a default session, returns that 
            session in a NullContextWrapper.

            >>> with tf.Session() as sess:
            ...     sess0 = w.get_session()
            ...     print(type(sess0).__name__)
            ...     sess0._NullContextWrapper__wrapped is sess
            NullContextWrapper
            True

            It's safe to use w.get_session() in a with block, even if
            there is a default session. Doing so won't call the
            `__enter__` or `__exit__` methods of the default session.

            >>> class AnnounceSession(tf.Session):
            ...     def __enter__(self):
            ...         print('__enter__ called')
            ...         return super().__enter__()
            ...     def __exit__(self, *args):
            ...         print('__exit__ called')
            ...         return super().__exit__(*args)
            >>> with AnnounceSession() as sess:
            ...     print('before nested with')
            ...     with w.get_session() as sess0:
            ...         assert sess is sess0
            ...         print('nested with')
            ...     print('after nested with')
            __enter__ called
            before nested with
            nested with
            after nested with
            __exit__ called
            
            Else, returns a new session each time:

            >>> sess2 = w.get_session()
            >>> sess2 is sess
            False
            >>> sess3 = w.get_session()
            >>> sess2 is sess3
            False
            >>> sess2.close()
            >>> sess3.close()
            
            >>> class Example(WrappedTF, AttributeTree):
            ...     def op(self):
            ...         with self.op_placement_context():
            ...             return tf.constant(1)
            ...
            ...     def depth(self):
            ...         tot = 0
            ...         if self.parent is not None:
            ...             tot += self.parent.depth()
            ...         with self.get_session() as sess:
            ...             tot += sess.run(self.op())
            ...         return tot
            >>> a = Example()
            >>> # we're about to do weird things with op placement, and we
            >>> # don't want it in the default graph where it can mess with
            >>> # other doctests.
            >>> a.tf_graph = tf.Graph()
            >>> a.child = Example()

            `Example` is a simple class that provides a method, `.depth()`,
            that uses TensorFlow to calculate an object's depth in the tree.

            >>> a.depth()
            1
            >>> a.child.depth()
            2

            `Example.op()` places its op based on the hierachical device 
            context. If we change `a`'s device context, we also change
            `a.child`'s.

            >>> print(a.child.op().device)
            
            >>> a.tf_device = '/job:worker/task:0'
            >>> print(a.child.op().device)
            /job:worker/task:0

            `a.child.depth()` will now result in an error:

            >>> a.child.depth()
            Traceback (most recent call last):
                ...
            tensorflow.python.framework.errors.InvalidArgumentError: ...

            `a.child.op()` is now being placed as if it were in a distributed 
            context, and the default session knows nothing about jobs or tasks.
            However, if we set `a.session_target` appropriately, 
            `a.child.get_session()` will return a session capable of
            running ops created with `a.child.op_placement_context`.

            >>> clusterdict = \\
            ...     { 'worker': ['localhost:2222']
            ...     , 'master': ['localhost:2223']
            ...     }
            >>> spec = tf.train.ClusterSpec(clusterdict)
            >>> worker = tf.train.Server(spec, job_name='worker', task_index=0)
            >>> worker.start()
            >>> master = tf.train.Server(spec, job_name='master', task_index=0)
            >>> a.tf_session_target = master.target

            `a.child.depth()` should now run smoothly.

            >>> a.child.depth()
            2

            In general, this means that as long as the session target of
            the root is set correctly, anything lower in the tree that uses
            `self.get_session()` should work without fuss.
        
        """
        def new_session():
            if self.tf_session_target is None:
                kwargs = {'graph': self.tf_graph}
            elif isinstance(self.tf_session_target, dict):
                kwargs = self.tf_session_target
            else:
                kwargs =\
                        { 'target': self.tf_session_target
                        , 'graph': self.tf_graph
                        }
            return WrappedTFSession(self, **kwargs)

        default = tf.get_default_session()
        if default is not None:
            return NullContextWrapper(default)
        else:
            if self.parent is None:
                return new_session()
            else:
                return self.highest_parent.get_session()

    def on_session_birth(self, session):
        """Called just after a session is created.
        
        Args:
            session (tf.Session): The created session.
          
        """
        pass

    def on_session_death(self, session):
        """Called just before a session is closed.
        
        Args:
            session (tf.Session): The dying session.
          
        """
        pass


    def _on_op_placement_context_change(self):
        """Called when the op placement context changes.
        
        When the op placement context changes, its cached TensorFlow ops are 
        no longer valid. This means that all cached ops created with the 
        `WrappedTF`'s device context and any ops that use ops created with the 
        `WrappedTF`'s device context are invalid.

        To deal with this, we clear the cache of any `WrappedTF` that is a
        direct ancestor, and any `WrappedTF` in the subtree.

        """
        self.clear_ancestor_caches()
        self.clear_subtree_caches()

    @overrides
    def _set_parent(self, new_parent):
        """Deals with cache clearing that happens when tree anatomy changes.
        
        If a `WrappedTF`'s ancestry changes, its op placement context
        will change, so we need to clear the appropriate caches. Note that
        since this involves clearing the caches of everything lower in the
        subtree, it is enough to clear the cache only when a node's direct
        parent changes.

        If a `WrappedTF` stops being the highest parent, i.e. it becomes the
        child of another `WrappedTF`, it should kill its session. If the new
        highest parent has a session, we should call `on_session_birth`
        across the subtree of the new child once it has been added to the
        tree.

        Examples:
            >>> from gptf.core.trees import AttributeTree
            >>> class Example(WrappedTF, AttributeTree):
            ...     def __init__(self, name):
            ...         super().__init__()
            ...         self.fallback_name = name
            >>> w = Example('w')
            >>> x = Example('x')
            >>> y = Example('y')
            >>> z = Example('z')
            >>> w.child_0 = x
            >>> y.child = z
            >>> def setup():
            ...     objs = [w, x, y, z]
            ...     # fill caches
            ...     for o in objs: o.cache['tf'] = 'compiled tf function'

            When `y` gets a parent, it should close its session and 
            clear its subtree's cache. `w`, which knows nothing about `y` so
            should not have any ops that depend on `y`'s device context,
            should not have its cache cleared; the same goes for `x`.

            >>> setup()
            >>> w.child_1 = y
            >>> 'tf' in w.cache and 'tf' in w.child_0.cache
            True
            >>> 'tf' in w.child_1.cache or 'tf' in w.child_1.child.cache
            False
            
            Deleting `w.child_1` means that `y` loses a parent. `w` knows
            about `y` and might have ops that depend on `y`'s device context,
            so we should clear its cache. `w.child_0` still knows nothing
            about `y`, so its cache should still not be cleared.

            >>> setup()
            >>> del w.child_1
            >>> 'tf' in w.child_0.cache
            True
            >>> 'tf' in w.cache 
            False
            >>> 'tf' in y.cache or 'tf' in y.child.cache
            False

            When a `WrappedTF` is copied between trees, the caches in the
            copy are cleared and session births are registered in the
            subtree of the copy.

            >>> setup()
            >>> y.child_1 = w.child_0
            >>> 'tf' in y.cache and 'tf' in y.child.cache
            True
            >>> 'tf' in w.cache and 'tf' in w.child_0.cache
            True
            >>> 'tf' in y.child_1.cache
            False

        """
        self._on_op_placement_context_change()
        super()._set_parent(new_parent)  # move to new tree

def tf_method(name_scope=True, rename_output=True, 
        cache=True, cache_limit=128):
    """Decorator version of `WrappedTF.op_placement_context`.
    
    Applies `instance.op_placement_context(name_scope=False)` 
    to `instance.method(...)`
    
    Args:
        name_scope (bool): If `True`, wraps the function in an
            appropriate `tf.name_scope()`.
        cache (bool): If `True`, applies caching to the function.
            Multiple calls with the same arguments will return
            the same result. See examples.
        cache_limit (int): The limit for the cache. 

    Examples:
        In the following example, `Example.method_a` is equivalent 
        to `Example.method_b`.

        >>> from gptf.core.trees import AttributeTree
        >>> class Example(WrappedTF, AttributeTree):
        ...     def method_a(self, a, b):
        ...         with self.op_placement_context(name_scope=False):
        ...             scope = self.long_name + '.method_a/'
        ...             with tf.name_scope(scope):
        ...                 result = tf.add(a, b)
        ...                 return tf.identity(result, name='0')  #scope[:-1])
        ...     @tf_method()
        ...     def method_b(self, a, b):
        ...         return tf.add(a, b)

        Devices are set properly in both methods:

        >>> e = Example()
        >>> e.tf_graph = tf.Graph()  # don't break other doctests
        >>> e.tf_device = '/job:worker/task:0'
        >>> a = e.method_a(2, 3)
        >>> print(a.device)
        /job:worker/task:0
        >>> b = e.method_b(2, 3)
        >>> b.device == a.device
        True

        The returned tensor(s) are given the name of the name scope.

        >>> print(a.name)
        unnamed.method_a/0:0
        >>> print(b.name)
        unnamed.method_b/0:0

        Multiple method calls produce unique names.

        >>> print(e.method_b(1, 2).name)
        unnamed.method_b/0_1:0

        If a method returns a sequence of tensors, they are named
        `<scope>/0`, `<scope>/1`, etc.

        >>> class DoubleReturnExample(WrappedTF, AttributeTree):
        ...     @tf_method(cache=False)
        ...     def method(self):
        ...         return tf.constant(1), tf.constant(2)
        >>> obj = DoubleReturnExample()
        >>> c, d = obj.method()
        >>> print(c.name)
        unnamed.method/0:0
        >>> print(d.name)
        unnamed.method/1:0
        >>> c, d = obj.method()
        >>> print(c.name)
        unnamed.method/0_1:0
        >>> print(d.name)
        unnamed.method/1_1:0

        Calls to other tensorflow methods do not cause nested name scopes.

        >>> class NestedExample(WrappedTF, AttributeTree):
        ...     def __init__(self, child):
        ...         super().__init__()
        ...         self.child = child
        ...     @tf_method()
        ...     def method(self, a, b):
        ...         print(self.child.method_b(a, b).name)
        >>> NestedExample(Example()).method(0, 0)
        unnamed.child.method_b/0:0

        Else, no attempt is made to rename the output.

        >>> class NumpyReturnExample(WrappedTF, AttributeTree):
        ...     @tf_method()
        ...     def method(self):
        ...         return self.get_session().run(tf.constant(1))
        >>> NumpyReturnExample().method()
        1

        If caching is enabled, then multiple calls with the same 
        arguments will result in the same return value.

        >>> tensor = e.method_b(5, 5)
        >>> tensor is e.method_b(5, 5)
        True
        
        This means that ops are not added to the graph multiple times
        for identical method calls, which is a good thing.

        If the cache is cleared (perhaps due to a device context
        change), the method cache is also cleared, and new tensors
        will be returned.

        >>> e.clear_cache()
        >>> tensor is e.method_b(5, 5)
        False
        
    """
    def decorator(method):
        @wraps(method)
        def wrapper(instance, *args, **kwargs):
            with ExitStack() as stack:
                op_cntxt = instance.op_placement_context(name_scope=False)
                stack.enter_context(op_cntxt)
                if name_scope:
                    scope = "{}.{}/".format(instance.long_name,method.__name__)
                    scope = INVALID_NAME_SCOPE_CHAR.sub("", scope)
                    stack.enter_context(tf.name_scope(scope))

                result = method(instance, *args, **kwargs)

                if name_scope and rename_output:
                    if (isinstance(result, Sequence) and 
                        all(isinstance(x, tf.Tensor) for x in result)):
                        return tuple(tf.identity(x, str(result.index(x)))
                                     for x in result)
                    elif isinstance(result, tf.Tensor):
                        return tf.identity(result, name='0')
                return result
        if cache:
            wrapper = cache_method(cache_limit)(wrapper)
        return wrapper
    return decorator
