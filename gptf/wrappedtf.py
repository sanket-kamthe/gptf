# standard library
from builtins import super, object
from functools import wraps
try:  # in case of rogue Python 2.7, use contextlib2 instead of contextlib
    from contextlib import contextmanager, ExitStack, suppress
except ImportError:
    from contextlib2 import contextmanager, ExitStack, suppress

# nonstandard library
import tensorflow as tf

# local
from .trees import TreeWithCache


class ReusableContextSession(tf.Session):
    """Monkey patches `tf.Session` so that it can be reused as a context.

    Note that this means that the session is not closed when its context ends.
    This makes `with ReusableContextSession():` equivalent to
    `with tf.Session().as_default():`.
    
    Examples:
        >>> sess = ReusableContextSession()

        The context is _reusable_, not reentrant. This is fine:
        >>> with sess:
        ...     pass
        >>> with sess:
        ...     pass

        This is not:
        >>> with sess:
        ...     with sess:    
        ...         pass
        Traceback (most recent call last):
            ...
        RuntimeError: generator didn't yield
    
    """
    @wraps(tf.Session.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__context_manager = None

    @wraps(tf.Session.__enter__)
    def __enter__(self):
        if self.__context_manager is None:
            self.__context_manager = self.as_default()
        self.__context_manager.__enter__()
        return self

    @wraps(tf.Session.__exit__)
    def __exit__(self, type_, value, traceback):
        self.__context_manager.__exit__(type_, value, traceback)
        self.__context_manager = None

class WrappedTF(TreeWithCache):
    """Provides facilities for keeping TensorFlow behind the scenes.

    WARNING: `WrappedTF` assumes that its parent, and indeed all things
    higher than it in the tree, are also `WrappedTF`. Make sure that the
    root of the tree has implemented `.get_session()`, and that
    the direct parent has implemented `.op_placement_context()`.

    Attributes:
        NO_DEVICE (object): A class-level constant, used to specify an
            empty op placement context. Do 
        tf_device (str | Callable[[tf.Operation], str] | tf.DeviceSpec
                | WrappedTF.NO_DEVICE | None):
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

    Examples:
        self._on_op_placement_context_change()
        self._tf_graph = value
        `op_placement_context` and `tf_method` can be used to apply
        the appropriate contexts to tensorflow methods. In the following
        
    """
    _NO_DEVICE = object()

    def __init__(self):
        super().__init__()
        self._tf_graph = None
        self._tf_device = None
        self._tf_session_target = None
        self._tf_session = None

    # This is an attempt to guard against assignement to _NO_DEVICE
    @property
    def NO_DEVICE(self):
        return self._NO_DEVICE

    @property
    def tf_device(self):
        return self._tf_device

    @tf_device.setter
    def tf_device(self, value):
        self._on_op_placement_context_change()
        self._tf_device = value

    @property
    def tf_graph(self):
        return self._tf_graph

    @tf_graph.setter
    def tf_graph(self, value):
        self._on_op_placement_context_change()
        self._tf_graph = value

    @property
    def tf_session_target(self):
        return self._tf_session_target

    @tf_session_target.setter
    def tf_session_target(self, value):
        self._maybe_kill_session()
        self._tf_session_target = value

    @staticmethod
    def tf_method(method):
        """Decorator version of `op_placement_context`.
        
        Applies `instance.op_placement_context(name_scope=False)` 
        to `instance.method(...)`, and opens a name scope that matches the
        method. See examples.
        
        Examples:
            In the following example, `Example.method_a` is equivalent 
            to `Example.method_b`.
            >>> class Example(WrappedTF):
            ...     def method_a(self):
            ...         with self.op_placement_context():
            ...             with tf.name_scope(self.long_name + '.method_a/'):
            ...                 a = tf.constant(2)
            ...                 b = tf.constant(3)
            ...                 return tf.add(a, b)
            ...     @WrappedTF.tf_method
            ...     def method_b(self):
            ...         a = tf.constant(2)
            ...         b = tf.constant(3)
            ...         return tf.add(a, b)

            Devices are set properly in both methods:
            >>> e = Example()
            >>> e.tf_device = '/job:worker/task:0'
            >>> a = e.method_a()
            >>> print(a.device)
            /job:worker/task:0
            >>> b = e.method_b()
            >>> b.device == a.device
            True

            The method name is appended to the name scope!
            >>> print(a.name)
            unnamed.method_a/Add:0
            >>> print(b.name)
            unnamed.method_b/Add:0
            
        """
        @wraps(method)
        def wrapper(instance, *args, **kwargs):
            scope = "{}.{}/".format(instance.long_name, method.__name__)
            with instance.op_placement_context(), tf.name_scope(scope):
                    return method(instance, *args, **kwargs)
        return wrapper

    @contextmanager
    def op_placement_context(self, name_scope=True):
        """Applies op placement rules based on the object hierarchy.

        Examples:
            Choose the op placement context by assigning to `.tf_device`:
            >>> a, b, c, d, e = [WrappedTF() for _ in range(5)]
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
                stack.enter_context(self.parent.op_placement_context())

            if self.tf_device is not None:
                dev = self.tf_device
                if dev is self.NO_DEVICE:
                    dev = None
                stack.enter_context(tf.device(dev))

            # enter "absolute" name scope by appending "/"
            stack.enter_context(tf.name_scope(self.long_name + "/"))

            yield

    def get_session(self):
        """Gets a TensorFlow session in which ops can be run.
        
        Returns the default session if there is one. Else, returns a 
        persistent session using the session target of the highest parent.

        Returns:
            (tf.Session): The default session if there is one, else a session 
            matching the session target of the highest parent.

        Examples:
            >>> w = WrappedTF()

            If there is already a default session, returns that one:
            >>> with tf.Session() as sess:
            ...     w.get_session() is sess
            True
            
            Else, returns the same session across multiple calls:
            >>> sess2 = w.get_session()
            >>> sess2 is sess
            False
            >>> sess2 is w.get_session()
            True
            >>> sess2.close()
            
            >>> class Example(WrappedTF):
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
        default = tf.get_default_session()
        if default is not None:
            return default
        else:
            if self.parent is None:
                self._maybe_create_session()
                return self._tf_session
            else:
                return self.highest_parent.get_session()

    def on_session_birth(self):
        """Called just after the session of the highest parent is created."""
        pass

    def on_session_death(self):
        """Called just before the session of the highest parent is closed."""
        pass

    def _maybe_create_session(self):
        """Handles session creation if necessary.
        
        If no session already exists, creates a session, then calls 
        `on_session_birth` for all objects lower in the tree.

        Examples:
            >>> class Example(WrappedTF):
            ...     def on_session_birth(self):
            ...         name = self.long_name
            ...         print('{}.on_session_birth called!'.format(name))
            ...         super().on_session_birth()
            >>> w = Example()
            >>> w.child = Example()
            >>> w.child.child = Example()

            >>> w.child._maybe_create_session()
            unnamed.child.on_session_birth called!
            unnamed.child.child.on_session_birth called!
        
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
            return ReusableContextSession(**kwargs)

        if self._tf_session is None:
            self._tf_session = new_session()
            for node in self:
                node.on_session_birth()

    def _maybe_kill_session(self): 
        """Handles session destruction if necessary.

        If a session already exists, kills it then calls 
        `on_session_death` for all objects in the tree.

        Examples:
            >>> class Example(WrappedTF):
            ...     def on_session_death(self):
            ...         name = self.long_name
            ...         print('{}.on_session_death called!'.format(name))
            ...         super().on_session_death()
            >>> w = Example()
            >>> w.child = Example()
            >>> w.child.child = Example()

            >>> w.child._tf_session = tf.Session()
            >>> w.child._maybe_kill_session()
            unnamed.child.on_session_death called!
            unnamed.child.child.on_session_death called!
        
        """
        if self._tf_session is not None:
            for node in self:
                node.on_session_death()
            self._tf_session.close()
            self._tf_session = None

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

    def on_new_parent(self, new_parent):
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
            >>> class Example(WrappedTF):
            ...     def __init__(self, name):
            ...         super().__init__()
            ...         self.fallback_name = name
            ...         self.should_print = True
            ...     def on_session_birth(self):
            ...         if self.should_print:
            ...             name = self.long_name
            ...             print('{}.on_session_birth called!'.format(name))
            ...         super().on_session_birth()
            ...     def on_session_death(self):
            ...         if self.should_print:
            ...             name = self.long_name
            ...             print('{}.on_session_death called!'.format(name))
            ...         super().on_session_death()
            >>> w = Example('w')
            >>> x = Example('x')
            >>> y = Example('y')
            >>> z = Example('z')
            >>> w.child_0 = x
            >>> y.child = z
            >>> def setup():
            ...     objs = [w, x, y, z]
            ...     # create sessions
            ...     for o in objs: o.should_print = False
            ...     for o in objs: o.get_session()
            ...     for o in objs: o.should_print = True
            ...     # fill caches
            ...     for o in objs: o.cache['tf'] = 'compiled tf function'

            When `y` gets a parent, it should close its session and 
            clear its subtree's cache. `w`, which knows nothing about `y` so
            should not have any ops that depend on `y`'s device context,
            should not have its cache cleared; the same goes for `x`.
            >>> setup()
            >>> w.child_1 = y
            y.on_session_death called!
            y.child.on_session_death called!
            w.child_1.on_session_birth called!
            w.child_1.child.on_session_birth called!
            >>> 'tf' in w.cache and 'tf' in w.child_0.cache
            True
            >>> 'tf' in w.child_1.cache or 'tf' in w.child_1.child.cache
            False

            There is now only one tree, of which `w` is the highest parent,
            so only `w` should be have a session.
            >>> def has_session(node):
            ...     # NB: ._tf_session is not a part of the public API.
            ...     return node._tf_session is not None
            >>> has_session(w)
            True
            >>> not any(has_session(i) for i in [x, y, z])
            True
            
            Deleting `w.child_1` means that `y` loses a parent. `w` knows
            about `y` and might have ops that depend on `y`'s device context,
            so we should clear its cache. `w.child_0` still knows nothing
            about `y`, so its cache should still not be cleared.
            >>> setup()
            >>> del w.child_1
            w.child_1.on_session_death called!
            w.child_1.child.on_session_death called!
            >>> 'tf' in w.child_0.cache
            True
            >>> 'tf' in w.cache 
            False
            >>> 'tf' in y.cache or 'tf' in y.child.cache
            False

            When a `WrappedTF` is moved between trees, the caches in the old
            tree are cleared and session births are registered in the subtree
            of the moving `WrappedTF`.
            >>> setup()
            >>> y.child_1 = w.child_0
            w.child_0.on_session_death called!
            y.child_1.on_session_birth called!
            >>> 'tf' in y.cache and 'tf' in y.child.cache
            True
            >>> 'tf' in w.cache or 'tf' in y.child_1.cache
            False

        """
        if self._tf_session is not None:
            self._maybe_kill_session()
        elif self.highest_parent._tf_session is not None:
            for node in self:
                node.on_session_death()
        self._on_op_placement_context_change()

        super().on_new_parent(new_parent)  # move to new tree

        if self.highest_parent._tf_session is not None:
            for node in self:
                node.on_session_birth()
