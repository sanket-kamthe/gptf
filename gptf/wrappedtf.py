# standard library
from functools import wraps
try:  # in case of rogue Python 2.7, use contextlib2 instead of contextlib
    from contextlib import contextmanager, ExitStack
except ImportError:
    from contextlib2 import contextmanager, ExitStack

# nonstandard library
import tensorflow as tf

# local
from .parentable import Parentable


class WrappedTF(Parentable):
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
        tf_session_target (str | dict | None):
            The target under which sessions should run. If this is `None`,
            no arguments will be passed to `tf.session()`. If this is a 
            dictionary, then its contents will be used as keyword arguments 
            for `tf.session()`. Else, this will be the sole argument for
            `tf.session()`. See `.get_session()`.

    Examples:
        `op_placement_context` and `tf_method` can be used to apply
        the appropriate contexts to tensorflow methods. In the following
        
    """
    _NO_DEVICE = object()

    def __init__(self):
        Parentable.__init__(self)
        self.tf_device = None
        self.tf_session_target = None
        self._tf_scope = None

    # This is an attempt to guard against assignement to _NO_DEVICE
    @property
    def NO_DEVICE(self):
        return self._NO_DEVICE

    @staticmethod
    def tf_method(method):
        """Decorator version of `op_placement_context`.
        
        Applies `instance.op_placement_context` to `instance.method(...)`.
        See examples.
        
        Examples:
            In the following example, `Example.method_a` is equivalent 
            to `Example.method_b`.
            >>> class Example(WrappedTF):
            ...     def method_a(self):
            ...         with self.op_placement_context():
            ...             a = tf.constant(2)
            ...             b = tf.constant(3)
            ...             return tf.add(a, b)
            ...     @WrappedTF.tf_method
            ...     def method_b(self):
            ...         a = tf.constant(2)
            ...         b = tf.constant(3)
            ...         return tf.add(a, b)
            >>> e = Example()
            >>> e.tf_device = '/job:worker/task:0'
            >>> a = e.method_a()
            >>> print(a.device)
            /job:worker/task:0
            >>> b = e.method_b()
            >>> b.device == a.device
            True
            
        """
        @wraps(method)
        def wrapper(instance, *args, **kwargs):
            with instance.op_placement_context():
                return method(instance, *args, **kwargs)
        return wrapper

    @contextmanager  # this also gives us context decorator power
    def op_placement_context(self):
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
            if self.parent is not None:
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
        """Gets a TensorFlow session in which ops should be run.
        
        Creates a new session, using the session_target of the highest parent.
        This, when combined with hierachical op placement, allows for
        cluster-aware AutoFlow, amongst other things.

        Returns:
            (tf.Session): A new session, using the session target of the
            highest parent.

        Examples:
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
        return tf.Session(self.highest_parent.tf_session_target)
