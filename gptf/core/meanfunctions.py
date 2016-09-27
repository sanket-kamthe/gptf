"""Provides mean functions."""
from builtins import super
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

from overrides import overrides
import numpy as np
import tensorflow as tf

from .params import Parameterized, ParamAttributes, ParamList, Param
from .wrappedtf import tf_method


class MeanFunction(with_metaclass(ABCMeta, Parameterized)):
    """Abstract base class for mean functions.

    Inheriting classes must define `.__call__()`.

    Examples:
        >>> class Ones(MeanFunction, ParamAttributes):
        ...     @tf_method()
        ...     @overrides
        ...     def __call__(self, X):
        ...         return tf.ones(tf.shape(X), dtype=X.dtype)

        >>> sess = tf.InteractiveSession()
        >>> X = tf.constant([1., 2., 3., 4.], dtype=tf.float64)
        >>> Ones()(X).eval()
        array([ 1.,  1.,  1.,  1.])

        You can do maths with mean functions! Addition and subtraction:
        >>> threes = Ones() + Ones() + Ones()
        >>> threes(X).eval()
        array([ 3.,  3.,  3.,  3.])
        >>> (threes - Ones())(X).eval()
        array([ 2.,  2.,  2.,  2.])
        >>> (-threes)(X).eval()
        array([-3., -3., -3., -3.])
        >>> abs(-threes)(X).eval()
        array([ 3.,  3.,  3.,  3.])

        Multiplication and division:
        >>> (threes * threes)(X).eval()
        array([ 9., 9., 9., 9.])
        >>> (Ones() / threes)(X).eval().round(3)
        array([ 0.333, 0.333, 0.333, 0.333])
        >>> (1 / threes)(X).eval().round(3)  # this works as a special case
        array([ 0.333, 0.333, 0.333, 0.333])
        >>> (threes * threes / threes * threes)(X).eval()
        array([ 9., 9., 9., 9.])

        >>> sess.close()

    """
    @abstractmethod
    def __call__(self, X):
        """Calls the mean function.

        Args:
            X (tf.Tensor): The input data. Each row of X represents one
                datum.

        Returns:
            (tf.Tensor): The mean of the posterior distribution.

        """
        NotImplemented

    def __pos__(self):
        return self

    def __neg__(self):
        return Negative(self)

    def __abs__(self):
        return Absolute(self)

    def __add__(self, other):
        if isinstance(other, Additive):
            return other.__radd__(self)
        else:
            return Additive(self, other)

    def __radd__(self, other):
        return Additive(other, self)

    def __sub__(self, other):
        return self + Negative(other)

    def __rsub__(self, other):
        return other + Negative(self)

    def __mul__(self, other):
        if isinstance(other, Multiplicative) or isinstance(other, Divisive):
            return other.__rmul__(self)
        else:
            return Multiplicative(self, other)

    def __rmul__(self, other):
        return Multiplicative(other, self)

    def __div__(self, other):
        return self.__truediv__(other)

    def __rdiv__(self, other):
        return self.__rtruediv__(other)

    def __truediv__(self, other):
        if isinstance(other, Divisive):
            return other.__rmul__(self)
        else:
            return Divisive(self, other)

    def __rtruediv__(self, other):
        if other == 1:
            return Divisive(_one(), self)
        return Divisive(other, self)

class Zero(MeanFunction, ParamAttributes):
    def __init__(self, num_latent_functions=0):
        """Initializer.

        Args:
            num_latent_functions (int): The number of latent functions,
                i.e. the number of columns of Y.

        """
        super().__init__()
        self.num_latent_functions = num_latent_functions

    @tf_method()
    @overrides
    def __call__(self, X):
        """Calls the mean function.

        Examples:
            >>> X = tf.constant([[ 1.], 
            ...                  [ 2.],
            ...                  [ 3.],
            ...                  [ 4.]], dtype=tf.float64)
            >>> mean_func = Zero()
            >>> sess = mean_func.get_session()
            >>> sess.run(mean_func(X))
            array([[ 0.], 
                   [ 0.], 
                   [ 0.],
                   [ 0.]])

        """
        return tf.zeros([tf.shape(X)[0], 1], dtype=tf.as_dtype(X.dtype))

    def __imul__(self, other):
        return self

    @overrides
    def __mul__(self, other):
        return Zero()

    @overrides
    def __rmul__(self, other):
        return Zero()

    @overrides
    def __add__(self, other):
        return other

    @overrides
    def __radd__(self, other):
        return other

class Linear(MeanFunction, ParamAttributes):
    """y_i = A x_i + b"""
    def __init__(self, A=np.ones((1, 1)), b=np.zeros((1,))):
        """Initializer.

        NB: `Y` is `self(X)`.

        Args:
            A (Param | np.ndarray): A matrix which maps each element of
                `X` to `Y`. If `X` is an `N`x`D` matrix and `Y` is 
                intended to be `N`x`Q`, then `A` should be `D`x`Q`.
            b (Param | np.ndarray): An additive constant to `Y`. If
                `Y` is intended to be `N`x`Q` then `b` should be a
                rank 1 tensor of length `Q`.

        """
        super().__init__()
        self.A = A if isinstance(A, Param) else Param(A)
        self.b = b if isinstance(b, Param) else Param(b)

    @tf_method()
    @overrides
    def __call__(self, X):
        """Calls the mean function.

        Examples:
            >>> X = tf.constant([[ 1., 2.], 
            ...                  [ 3., 4.],
            ...                  [ 0., 0.],
            ...                  [ 1., 3.]], dtype=tf.float64)            
            >>> A = np.array([[ 1., 2., 3., 4.],
            ...               [ 1., 1., 1., 1.]])
            >>> b = np.array([ 1., 2., 3., 4.])
            >>> mean_func = Linear(A, b)
            >>> sess = mean_func.get_session()
            >>> sess.run(mean_func(X))
            array([[  4.,   6.,   8.,  10.], 
                   [  8.,  12.,  16.,  20.], 
                   [  1.,   2.,   3.,   4.],
                   [  5.,   7.,   9.,  11.]])

        """
        return tf.matmul(X, self.A.tensor) + self.b.tensor

class Constant(MeanFunction, ParamAttributes):
    """y_i = c,,"""
    def __init__(self, c=np.array([0])):
        """Initializer.

        Args:
            c (Param | np.ndarray): The constant mean value; a rank 1
                tensor.

        """
        super().__init__()
        self.c = c if isinstance(c, Param) else Param(c)

    @tf_method()
    @overrides
    def __call__(self, X):
        """Calls the mean function.

        Examples:
            >>> X = tf.constant([[ 1.], 
            ...                  [ 2.],
            ...                  [ 3.],
            ...                  [ 4.]], dtype=tf.float64)            
            >>> c = np.array([ 1., 2., 3.])
            >>> mean_func = Constant(c)
            >>> sess = mean_func.get_session()
            >>> sess.run(mean_func(X))
            array([[ 1., 2., 3.], 
                   [ 1., 2., 3.], 
                   [ 1., 2., 3.],
                   [ 1., 2., 3.]])

        """
        return tf.tile(tf.expand_dims(self.c.tensor, 0), (tf.shape(X)[0], 1))

class Negative(MeanFunction, ParamAttributes):
    """The negative of a mean function."""
    def __init__(self, function):
        super().__init__()
        self.negated = function

    @tf_method()
    @overrides
    def __call__(self, X):
        return tf.neg(self.negated(X))

    @overrides
    def __neg__(self):
        return self.negated.copy()

class Absolute(MeanFunction, ParamAttributes):
    """The absolute value of a mean function."""
    def __init__(self, function):
        super().__init__()
        self.absolute = function

    @tf_method()
    @overrides
    def __call__(self, X):
        return tf.abs(self.absolute(X))

    @overrides
    def __abs__(self):
        return self.copy()

class Additive(MeanFunction, ParamList):
    """The addition of mean functions."""
    def __init__(self, *functions):
        super().__init__()
        self.extend(functions)

    @tf_method()
    @overrides
    def __call__(self, X):
        vals = tf.pack([f(X) for f in self.children])
        return tf.reduce_sum(vals, 0)

    def __iadd__(self, other):
        self.append(other)
        return self

    @overrides
    def __add__(self, other):
        m = self.copy()
        m += other
        return m

    @overrides
    def __radd__(self, other):
        m = self.copy()
        m.insert(0, other)
        return m

class Multiplicative(MeanFunction, ParamList):
    """The multiplication of mean functions."""
    def __init__(self, *functions):
        super().__init__()
        self.extend(functions)

    @tf_method()
    @overrides
    def __call__(self, X):
        vals = tf.pack([f(X) for f in self.children])
        return tf.reduce_prod(vals, 0)

    def __imul__(self, other):
        self.append(other)
        return self

    @overrides
    def __mul__(self, other):
        m = self.copy()
        m += other
        return m

    @overrides
    def __rmul__(self, other):
        m = self.copy()
        m.insert(0, other)
        return m

class Divisive(MeanFunction, ParamAttributes):
    """The division of mean functions.
    
    `Divisive(a, b)(X)` returns `a(X) / b(X)`.

    """
    def __init__(self, numerator, denominator):
        super().__init__()
        self.numerator = numerator
        self.denominator = denominator

    def clone(self):
        return Divisive(self.numerator, self.denominator)

    @tf_method()
    @overrides
    def __call__(self, X):
        return tf.div(self.numerator(X), self.denominator(X))
    
    def __imul__(self, other):
        self.numerator *= other
        return self

    @overrides
    def __mul__(self, other):
        m = self.clone()
        m.numerator *= other
        return m

    @overrides
    def __rmul__(self, other):
        m = self.clone()
        m.numerator = other * m.numerator
        return m

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __itruediv__(self, other):
        self.denominator *= other
        return self

    @overrides
    def __truediv__(self, other):
        m = self.clone()
        m.denominator *= other
        return m

    @overrides
    def __rtruediv__(self, other):
        m = Divisive(self.denominator, self.numerator)
        if other != 1:
            m.numerator = other * m.numerator
        return m

class _one(MeanFunction, ParamAttributes):
    @tf_method()
    @overrides
    def __call__(self, X):
        return tf.ones([], dtype=tf.as_dtype(X.dtype))

    def __imul__(self, other):
        return other

    @overrides
    def __mul__(self, other):
        return other

    @overrides
    def __rmul__(self, other):
        return other
