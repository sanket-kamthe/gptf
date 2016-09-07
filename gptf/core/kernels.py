"""Provides kernels."""
from __future__ import division
from builtins import super
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod
from copy import copy

from overrides import overrides
import numpy as np
import tensorflow as tf

from .params import Param, Parameterized, ParamAttributes, ParamList, autoflow
from .transforms import Exp
from .wrappedtf import tf_method


class Kernel(with_metaclass(ABCMeta, Parameterized)):
    """Base class for kernels.

    Inheriting classes must define `.K()`, `.Kdiag()`.

    Examples:
        >>> class Ones(Kernel, ParamAttributes):
        ...     @tf_method
        ...     @overrides
        ...     def K(self, X, X2):
        ...         if X2 is None:
        ...             X2 = X
        ...         return tf.ones((tf.shape(X)[0], tf.shape(X2)[0]), 
        ...                        dtype=X.dtype)
        ...     @tf_method
        ...     @overrides
        ...     def Kdiag(self, X):
        ...         return tf.ones((tf.shape(X)[0],), dtype=X.dtype)

        >>> sess = tf.InteractiveSession()
        >>> X = tf.constant([[1.]], dtype=tf.float64)
        >>> X2 = tf.constant([[1.], [2.]], dtype=tf.float64)
        >>> Ones().K(X, X2).eval()
        array([[ 1., 1.]])

        You can do maths with kernels! Addition and subtraction:
        >>> threes = Ones() + Ones() + Ones()
        >>> threes.K(X, X2).eval()
        array([[ 3., 3.]])
        >>> (threes - Ones()).K(X, X2).eval()
        array([[ 2., 2.]])
        >>> (-threes).K(X, X2).eval()
        array([[-3., -3.]])
        >>> abs(-threes).K(X, X2).eval()
        array([[ 3.,  3.]])

        Multiplication and division:
        >>> (threes * threes).K(X, X2).eval()
        array([[ 9., 9.]])
        >>> (Ones() / threes).K(X, X2).eval().round(3)
        array([[ 0.333, 0.333]])
        >>> (1 / threes).K(X, X2).eval().round(3)  # 1 / Kernel works
        array([[ 0.333, 0.333]])
        >>> (threes * threes / threes * threes).K(X, X2).eval()
        array([[ 9., 9.]])

        >>> sess.close()

    """
    @abstractmethod
    def K(self, X, X2=None):
        """Builds a tensor that computes the covariance function.

        Args:
            X (tf.Tensor): A tensor of shape `N`x`D`.
            X2 (tf.Tensor | None): A tensor of shape `M`x`D`. If `None`,
                the second argument will be assumed symmetrical.

        Returns:
            (tf.Tensor): The covariance function K(X, X2). Shape `N`x`M`

        """
        NotImplemented

    @abstractmethod
    def Kdiag(self, X):
        """Builds a tensor that computes the diagonal of `self.K(X)`

        Args:
            X (tf.Tensor): A tensor of shape `N`x`D`.

        Returns:
            (tf.Tensor): The diagonal of `self.K(X)`. Shape `N`.

        """
        NotImplemented

    @autoflow((tf.float64, (None, None)), (tf.float64, (None, None)))
    def compute_K(self, X, Z):
        """An autoflowed version of K(X, Z)"""
        return self.K(X, Z)

    @autoflow((tf.float64, (None, None)))
    def compute_K_symm(self, X):
        """An autoflowed version of K(X)"""
        return self.K(X)

    @autoflow((tf.float64, (None, None)))
    def compute_Kdiag(self, X):
        """An autoflowed version of Kdiag(X)"""
        return self.K(X)

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

class PartiallyActive(Kernel, ParamAttributes):
    """A kernel that only uses some of the dimensions of the input.
    
    Attributes:
        wrapped (Kernel): The wrapped kernel.
        active_dims (slice | Iterable[int]): The active dimensions of
            the input. Each input to `.K() or `.Kdiag()` will be sliced
            along its second axis so that only the active dimensions
            remain before being passed to `.wrapped.K()` or 
            `.wrapped.Kdiag()`.

    Examples:
        First, we define an example kernel that lets us inspect the
        arguments passed to `.K()`.
        >>> class Example(Kernel, ParamAttributes):
        ...     @tf_method
        ...     @overrides
        ...     def K(self, X, X2):
        ...         return X, X2
        ...     @tf_method
        ...     @overrides
        ...     def Kdiag(self, X, X2):
        ...         NotImplemented

        Slices are applied to the second axis of the input.
        A slice of `:3` turns `X` into `X[:, :3]`.
        >>> k = PartiallyActive(Example(), slice(None, 3))  # :3
        >>> X = np.arange(5).reshape(1, -1)
        >>> X_k, _ = k.compute_K(X, X)
        >>> X_k.shape
        (1, 3)
        >>> X_k[0]
        array([ 0., 1., 2.])

        If active_dims is a sequence of `int`, only dimensions in the
        sequence will be passed through.
        >>> k.active_dims = [1, 3, 4]
        >>> X_k, _ = k.compute_K(X, X)
        >>> X_k.shape
        (1, 3)
        >>> X_k[0]
        array([ 1., 3., 4.])
            
    """
    def __init__(self, wrapped, active_dims):
        """Initialiser.

        Args:
            wrapped (Kernel): The kernel to wrap. This will be used to
                get the values of K and Kdiag.
            active_dims (slice | Iterable[int]): The active dimensions
                of the kernel.

        """
        super().__init__()
        self.wrapped = wrapped
        self._active_dims = active_dims

    @property
    def active_dims(self):
        return self._active_dims

    @active_dims.setter
    def active_dims(self, value):
        self.clear_cache()
        self.clear_ancestor_caches()
        self._active_dims = value
      
    def _slice(self, X, X2):
        if isinstance(self.active_dims, slice):
            X = X[:, self.active_dims]
            if X2 is not None:
                X2 = X2[:, self.active_dims]
            return X, X2
        else:
            X = tf.transpose(tf.gather(tf.transpose(X), self.active_dims))
            if X2 is not None:
                X2 = tf.transpose(tf.gather(tf.transpose(X2), self.active_dims))
            return X, X2

    @tf_method
    @overrides
    def K(self, X, X2=None):
        return self.wrapped.K(*self._slice(X, X2))

    @tf_method
    @overrides
    def Kdiag(self, X, X2=None):
        return self.wrapped.Kdiag(*self._slice(X, X2))

class Static(Kernel, ParamAttributes):
    """Base class for static kernels.
    
    Kernels that do not depend on the value of the inputs are static.
    The only parameter they require is a variance.

    Attributes:
        variance (Param): The variance of the kernel.

    """
    def __init__(self, variance=1.0):
        """Initialiser.

        NB: It is strongly recommended that `Param` arguments have a 
        transform applied to ensure that they are positive.

        Args:
            variance (Param | float): The variance of the kernel.

        """
        super().__init__()
        self.variance = (variance if isinstance(variance, Param)
                         else Param(variance, transform=Exp()))

    @tf_method
    @overrides
    def Kdiag(self, X):
        return tf.fill([tf.shape(X)[0]], self.variance.tensor)

class White(Static):
    """The White kernel."""
    def K(self, X, X2=None):
        if X2 is None:
            d = tf.fill([tf.shape(X)[0]], self.variance.tensor)
            return tf.diag(d)
        else:
            return tf.zeros((tf.shape(X)[0], tf.shape(X2)[0]), dtype=X.dtype)

class Constant(Static):
    """The Constant (a.k.a. Bias) kernel."""
    def K(self, X, X2):
        if X2 is None:
            X2 = X
        return tf.fill((tf.shape(X)[0], tf.shape(X2)[0]), self.variance.tensor)

Bias = Constant  # a rose by another other name

class Stationary(with_metaclass(ABCMeta, Kernel, ParamAttributes)):
    """Base class for stationary kernels.

    Inheriting classes must define `.trueK` instead of `.K`.

    Stationary kernels depend only on
        r = || X - X2 ||

    This class handles ARD (Automatic Relevance Determination)
    behaviour. If ARD, then the kernal has one length scale per input
    dimension. Otherwise, the kernel is isotropic (one length scale).

    Attributes:
        variance (Param): The variance of the kernel.
        lengthscales (Param | None): The length scale parameters of 
            the kernel. Points are divided by the length scale(s)
            before distances are found between them.
            Note that if ARD is used, this will be `None` until the
            first call to `.K()`.

    """
    def __init__(self, variance=1.0, lengthscales=1.0, ARD=False):
        """Initialiser.

        NB: It is strongly recommended that `Param` arguments have a 
        transform applied to ensure that they are positive.

        Args:
            variance (Param | float): The variance parameter.
            lengthscales (Param | float | np.ndarray): The initial value
                of the length scales. If this is a `Param`, then `ARD`
                and `input_dims` will be ignored and `Param` will be
                used as is.
            ARD (bool): If `ARD`, `float` values for `lengthscales` 
                will be converted to an 

        """
        super().__init__()
        self.variance = (variance if isinstance(variance, Param)
                else Param(variance, transform=Exp()))
        self._ARD = ARD
        if ARD:
            self.lengthscales = None
            self._initial_lengthscales = lengthscales
        elif isinstance(lengthscales, Param):
            self.lengthscales = lengthscales
        else:
            self.lengthscales = Param(lengthscales, transform=Exp())

    @staticmethod
    def trueK(self, X, X2=None):
        """See Kernel.K.__doc__"""
        NotImplemented

    @tf_method
    @overrides
    def K(self, X, X2=None):
        if self._ARD and self.lengthscales is None:
            # accept float or array
            base = np.ones(tf.get_rank(X))
            l = self._initial_lengthscales * base
            self.lengthscales = Param(l, transform=Exp())
        return self.trueK(X, X2)

    @tf_method
    @overrides
    def Kdiag(self, X):
        return tf.fill((tf.shape(X)[0],), self.variance.tensor)

    @tf_method
    def square_dist(self, X, X2=None):
        """The squared distances between X and X2.

        Args:
            X (tf.Tensor): A tensor of shape `N`x`D`.
            X2 (Tf.Tensor | None): A tensor of shape `M`x`D`. If none,
                assumed to be symmetrical.
        
        Returns:
            (tf.Tensor): A tensor of shape `N`x`M`, where
            `self.squared_dist(X, X2)[a, b]` is the squared distance
            between `X[a]` and `X2[b]`.

        """
        X = X / self.lengthscales
        X2 = X if X2 is None else X2 / self.lengthscales
        X = tf.expand_dims(X, 1)
        X2 = tf.expand_dims(X, 0)
        return tf.reduce_sum(tf.square(X - X2), 2)

    @tf_method
    def euclid_dist(self, X, X2):
        """The Euclidean distances between X and X2.

        Args:
            X (tf.Tensor): A tensor of shape `N`x`D`.
            X2 (Tf.Tensor | None): A tensor of shape `M`x`D`. If none,
                assumed to be symmetrical.
        
        Returns:
            (tf.Tensor): A tensor of shape `N`x`M`, where
            `self.squared_dist(X, X2)[a, b]` is the Euclidean distance
            between `X[a]` and `X2[b]`.

        """
        return tf.sqrt(self.squared_dist(X, X2))

class RBF(Stationary):
    """The Radial Basis Function (RBF) or squared exponential kernel."""
    @tf_method
    @overrides
    def trueK(self, X, X2=None):
        return self.variance * tf.exp(-self.squared_dist(X, X2)/2)
        
class Exponential(Stationary):
    """The Exponential kernel."""
    @tf_method
    @overrides
    def trueK(self, X, X2=None):
        return self.variance * tf.exp(-self.euclid_dist(X, X2)/2)

class Matern12(Stationary):
    """The Matern 1/2 kernel"""
    @tf_method
    @overrides
    def trueK(self, X, X2=None):
        r = self.euclid_dist(X, X2)
        return self.variance * tf.exp(-r)

class Matern32(Stationary):
    """The Matern 3/2 kernel"""
    @tf_method
    @overrides
    def trueK(self, X, X2=None):
        r = self.euclid_dist(X, X2)
        sqrt3 = np.sqrt(3.)
        return self.variance * (1. + sqrt3*r) * tf.exp(-sqrt3*r)

class Matern52(Stationary):
    """The Matern 5/2 kernel"""
    @tf_method
    @overrides
    def trueK(self, X, X2=None):
        r2 = self.squared_dist(X, X2)
        r = tf.sqrt(r2)
        sqrt5 = np.sqrt(5.)
        return self.variance * (1. + sqrt5*r + 5/3*r2) * tf.exp(-sqrt5*r)

class Cosine(Stationary):
    """The Cosine kernel."""
    @tf_method
    @overrides
    def trueK(self, X, X2=None):
        return self.variance * tf.cos(self.euclid_distance(X, X2))

#TODO: Implement more kernels

class Negative(Kernel, ParamAttributes):
    """The negative of a kernel."""
    def __init__(self, kernel):
        super().__init__()
        self.negated = kernel

    @tf_method
    @overrides
    def K(self, X, X2=None):
        return tf.neg(self.negated.K(X, X2))

    @tf_method
    @overrides
    def Kdiag(self, X):
        return tf.neg(self.negated.Kdiag(X))

    @overrides
    def __neg__(self):
        return self.negated.copy()

class Absolute(Kernel, ParamAttributes):
    """The absolute value of a kernel."""
    def __init__(self, kernel):
        super().__init__()
        self.absolute = kernel

    @tf_method
    @overrides
    def K(self, X, X2=None):
        return tf.abs(self.absolute.K(X, X2))

    @tf_method
    @overrides
    def Kdiag(self, X):
        return tf.abs(self.absolute.Kdiag(X))

    @overrides
    def __abs__(self):
        return self.copy()

class Additive(Kernel, ParamList):
    """The addition of mean functions."""
    def __init__(self, *kernels):
        super().__init__()
        self.extend(kernels)

    @tf_method
    @overrides
    def K(self, X, X2=None):
        vals = tf.pack([k.K(X, X2) for k in self.children])
        return tf.reduce_sum(vals, 0)

    @tf_method
    @overrides
    def Kdiag(self, X):
        vals = tf.pack([k.Kdiag(X) for k in self.children])
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

class Multiplicative(Kernel, ParamList):
    """The multiplication of kernels."""
    def __init__(self, *kernels):
        super().__init__()
        self.extend(kernels)

    @tf_method
    @overrides
    def K(self, X, X2=None):
        vals = tf.pack([k.K(X, X2) for k in self.children])
        return tf.reduce_prod(vals, 0)

    @tf_method
    @overrides
    def Kdiag(self, X):
        vals = tf.pack([k.Kdiag(X) for k in self.operands])
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

class Divisive(Kernel, ParamAttributes):
    """The division of kernels.
    
    `Divisive(a, b)(X)` returns `a(X) / b(X)`.

    """
    def __init__(self, numerator, denominator):
        super().__init__()
        self.numerator = numerator
        self.denominator = denominator

    @tf_method
    @overrides
    def K(self, X, X2=None):
        return tf.div(self.numerator.K(X, X2), self.denominator.K(X, X2))

    @tf_method
    @overrides
    def Kdiag(self, X):
        return tf.div(self.numerator.Kdiag(X), self.denominator.Kdiag(X))
    
    def __imul__(self, other):
        self.numerator *= other
        return self

    @overrides
    def __mul__(self, other):
        m = self.copy()
        m.numerator *= other
        return m

    @overrides
    def __rmul__(self, other):
        m = self.copy()
        m.numerator = other * m.numerator
        return m

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __itruediv__(self, other):
        self.denominator *= other
        return self

    @overrides
    def __truediv__(self, other):
        m = self.copy()
        m.denominator *= other
        return m

    @overrides
    def __rtruediv__(self, other):
        m = Divisive(self.denominator, self.numerator)
        if other != 1:
            m.numerator = other * m.numerator
        return m

class _one(Kernel, ParamAttributes):
    @tf_method
    @overrides
    def K(self, X, X2=None):
        return tf.ones([], dtype=X.dtype)

    @tf_method
    @overrides
    def Kdiag(self, X):
        return tf.ones([], dtype=X.dtype)

    def __imul__(self, other):
        return other

    @overrides
    def __mul__(self, other):
        return other

    @overrides
    def __rmul__(self, other):
        return other
