"""Provides facilities for calculating likelihoods, densities etc."""
from builtins import super
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

from overrides import overrides
import numpy as np
import tensorflow as tf

from . import densities, transforms
from .params import Parameterized, ParamAttributes, Param
from .wrappedtf import tf_method


class Likelihood(with_metaclass(ABCMeta, Parameterized)):
    """Abstract base class for likelihoods.

    Inheriting classes must implement `.logp()`, `.conditional_mean()`
    and `.conditional_variance()`.

    """
    @abstractmethod
    def logp(self, F, Y):
        """The log density of the data given the function values.

        Args:
            F (tf.Tensor): A set of proposed function values.
            Y (tf.Tensor): The data.

        Returns:
            (tf.Tensor): The log density of the data; `p(Y|F)`.

        """
        NotImplemented

    @abstractmethod
    def conditional_mean(self, F):
        """Computes the mean of the data given values of the latent func.

        If this object represents `p(y|f)`, then this method computes
            \int y p(y|f) dy

        Args:
            F (tf.Tensor): Proposed values for the latent function.

        Returns:
            (tf.Tensor): The mean of the data given the value of the
            latent function.

        """
        NotImplemented

    @abstractmethod
    def conditional_variance(self, F):
        """Computes the variance of the data given values of the latent func.

        If this object represents `p(y|f)`, then this method computes
            \int y^2 p(y|f) dy - (\int y p(y|f) dy)^2

        Args:
            F (tf.Tensor): Proposed values for the latent function.

        Returns:
            (tf.Tensor): The variance of the data given the value of the
            latent function.

        """
        NotImplemented

    @tf_method()
    def predict_mean_and_var(self, mu_F, var_F):
        """The mean & variance of Y given a mean & variance of the latent func.

        if `q(f) = N(mu_F, var_F)` and this object represents `p(y|f)`,
        then this method computes the predictive mean
            \int\int y p(y|f) q(f) df dy

        and the predictive variance
            \int\int y^2 p(y|f) q(f) df dy - (\int\int y p(y|f) q(f) df dy)^2

        Here, we implement a Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.

        """
        NotImplemented

    @tf_method()
    def predict_density(self, mu_F, var_F, Y):
        """The (log) density of Y given a mean & variance of the latent func.

        if `q(f) = N(mu_F, var_F)` and this object represents `p(y|f)`,
        then this method computes the predictive density
            \int p(y=Y|f) q(f) df

        Here, we implement a Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian, Poisson) will implement specific cases.

        """
        NotImplemented

    @tf_method()
    def variational_expectations(self, mu_F, var_F, Y):
        """Compute the expected log density of the data.

        if `q(f) = N(mu_F, var_F)` and this object represents `p(y|f)`,
        then this method computes the predictive density
            \int (\log p(y|f)) q(f) df

        Here, we implement a Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian, Poisson) will implement specific cases.

        """
        NotImplemented


class Gaussian(Likelihood, ParamAttributes):
    def __init__(self, variance=1.):
        """Initialiser.

        Args:
            variance (Param | float): The variance of the likelihood.
                Defaults to 1.

        """
        super().__init__()
        self.variance = (variance if isinstance(variance, Param)
                         else Param(variance, transform=transforms.Exp()))

    @tf_method()
    @overrides
    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance.tensor)

    @tf_method()
    @overrides
    def conditional_mean(self, F):
        return tf.identity(F)

    @tf_method()
    @overrides
    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), self.variance.tensor)

    @tf_method()
    @overrides
    def predict_mean_and_var(self, mu_F, var_F):
        return tf.identity(mu_F), var_F + self.variance.tensor

    @tf_method()
    @overrides
    def predict_density(self, mu_F, var_F, Y):
        return densities.gaussian(mu_F, Y, var_F + self.variance.tensor)

    @tf_method()
    @overrides
    def variational_expectations(self, mu_F, var_F, Y):
        return (-.5 * tf.log(2*np.pi) - .5 * tf.log(self.variance.tensor) -
                .5 * (tf.square(Y - mu_F) + var_F) / self.variance.tensor)
