"""Provides classes for Robust Bayesian Committee Machines."""
#TODO: Work out the maths for a non-zero prior mean function?
from builtins import super
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

from overrides import overrides

from gptf import GPModel, Parameterized, ParamList, tf_method

class WeightFunction(with_metaclass(ABCMeta, Parameterized)):
    @abstractmethod
    def __call__(self, experts, points):
        """Tensors for the weights for the experts at the points.

        Args:
            experts (Sequence[GPModel]): the experts to calculate
                the weights of.
            points (Sequence[GPModel]): the points at which to
                calculate the weights of the model.

        Returns:
            (Tuple[tf.Tensor]): The weights of the experts at the
            points. Each tensor has shape `(num_points)`
        """
        

class Reduction(GPModel, ParamList):
    ...


class PoE(GPModel, ParamList):
    """Combines the predictions of its children using the PoE model.
    
    In the Product of Experts (PoE) model, the variance of the 
    posterior distribution is the harmonic mean of the posterior 
    variances of the child experts. The mean of the posterior is a 
    weighted sum of the means of the child experts.
        
    .. math::
        σ^{-2}_PoE = \sum_{k=1}^{M} σ^{-2}_k
        μ_PoE = σ^2_PoE \sum_{k=1}^{M} μ_k σ^{-2}_k

    where :math:`μ_PoE` and :math:`σ^2_PoE` are the final posterior
    mean and variance, :math:`M` is the number of child experts and
    :math:`μ_k` and :math:`σ^2_k` are the posterior mean and variance
    for the :math:`k`th child.

    """
    def __init__(self, children):
        """Initialiser.

        Args:
            children (Sequence[GPModel]): The experts to combine the
                opinions of.

        """
        super().__init__()
        self.extend(children)

    @tf_method()
    @overrides
    def build_log_likelihood(self):
        lmls = [child.build_log_likelihood() for child in self.children]
        return tf.reduce_sum(tf.pack(lmls, 0), 0)

    @tf_method()
    @overrides
    def build_prior_mean_var(self, test_points):
        mu, var = zip(*[child.build_prior_mean_var(test_points)
                        for child in self.children])
        mu, var = tf.pack(mu, 0), tf.pack(var, 0)
        return tf.reduce_sum(mu, 0) / len(mu), tf.reduce_sum(var, 0) / len(var)

    @tf_method()
    @overrides
    def build_posterior_mean_var(self, test_points):
        mu, var = zip(*[child.build_posterior_mean_var(test_points)
                        for child in self.children])
        mu, var = tf.pack(mu, 0), tf.pack(var, 0)
        joint_var = 1 / tf.reduce_sum(1 / var, 0)
        joint_mu = joint_var * tf.reduce_sum(mean / var, 0)
        return joint_mu, joint_var

class gPoE(GPModel, ParamList):
    """Combines the predictions of its children using the gPoE model.
    
    The generalised Product of Experts (gPoE) model is similar to the
    PoE model, except that we give each expert a weight. The variance
    of the posterior distribution is a weighted harmonic mean of the 
    posterior variances of the child experts. The mean of the 
    posterior is a weighted sum of the means of the child experts.

    .. math::
        σ^{-2}_gPoE = \sum_{k=1}^{M} β_k σ^{-2}_c
        μ_gPoE = σ^2_gPoE \sum_{k=1}^{M} β_k μ_k σ^{-2}_k

    where :math:`μ_gPoE` and :math:`σ^2_PoE` are the final posterior
    mean and variance, :math:`M` is the number of child experts and
    :math:`μ_k` and :math:`σ^2_k` are the posterior mean and variance
    for the :math:`k`th child and :math:`β_k` is the weight of the
    :math:`k`th child.

    Note that when :math:`\sum_{k} β_k = 1`, the model falls back to
    the prior outside the range of the data.

    """
    def __init__(self, children, weightfunction):
        """Initialiser.

        Args:
            children (Sequence[GPModel]): The experts to combine the
                opinions of.
            weightfunction (Sequence[tf.Tensor]): The weight

        """
        super().__init__()
        self.extend(children)
        self.weights = weights

    @tf_method()
    @overrides
    def build_log_likelihood(self):
        lmls = [child.build_log_likelihood() for child in self.children]
        return tf.reduce_sum(tf.pack(lmls, 0), 0)

    @tf_method()
    @overrides
    def build_prior_mean_var(self, test_points):
        mu, var = zip(*[child.build_prior_mean_var(test_points)
                        for child in self.children])
        mu, var = tf.pack(mu, 0), tf.pack(var, 0)
        return tf.reduce_sum(mu, 0) / len(mu), tf.reduce_sum(var, 0) / len(var)

    @tf_method()
    @overrides
    def build_posterior_mean_var(self, test_points):
        mu, var =  zip(*[child.build_posterior_mean_var(test_points)
                         for child in self.children])
        mu, var = tf.pack(mu, 0), tf.pack(var, 0)
        joint_var = 1 / tf.reduce_sum(1 / var, 0)
        joint_mu = joint_var * tf.reduce_sum(mean / var, 0)
        return joint_mu, joint_var

