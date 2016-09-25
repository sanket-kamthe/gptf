# -*- encoding: utf-8 -*-
"""Provides classes for Robust Bayesian Committee Machines."""
#TODO: Work out the maths for a non-zero prior mean function?
from __future__ import division
from builtins import super, zip, map
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
import operator

from overrides import overrides
import numpy as np
import tensorflow as tf

from gptf import GPModel, Parameterized, ParamList, ParamAttributes, tf_method

def cao_fleet_weights(experts, points):
    """The predictive power of the experts at the points.

    The predictive power is calculated as

    .. math::
        β_k(x_\star)=\\frac{1}{2} (\ln (σ^\star_k)^2 
                                   - \ln σ^{-2}_k(x_\star))

    where :math:`β_k(x_\star)` is the predictive power of the
    :math:`k`th expert at the point :math:`x_\star`, 
    :math:`(σ^\star_k)^2` is the prior variance of the :math:`k`th
    expert and :math:`σ^{-2}_k(x_\star)` is the posterior variance
    of the :math:`k`th expert at the point :math:`x_\star`.

    Args:
        experts (Sequence[GPModel]): the experts to calculate
            the weights of.
        points (Sequence[GPModel]): the points at which to
            calculate the weights of the model.

    Returns:
        (Tuple[tf.Tensor]): The weights of the experts at the
        points. Each tensor has shape `(num_points)`.

    """
    def weight(expert):
        prior_var = expert.prior_mean_var(points)[1]
        post_var = expert.posterior_mean_var(points)[1]
        weight = .5 * (tf.log(prior_var) - tf.log(post_var))
        # return very small value instead of zero
        return tf.maximum(weight, np.finfo(np.float64).eps)
    return tuple(weight(expert) for expert in experts)

def equal_weights(experts, points):
    """Gives each expert an equal weight.

    .. math::
        \\forall k \in 0..M,\ β_k = 1 / M

    where :math:`β_k` is the weight of the :math:`k`th expert at
    every point and :math:`M` is the number of experts.

    The dtype returned matches the dtype of `points`.
    
    Args:
        experts (Sequence[GPModel]): the experts to calculate
            the weights of.
        points (Sequence[GPModel]): the points at which to
            calculate the weights of the model.
    
    Returns:
        (Tuple[tf.Tensor]): The weights of the experts at the
        points. Each tensor has shape `(num_points)`.

    """
    beta = tf.constant(1 / len(experts), dtype=points.dtype)
    beta = tf.fill((tf.shape(points)[0],), beta)
    return tuple(beta for _ in experts)

def ones_weights(experts, points):
    """All weights are 1.

    The dtype returned matches the dtype of `points`.
    
    Args:
        experts (Sequence[GPModel]): the experts to calculate
            the weights of.
        points (Sequence[GPModel]): the points at which to
            calculate the weights of the model.
    
    Returns:
        (Tuple[tf.Tensor]): The weights of the experts at the
        points. Each tensor has shape `(num_points)`.

    """
    ones = tf.ones((tf.shape(points)[0],), dtype=points.dtype)
    return tuple(ones for _ in experts)

class PoEReduction(GPModel, ParamList):
    """Combines the predictions of its children using the PoE model.
    
    In the Product of Experts (PoE) model, the variance of the 
    posterior distribution is the harmonic mean of the posterior 
    variances of the child experts. The mean of the posterior is a 
    weighted sum of the means of the child experts multiplied by
    the posterior variance.
        
    .. math::

        σ^{-2}_PoE &= \sum_{k=1}^{M} σ^{-2}_k \\\\
        μ_PoE &= σ^2_PoE \sum_{k=1}^{M} μ_k σ^{-2}_k

    where :math:`μ_PoE` and :math:`σ^2_PoE` are the final posterior
    mean and variance, :math:`M` is the number of child experts and
    :math:`μ_k` and :math:`σ^2_k` are the posterior mean and variance
    for the :math:`k`th child.

    Attributes: See `GPModel` and `ParamList`.

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
        """The arithetic mean of the prior mean / variance of the chilren."""
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

class gPoEReduction(GPModel, ParamList):
    """Combines the predictions of its children using the gPoE model.
    
    The generalised Product of Experts (gPoE) model is similar to the
    PoE model, except that we give each expert a weight. The variance
    of the posterior distribution is a weighted harmonic mean of the 
    posterior variances of the child experts. The mean of the 
    posterior is a weighted sum of the means of the child experts
    multiplied by the posterior variance.

    .. math::

        σ^{-2}_gPoE &= \sum_{k=1}^{M} β_k σ^{-2}_c \\\\
        μ_gPoE &= σ^2_gPoE \sum_{k=1}^{M} β_k μ_k σ^{-2}_k

    where :math:`μ_gPoE` and :math:`σ^2_PoE` are the final posterior
    mean and variance, :math:`M` is the number of child experts and
    :math:`μ_k` and :math:`σ^2_k` are the posterior mean and variance
    for the :math:`k`th child and :math:`β_k` is the weight of the
    :math:`k`th child.

    Note that when :math:`\sum_{k} β_k = 1`, the model falls back to
    the prior outside the range of the data.

    Attributes:
        weightfunction(Callable[[Sequence[GPModel], tf.Tensor],
            Tuple[tf.Tensor]]): A function used to calculate the
            weight of the experts.

    For other attributes, see `GPModel` and `ParamList`.

    """
    def __init__(self, children, weightfunction):
        """Initialiser.

        Args:
            children (Sequence[GPModel]): The experts to combine the
                opinions of.
            weightfunction (Callable[[Sequence[GPModel], tf.Tensor],
                Tuple[tf.Tensor]]): A function used to calculate the 
                weights of the experts.

        """
        super().__init__()
        self.extend(children)
        self.weightfunction = weightfunction

    @tf_method()
    @overrides
    def build_log_likelihood(self):
        lmls = [child.build_log_likelihood() for child in self.children]
        return tf.reduce_sum(tf.pack(lmls, 0), 0)

    @tf_method()
    @overrides
    def build_prior_mean_var(self, test_points):
        """A weighted mean of the prior means and variances of the experts."""
        mu, var = zip(*[child.build_prior_mean_var(test_points)
                        for child in self.children])
        weight = self.weightfunction(self.children, test_points)
        mu, var, weight = map(lambda x: tf.pack(x, 0), (mu, var, weight))

        weight_total = tf.reduce_sum(weight, 0)
        joint_mu = tf.reduce_sum(weight * mu, 0) / weight_total
        joint_var = tf.reduce_sum(weight * var, 0) / weight_total
        return joint_mu, joint_var

    @tf_method()
    @overrides
    def build_posterior_mean_var(self, test_points):
        mu, var = zip(*[child.build_posterior_mean_var(test_points)
                        for child in self.children])
        weight = self.weightfunction(self.children, test_points)
        mu, var, weight = map(lambda x: tf.pack(x, 0), (mu, var, weight))

        joint_var = 1 / tf.reduce_sum(weight / var, 0)
        joint_mu = joint_var * tf.reduce_sum(weight * mean / var, 0)
        return joint_mu, joint_var

class BCMReduction(GPModel, ParamList):
    """Combines the predictions of its children using the BCM model.
    
    In the Bayesian Committee Machine (BCM) model, the variance of the 
    posterior distribution is the harmonic mean of the posterior
    variances of the child experts, with a correction term based on
    the prior variance. The mean of the posterior is a weighted sum of 
    the means of the child experts multiplied by the posterior variance.
        
    .. math::

        σ^{-2}_BCM &= (\sum_{k=1}^{M} σ^{-2}_k)
                      + (1 - M) σ^{-2}_{\star\star} \\\\
        μ_BCM &= σ^2_BCM \sum_{k=1}^{M} μ_k σ^{-2}_k

    where :math:`μ_BCM` and :math:`σ^2_BCM` are the final posterior
    mean and variance, :math:`M` is the number of child experts,
    :math:`σ^{-2}_{\star\star}` is the prior precision and
    :math:`μ_k` and :math:`σ^2_k` are the posterior mean and variance
    for the :math:`k`th child.

    Attributes: See `GPModel` and `ParamList`.

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
        """The arithetic mean of the prior mean / variance of the chilren."""
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
        prior_precision = 1 / self.build_prior_mean_var(points)[1]
        joint_var = 1/ (tf.reduce_sum(1 / var, 0)
                        + (1 - len(self.children)) * prior_precision)
        joint_mu = joint_var * tf.reduce_sum(mean / var, 0)
        return joint_mu, joint_var

class rBCMReduction(GPModel, ParamList):
    """Combines the predictions of its children using the rBCM model.
    
    In the Bayesian Committee Machine (rBCM) model, the variance of the 
    posterior distribution is the harmonic mean of the posterior
    variances of the child experts, with a correction term based on
    the prior variance and the sum of the weights. The mean of the 
    posterior is a weighted sum of the means of the child experts.
        
    .. math::

        σ^{-2}_rBCM &= (\sum_{k=1}^{M} β_k σ^{-2}_k)
                      + (1 - \sum_{k=1}^{M} β_k) σ^{-2}_{\star\star}\\\\
        μ_rBCM &= σ^2_rBCM \sum_{k=1}^{M} β_k μ_k σ^{-2}_k

    where :math:`μ_rBCM` and :math:`σ^2_rBCM` are the final posterior
    mean and variance, :math:`M` is the number of child experts,
    :math:`σ^{-2}_{\star\star}` is the prior precision and
    :math:`μ_k` and :math:`σ^2_k` are the posterior mean and variance
    for the :math:`k`th child and :math:`β_k` is the weight of the
    :math:`k`th child.

    The model always falls back to the prior outside of the range of
    the data.

    Attributes:
        weightfunction(Callable[[Sequence[GPModel], tf.Tensor],
            Tuple[tf.Tensor]]): A function used to calculate the
            weight of the experts.

    For other attributes, see `GPModel` and `ParamList`.

    """
    def __init__(self, children, weightfunction):
        """Initialiser.

        Args:
            children (Sequence[GPModel]): The experts to combine the
                opinions of.
            weightfunction (Callable[[Sequence[GPModel], tf.Tensor],
                Tuple[tf.Tensor]]): A function used to calculate the 
                weights of the experts.

        """
        super().__init__()
        self.extend(children)
        self.weightfunction = weightfunction

    @tf_method()
    @overrides
    def build_log_likelihood(self):
        lmls = [child.build_log_likelihood() for child in self.children]
        return tf.reduce_sum(tf.pack(lmls, 0), 0)

    @tf_method()
    @overrides
    def build_prior_mean_var(self, test_points):
        """A weighted mean of the prior means and variances of the experts."""
        mu, var = zip(*[child.build_prior_mean_var(test_points)
                        for child in self.children])
        weight = self.weightfunction(self.children, test_points)
        mu, var, weight = map(lambda x: tf.pack(x, 0), (mu, var, weight))

        weight_total = tf.reduce_sum(weight, 0)
        joint_mu = tf.reduce_sum(weight * mu, 0) / weight_total
        joint_var = tf.reduce_sum(weight * var, 0) / weight_total
        return joint_mu, joint_var

    @tf_method()
    @overrides
    def build_posterior_mean_var(self, test_points):
        mu, var = zip(*[child.build_posterior_mean_var(test_points)
                        for child in self.children])
        weight = self.weightfunction(self.children, test_points)
        mu, var, weight = map(lambda x: tf.pack(x, 0), (mu, var, weight))

        prior_precision = 1 / self.build_prior_mean_var(test_points)[1]
        joint_var = 1 / (tf.reduce_sum(weight / var, 0)
                         + (1 - tf.reduce_sum(weight, 0)) * prior_precision)
        joint_mu = joint_var * tf.reduce_sum(weight * mean / var, 0)
        return joint_mu, joint_var

class PriorDivisorReduction(GPModel, ParamAttributes):
    """Divides by the prior in proportion to the weight.

    This is mostly a convenience cljkass for constructing a hierarchical
    rBCM. It delegates the `.build_log_likelihood()` and 
    `.build_prior_mean_var()` methods to the child expert, and corrects
    the posterior mean and variance with a prior division term.
    
    Attributes:
        child (GPModel): The expert to correct the opinion of.
        weightfunction (Callable[[Sequence[GPModel], tf.Tensor],
            Tuple[tf.Tensor]]): A function used to calculate the 
            weights of the experts.

    """
    def __init__(self, child, weightfunction):
        """Initialiser.

        Args:
            child (GPModel): The expert to correct the opinion of.
            weightfunction (Callable[[Sequence[GPModel], tf.Tensor],
                Tuple[tf.Tensor]]): A function used to calculate the 
                weights of the experts.

        """
        self.child = child
        self.weightfunction = weightfunction

    @tf_method()
    @overrides
    def build_log_likelihood(self):
        return child.build_log_likelihood() 

    @tf_method()
    @overrides
    def build_prior_mean_var(self, test_points):
        return self.child.build_prior_mean_var(test_points)

    @tf_method
    @overrides
    def build_posterior_mean_var(self, test_points):
        mu, var = self.child.build_posterior_mean_var(test_points)
        weight = self.weightfunction([self.child], test_points)
        prior_precision = 1 / self.build_prior_mean_var(test_points)[1]
        joint_var = 1 / (1 / var + (1 - weight) * prior_precision)
        joint_mu = joint_var * mu / var
        return joint_mu, joint_var

class DistributedrBCM(GPModel, ParamAttributes):
    """Acts as an rBCM, but distributes its computation."""
    def __init__(self, experts, architecture, clusterspec, 
            worker_job='worker', param_server_job='ps', master_job=None):
        """Initialiser.

        Args:
            experts (GPModel | Sequence[GPModel]): The experts to 
                combine the opinions of. If this is a `GPModel`, then it
                will be shallow-copied to fill out the architecture.
                If it is a sequence of `GPModel`s, then the architecture
                will have to have as many nodes in its final layer as
                there are in the sequence.
            architecture (Sequence[int]): The branching factors at each
                layer of the rBCM.
            clusterspec (tf.train.ClusterSpec): The cluster to
                distribute tasks over.
            worker_job (str): The job to assign computationally
                expensive tasks to.
            param_server_job (str): The job to assign paramaters to.
            master_job (str | None): The job to coordinate other jobs
                from. This is used to find the `tf_session_target`. 
                If `None`, the first task of `param_server_job`
                will be used as the session target. Otherwise, the 
                first task of the specified job is used.

        """
        # make sure there are enough experts
        num_experts = reduce(operator.mul, architecture, 1)
        if isinstance(experts, GPModel):
            experts = [experts.copy() for _ in range(num_experts)]
        else:
            assert len(experts) == num_experts

        
