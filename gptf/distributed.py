# -*- encoding: utf-8 -*-
"""Provides classes for Robust Bayesian Committee Machines."""
#TODO: Work out the maths for a non-zero prior mean function?
from __future__ import division
from builtins import super, zip, map, range
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod
from functools import reduce
import operator
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

from overrides import overrides
import numpy as np
import tensorflow as tf

from gptf import GPModel, Parameterized, ParamList, ParamAttributes, tf_method

def cao_fleet_weights(experts, X, Y, points):
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
        X (tf.Tensor): the training inputs, shape `[n, point_dims]`.
        Y (tf.Tensor): the training outputs, shape `[n, num_latent]`.
        points (tf.Tensor): the points at which to calculate the 
            weights of the model, shape `[m, point_dims]`.

    Returns:
        (Tuple[tf.Tensor]): The weights of the experts at the
        points. Each tensor has shape `(num_points, num_latent)`.

    """
    def weight(expert, X, Y):
        num_latent = tf.shape(Y)[1]
        prior_var = expert.build_prior_mean_var(points, num_latent)[1]
        post_var = expert.build_posterior_mean_var(X, Y, points)[1]
        weight = .5 * (tf.log(prior_var) - tf.log(post_var))
        # return very small value instead of zero
        return tf.maximum(weight, np.finfo(np.float64).eps)
    M = len(experts)
    chunks = zip(experts, tf.split(0, M, X), tf.split(0, M, Y))
    return tuple(weight(*chunk) for chunk in chunks)

def equal_weights(experts, X, Y, points):
    """Gives each expert an equal weight.

    .. math::
        \\forall k \in 0..M,\ β_k = 1 / M

    where :math:`β_k` is the weight of the :math:`k`th expert at
    every point and :math:`M` is the number of experts.

    The dtype returned matches the dtype of `points`.
    
    Args:
        experts (Sequence[GPModel]): the experts to calculate
            the weights of.
        X (tf.Tensor): the training inputs, shape `[n, point_dims]`.
        Y (tf.Tensor): the training outputs, shape `[n, num_latent]`.
        points (tf.Tensor): the points at which to calculate the 
            weights of the model, shape `[m, point_dims]`.

    Returns:
        (Tuple[tf.Tensor]): The weights of the experts at the
        points. Each tensor has shape `(num_points, num_latent)`.

    """
    beta = tf.constant(1 / len(experts), dtype=points.dtype)
    beta = tf.fill((tf.shape(points)[0], 1), beta)
    return tuple(beta for _ in experts)

def ones_weights(experts, X, Y, points):
    """All weights are 1.

    The dtype returned matches the dtype of `points`.
    
    Args:
        experts (Sequence[GPModel]): the experts to calculate
            the weights of.
        X (tf.Tensor): the training inputs, shape `[n, point_dims]`.
        Y (tf.Tensor): the training outputs, shape `[n, num_latent]`.
        points (tf.Tensor): the points at which to calculate the 
            weights of the model, shape `[m, point_dims]`.

    Returns:
        (Tuple[tf.Tensor]): The weights of the experts at the
        points. Each tensor has shape `(num_points, num_latent)`.

    """
    ones = tf.ones((tf.shape(points)[0], 1), dtype=points.dtype)
    return tuple(ones for _ in experts)

class Reduction(GPModel, ParamList):
    """Common code for distributed GP reductions."""
    @tf_method()
    @overrides
    def build_log_likelihood(self, X, Y):
        """The sum of the log likelihoods of the children."""
        chunks = self._get_chunks(X, Y)
        lmls = [child.build_log_likelihood(Xk, Yk) for child, Xk, Yk in chunks]
        return tf.reduce_sum(tf.pack(lmls, 0), 0)

    @tf_method()
    @overrides
    def build_prior_mean_var(self, test_points, num_latent, full_cov=False):
        """The arithetic mean of the prior mean / variance of the chilren."""
        if full_cov:
            raise NotImplementedError("Full covariance matrix not yet"
                    "implemented for distributed GPs.")
        mu, var = zip(*[child.build_prior_mean_var(
                            test_points, num_latent, full_cov
                        ) for child in self.children])
        M = float(len(self.children))
        mu, var = tf.pack(mu, 0), tf.pack(var, 0)
        return tf.reduce_sum(mu, 0) / M, tf.reduce_sum(var, 0) / M

    @tf_method()
    def _get_chunks(self, X, Y):
        """Seperates X and Y into chunks, pairing each chunk with a child."""
        M = len(self.children)
        return zip(self.children, tf.split(0, M, X), tf.split(0, M, Y))

class PoEReduction(Reduction):
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
    def build_posterior_mean_var(self, X, Y, test_points, full_cov=False):
        if full_cov:
            raise NotImplementedError("Full covariance matrix not yet"
                    "implemented for distributed GPs.")
        chunks = self._get_chunks(X, Y)
        mu, var = zip(*[child.build_posterior_mean_var(
                            Xk, Yk, test_points, full_cov
                        ) for child, Xk, Yk in chunks])
        mu, var = tf.pack(mu, 0), tf.pack(var, 0)
        joint_var = 1 / tf.reduce_sum(1 / var, 0)
        joint_mu = joint_var * tf.reduce_sum(mu / var, 0)
        return joint_mu, joint_var

class gPoEReduction(Reduction):
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
        weightfunction (Callable[[Sequence[GPModel], tf.Tensor,
            tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]): 
            A function used to calculate the weight of the experts.

    For other attributes, see `GPModel` and `ParamList`.

    """
    def __init__(self, children, weightfunction):
        """Initialiser.

        Args:
            children (Sequence[GPModel]): The experts to combine the
                opinions of.
            weightfunction (Callable[[Sequence[GPModel], tf.Tensor,
                tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]): 
                A function used to calculate the weight of the experts.

        """
        super().__init__()
        self.extend(children)
        self.weightfunction = weightfunction
    #TODO: better weight functions that don't need X and Y
    #@tf_method()
    #@overrides
    #def build_prior_mean_var(self, test_points, num_latent, full_cov=False):
    #    """A weighted mean of the prior means and variances of the experts."""
    #    if full_cov:
    #        raise NotImplementedError("Full covariance matrix not yet"
    #                "implemented for distributed GPs.")
    #    mu, var = zip(*[child.build_prior_mean_var(
    #                        test_points, num_latent, full_cov
    #                    ) for child in self.children])
    #    weight = self.weightfunction(self.children, test_points)
    #    mu, var, weight = map(lambda x: tf.pack(x, 0), (mu, var, weight))

    #    w_total = tf.reduce_sum(weight, 0)
    #    joint_mu = tf.reduce_sum(weight * mu, 0) / w_total
    #    joint_var = tf.reduce_sum(weight * var, 0) / w_total
    #    return joint_mu, joint_var

    @tf_method()
    @overrides
    def build_posterior_mean_var(self, X, Y, test_points, full_cov=False):
        if full_cov:
            raise NotImplementedError("Full covariance matrix not yet"
                    "implemented for distributed GPs.")
        chunks = self._get_chunks(X, Y)
        mu, var = zip(*[child.build_posterior_mean_var(
                            Xk, Yk, test_points, full_cov
                        ) for child, Xk, Yk in chunks])
        weight = self.weightfunction(self.children, X, Y, test_points)
        mu, var, weight = map(lambda x: tf.pack(x, 0), (mu, var, weight))

        joint_var = 1 / tf.reduce_sum(weight / var, 0)
        joint_mu = joint_var * tf.reduce_sum(weight * mu / var, 0)
        return joint_mu, joint_var

class BCMReduction(Reduction):
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
    def build_posterior_mean_var(self, X, Y, test_points, full_cov=False):
        if full_cov:
            raise NotImplementedError("Full covariance matrix not yet"
                    "implemented for distributed GPs.")
        chunks = self._get_chunks(X, Y)
        mu, var = zip(*[child.build_posterior_mean_var(
                            Xk, Yk, test_points, full_cov
                        ) for child, Xk, Yk in chunks])
        mu, var = tf.pack(mu, 0), tf.pack(var, 0)

        num_latent = tf.shape(Y)[1]
        prior_var = self.build_prior_mean_var(test_points, num_latent)[1]
        joint_var = 1/ (tf.reduce_sum(1 / var, 0)
                        + (1 - len(self.children)) / prior_var)
        joint_mu = joint_var * tf.reduce_sum(mu / var, 0)
        return joint_mu, joint_var

class rBCMReduction(Reduction):
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
        weightfunction (Callable[[Sequence[GPModel], tf.Tensor,
            tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]): 
            A function used to calculate the weight of the experts.

    For other attributes, see `GPModel` and `ParamList`.

    """
    def __init__(self, children, weightfunction):
        """Initialiser.

        Args:
            children (Sequence[GPModel]): The experts to combine the
                opinions of.
            weightfunction (Callable[[Sequence[GPModel], tf.Tensor,
                tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]): 
                A function used to calculate the weight of the experts.

        """
        super().__init__()
        self.extend(children)
        self.weightfunction = weightfunction

    #TODO: better weight functions that don't need X and Y
    #@tf_method()
    #@overrides
    #def build_prior_mean_var(self, test_points, num_latent, full_cov=False):
    #    """A weighted mean of the prior means and variances of the experts."""
    #    if full_cov:
    #        raise NotImplementedError("Full covariance matrix not yet"
    #                "implemented for distributed GPs.")
    #    mu, var = zip(*[child.build_prior_mean_var(
    #                        test_points, num_latent, full_cov
    #                    ) for child in self.children])
    #    weight = self.weightfunction(self.children, test_points)
    #    mu, var, weight = map(lambda x: tf.pack(x, 0), (mu, var, weight))

    #    w_total = tf.reduce_sum(weight, 0)
    #    joint_mu = tf.reduce_sum(weight * mu, 0) / w_total
    #    joint_var = tf.reduce_sum(weight * var, 0) / w_total
    #    return joint_mu, joint_var

    @tf_method()
    @overrides
    def build_posterior_mean_var(self, X, Y, test_points, full_cov=False):
        if full_cov:
            raise NotImplementedError("Full covariance matrix not yet"
                    "implemented for distributed GPs.")
        chunks = self._get_chunks(X, Y)
        mu, var = zip(*[child.build_posterior_mean_var(
                            Xk, Yk, test_points, full_cov
                        ) for child, Xk, Yk in chunks])
        weight = self.weightfunction(self.children, X, Y, test_points)
        mu, var, weight = map(lambda x: tf.pack(x, 0), (mu, var, weight))

        num_latent = tf.shape(Y)[1]
        prior_var = self.build_prior_mean_var(test_points, num_latent)[1]
        joint_var = 1 / (tf.reduce_sum(weight / var, 0)
                         + (1 - tf.reduce_sum(weight, 0)) / prior_var)
        joint_mu = joint_var * tf.reduce_sum(weight * mu / var, 0)
        return joint_mu, joint_var

class PriorDivisorReduction(GPModel, ParamAttributes):
    """Divides by the prior in proportion to the weight.

    This is mostly a convenience cljkass for constructing a hierarchical
    rBCM. It delegates the `.build_log_likelihood()` and 
    `.build_prior_mean_var()` methods to the child expert, and corrects
    the posterior mean and variance with a prior division term.
    
    Attributes:
        child (GPModel): The expert to correct the opinion of.
        weightfunction (Callable[[Sequence[GPModel], tf.Tensor,
            tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]): 
            A function used to calculate the weight of the expert.

    """
    def __init__(self, child, weightfunction):
        """Initialiser.

        Args:
            child (GPModel): The expert to correct the opinion of.
            weightfunction (Callable[[Sequence[GPModel], tf.Tensor,
                tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]): 
                A function used to calculate the weight of the expert.

        """
        super().__init__()
        self.child = child
        self.weightfunction = weightfunction

    @tf_method()
    @overrides
    def build_log_likelihood(self, X, Y):
        return child.build_log_likelihood(X, Y) 

    @tf_method()
    @overrides
    def build_prior_mean_var(self, test_points, num_latent, full_cov=False):
        return self.child.build_prior_mean_var(
                test_points, num_latent, full_cov)

    @tf_method()
    @overrides
    def build_posterior_mean_var(self, X, Y, test_points, full_cov=False):
        if full_cov:
            raise NotImplementedError("Full covariance matrix not yet"
                    "implemented for distributed GPs.")
        mu, var = self.child.build_posterior_mean_var(
                X, Y, test_points, full_cov
        )
        weight = self.weightfunction([self.child], X, Y, test_points)[0]

        num_latent = tf.shape(Y)[1]
        prior_var = self.build_prior_mean_var(test_points, num_latent)[1]
        joint_var = 1 / (1 / var + (1 - weight) / prior_var)
        joint_mu = joint_var * mu / var
        return joint_mu, joint_var

def chunks(n, iterable, padvalue=None):
    """Iterator for equal-sized chunks of an iterable.

    Args:
        n (int): The size of each chunk.
        iterable (Iterable): The iterable to seperate.
        padvalue: A value to pad the last chunk with if `n` does not
            evenly divide len(list).

    Examples:
        >>> for chunk in chunks(3, 'abcdefg', 'x'):
        ...     print(chunk)
        ('a', 'b', 'c')
        ('d', 'e', 'f')
        ('g', 'x', 'x')
    
    """
    return zip_longest(*((iter(iterable),) * n), fillvalue=padvalue)

def tree_rBCM(experts, weightfunction, architecture):
    """An rBCM, expressed as a tree of reductions.

    Args:
        experts (GPModel | Sequence[GPModel]): The experts to 
            combine the opinions of. If this is a `GPModel`, then it
            will be shallow-copied to fill out the architecture.
            If it is a sequence of `GPModel`s, then the architecture
            will have to have as many nodes in its final layer as
            there are in the sequence, i.e.

                len(experts) == reduce(operator.mul,architecture,1)

        weightfunction (Callable[[Sequence[GPModel], tf.Tensor,
            tf.Tensor, tf.Tensor], Tuple[tf.Tensor]]): 
            A function used to calculate the weight of the experts.
        architecture (Sequence[int]): The branching factors at each
            layer of the rBCM.

    """
    # make sure there are enough experts
    assert len(architecture) > 0
    num_experts = reduce(operator.mul, architecture, 1)
    if isinstance(experts, GPModel):
        experts = (experts,) * num_experts
    else:
        assert len(experts) == num_experts

    chunksize = architecture.pop(-1)
    layer = [gPoEReduction(c, weightfunction)
             for c in chunks(chunksize, experts)]

    for chunksize in reversed(architecture):
        layer = [PoEReduction(c) for c in chunks(chunksize, layer)]
            
    assert len(layer) == 1
    def total_weight(_, X, Y, test_points):
        weights = weightfunction(experts, X, Y, test_points)
        return [tf.reduce_sum(tf.pack(weights, 0), 0)]

    return PriorDivisorReduction(layer[0], total_weight)

def distributed_tree_BCM(experts, clusterspec, worker_job='worker', 
        param_server_job='ps', target_job=None, target_protocol='grpc'):
    """Constructs a TreerBCM and distributes its computations.

    Preconditions:
        if `experts` is a sequence, the number of worker tasks must
        evenly divide `len(experts)`.
    
    If `experts` is a sequence, the architecture of the `TreerBCM` will
    be `[n, m]`, where `n` is the number of worker tasks and 
    `m = len(experts) / n`. Each `gPoEReduction` will be pinned to a 
    different worker task.
    
    If `experts` is a `GPModel`, the architecture of the `TreerBCM` will 
    be `[n]`, where `n` is the number of worker tasks. `experts` will be
    copied to fill out the architecture, with each copy being pinned to 
    a different worker task.

    `Param`s will be pinned to parameter server tasks in a round-robin
    fashion, based on the order they appear in `TreerBCM.params`.

    Additionally, the `.tf_graph` of the `TreerBCM` will be set to a
    new graph and the `.tf_session_target` will be set to either the
    first task of the target job or the first task of the parameter
    server job.

    Args:
        experts (GPModel | Sequence[GPModel]): The experts to 
            combine the opinions of. If this is a `GPModel`, then it
            will be shallow-copied to fill out the architecture.
            If it is a sequence of `GPModel`s, then the architecture
            will have as many nodes in its final layer as
            there are in the sequence.
        clusterspec (tf.train.ClusterSpec): The cluster to
            distribute tasks over.
        worker_job (str): The job to assign computationally
            expensive tasks to.
        param_server_job (str): The job to assign paramaters to.
        target_job (str | None): The job to coordinate other jobs
            from. This is used to find the `tf_session_target`. 
            If `None`, the first task of `param_server_job`
            will be used as the session target. Otherwise, the 
            first task of the specified job is used.
        target_protocol (str): The protocol to use when connecting to
            the target server. Defaults to 'grpc'.

    """
    num_worker_tasks = len(clusterspec.job_tasks(worker_job))
    if isinstance(experts, GPModel):
        architecture = [num_worker_tasks]
    else:
        assert len(experts) % num_worker_tasks == 0
        architecture = [num_worker_tasks, len(experts) / num_worker_tasks]

    rBCM = tree_rBCM(experts, architecture)

    rBCM.tf_graph = tf.Graph()
    
    target = (clusterspec.job_tasks(target_job)[0] if target_job 
              else clusterspec.job_tasks(parameter_server_job)[0])
    rBCM.tf_session_target = '{}://{}'.format(target_protocol, target)

    for index, child in enumerate(rBCM.child.children):
        child.tf_device = tf.DeviceSpec(job=worker_job, task=index)

    for index, param in enumerate(rBCM.params):
        i = index % len(clusterspec.job_tasks(param_server_job))
        param.tf_device = tf.DeviceSpec(job=param_server_job, task=i)

    return rBCM
