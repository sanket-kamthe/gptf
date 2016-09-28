# -*- encoding: utf-8 -*-
"""Provides base classes for models of all kinds."""
from builtins import super, range
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
from scipy.optimize import OptimizeResult

from . import tfhacks, utils
from .params import Parameterized, ParamAttributes, DataHolder, autoflow
from .wrappedtf import tf_method

class Model(with_metaclass(ABCMeta, Parameterized)):
    """Base class for models. 
    
    Inheriting classes must define `.build_log_likelihood(self)`.

    `Param` and `Parameterized` objects that are children of the model
    can be used in the tensorflow expression. Children on the model are
    defined like so:

    >>> from overrides import overrides
    >>> from gptf import Param, ParamAttributes
    >>> class Example(Model, ParamAttributes):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.x = Param(1.)  # create new Param child
    ...
    ...     @tf_method()
    ...     @overrides
    ...     def build_log_likelihood(self, X, Y):
    ...         return 3 - self.x.tensor  # use Param in expression

    The `.optimize` method can be used to optimize the parameters of the
    model to minimise the likelihood. The loss function (the negative of
    the sum of the likelihood and any priors) is cached in the WrappedTF
    cache, and lazily recompiled when the cache is cleared, e.g. on 
    recompile.

    """
    @abstractmethod
    def build_log_likelihood(self, X, Y):
        """Builds the log likelihood of the model w.r.t. the data.

        Args:
            X (tf.Tensor): The training inputs.
            Y (tf.Tensor): The training outputs.

        Returns:
            (tf.Tensor): A tensor that, when run, calculates the log
            likelihood of the model.

        """
        NotImplemented

    @tf_method()
    def build_log_prior(self):
        NotImplemented

    @autoflow((tf.float64, [None, None]), (tf.float64, [None, None]))
    def compute_log_likelihood(self, X, Y):
        """Computes the likelihood of the model w.r.t. the data.
        
        Returns:
            (np.ndarray): The log likelihood of the model.

        """
        return self.build_log_likelihood(X, Y)

    @autoflow()
    def compute_log_prior(self):
        NotImplemented

    @tf_method(cache=False)
    def optimize(self, X, Y, method='L-BFGS-B', callback=None, 
            maxiter=1000, **kw):
        """Optimize the model by maximising the log likelihood.

        Maximises the sum of the log likelihood given X & Y and any 
        priors with respect to any free variables.

        Args:
            X (np.ndarray | tf.Tensor): The training inputs.
            Y (np.ndarray | tf.Tensor): The training outputs.
            method (tf.train.Optimizer | str): The means by which to
                optimise. If `method` is a string, it will be passed as
                the `method` argument to the initialiser of
                `tf.contrib.opt.ScipyOptimizerInterface`. Else, it
                will be treated as an instance of `tf.train.Optimizer`
                and its `.minimize()` method will be used as the training
                step.
            callback (Callable[[np.ndarray], ...]): A function that will
                be called at each optimization step with the current value
                of the variable vector (a vector constructed by flattening
                the free state of each free `Param` and then concatenating 
                them in the order the `Param`\ s are returned by `.params`.
            maxiter (int): The maximum number of iterations of the optimizer.
            **kw: Additional keyword arguments are passed through to the
                optimizer.

        Returns:
            (scipy.OptimizeResult) The result of the optimisation.

        Examples:
            Let's construct a very simple model for demonstration 
            purposes. It has two (scalar) parameters, `.a` and `.b`, 
            which are constrained to be positive, and its likelihood is
            `10 - a - b`, regardless of X and Y.

            >>> import numbers
            >>> import numpy as np
            >>> from overrides import overrides
            >>> from gptf import Param, ParamAttributes, transforms
            >>> class Example(Model, ParamAttributes):
            ...     def __init__(self, a, b):
            ...         assert isinstance(a, numbers.Number)
            ...         assert isinstance(b, numbers.Number)
            ...         super().__init__()
            ...         self.a = Param(a, transform=transforms.Exp(0.))
            ...         self.b = Param(b, transform=transforms.Exp(0.))
            ...     @tf_method()
            ...     @overrides
            ...     def build_log_likelihood(self, X, Y):
            ...         return 10. - self.a.tensor - self.b.tensor

            We won't care about the values of X and Y.

            >>> X = np.array(0.)
            >>> Y = np.array(0.)

            .. rubric:: TensorFlow optimizers

            We can optimise the parameters of the model using a TensorFlow
            optimizer like so:

            >>> m = Example(3., 4.)
            >>> opt = tf.train.GradientDescentOptimizer(learning_rate=1)
            >>> m.optimize(X, Y, opt)  # use None for X, Y
            message: 'Finished iterations.'
            success: True
                  x: array([..., ...])

            After the optimisation, both parameters are optimised
            towards 0, but are still positive. The constraints on the 
            parameters have been respected.

            >>> print("m.a: {:.3f}".format(np.asscalar(m.a.value)))
            m.a: 0.001
            >>> print("m.b: {:.3f}".format(np.asscalar(m.b.value)))
            m.b: 0.001

            If we fix a parameter, it is not optimized:
            
            >>> m.a = 5.
            >>> m.b = 1.
            >>> m.b.fixed = True
            >>> m.optimize(X, Y, opt)
            message: 'Finished iterations.'
            success: True
                  x: array([...])
            >>> print("m.a: {:.3f}".format(np.asscalar(m.a.value)))
            m.a: 0.001
            >>> print("m.b: {:.3f}".format(np.asscalar(m.b.value)))
            m.b: 1.000

            .. rubric:: SciPy optimizers

            We can optimise the parameters of the model using a SciPy
            optimizer by provided a string value for `method`:

            >>> m = Example(3., 4.)
            >>> m.optimize(X, Y, 'L-BFGS-B', disp=False, ftol=.0001)
            message: 'SciPy optimizer completed successfully.'
            success: True
                  x: array([..., ...])

            As for TensorFlow optimizers, after the optimisation both 
            parameters are optimised towards 0, but are still positive. 
            The constraints on the parameters have been respected.

            >>> print("m.a: {:.3f}".format(np.asscalar(m.a.value)))
            m.a: 0.000
            >>> print("m.b: {:.3f}".format(np.asscalar(m.b.value)))
            m.b: 0.000

            If we fix a parameter, it is not optimized:

            >>> m.a = 5.
            >>> m.b = 1.
            >>> m.b.fixed = True
            >>> m.optimize(X, Y, 'L-BFGS-B', disp=False, ftol=.0001)
            message: 'SciPy optimizer completed successfully.'
            success: True
                  x: array([...])
            >>> print("m.a: {:.3f}".format(np.asscalar(m.a.value)))
            m.a: 0.000
            >>> print("m.b: {:.3f}".format(np.asscalar(m.b.value)))
            m.b: 1.000

            .. rubric:: Miscellaneous

            Optimisation still works, even with weird device contexts and
            session targets.

            >>> # set up a distributed execution environment
            >>> clusterdict = \\
            ...     { 'worker': ['localhost:2226']
            ...     , 'master': ['localhost:2227']
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

            TensorFlow:

            >>> m.a = 4.5
            >>> m.optimize(X, Y, opt)
            message: 'Finished iterations.'
            success: True
                  x: array([...])
            >>> print("m.a: {:.3f}".format(np.asscalar(m.a.value)))
            m.a: 0.001
            >>> print("m.b: {:.3f}".format(np.asscalar(m.b.value)))
            m.b: 1.000
            
            SciPy:

            >>> m.a = 4.5
            >>> m.optimize(X, Y, 'L-BFGS-B', disp=False, ftol=.0001)
            message: 'SciPy optimizer completed successfully.'
            success: True
                  x: array([...])
            >>> print("m.a: {:.3f}".format(np.asscalar(m.a.value)))
            m.a: 0.001
            >>> print("m.b: {:.3f}".format(np.asscalar(m.b.value)))
            m.b: 1.000

        """
        X_key = X if isinstance(X, tf.Tensor) else None
        Y_key = Y if isinstance(Y, tf.Tensor) else None
        key = ("_Model__loss", X_key, Y_key)
        if key not in self.cache:
            X_tensor = (X if isinstance(X, tf.Tensor) else
                        tf.placeholder(tf.as_dtype(X.dtype)))
            Y_tensor = (Y if isinstance(Y, tf.Tensor) else
                        tf.placeholder(tf.as_dtype(Y.dtype)))
            self.cache[key] = (self._compile_loss(X_tensor, Y_tensor),
                               X_tensor, Y_tensor)
        loss, X_tensor, Y_tensor = self.cache[key]

        feed_dict = self.feed_dict
        if not isinstance(X, tf.Tensor): feed_dict[X_tensor] = X
        if not isinstance(Y, tf.Tensor): feed_dict[Y_tensor] = Y

        variables = [p.free_state for p in self.params if not p.fixed]
        variables = utils.unique(variables)
        free_state = tf.concat(0, [tf.reshape(v, [-1]) for v in variables])

        with self.get_session() as sess:
            try:
                if type(method) is str:
                    success_msg = "SciPy optimizer completed successfully."
                    options = {'maxiter': maxiter, 'disp': True}
                    options.update(kw)
                    optimizer = ScipyOptimizerInterface(
                        loss, var_list=variables, method=method, 
                        options=options
                    )
                    optimizer.minimize(self.get_session(), feed_dict, 
                            step_callback=callback)
                else:
                    # treat method as TensorFlow optimizer.
                    success_msg = "Finished iterations."
                    opt_step = method.minimize(loss, var_list=variables, **kw)
                    for _ in range(maxiter):
                        sess.run(opt_step, feed_dict=feed_dict)
                        if callback is not None:
                            callback(sess.run(free_state))
            except KeyboardInterrupt:
                return OptimizeResult\
                        ( x=sess.run(free_state)
                        , success=False
                        , message="Keyboard interrupt."
                        )

            return OptimizeResult\
                    ( x=sess.run(free_state)
                    , success=True
                    , message=success_msg
                    )

    def _compile_loss(self, X, Y):
        return -self.build_log_likelihood(X, Y)

class GPModel(Model):
    """A base class for Guassian Process models.

    A Gaussian process model is a model of the form

    .. math::

        θ ~ p(θ)

        f ~ GP(m(x), k(x, x'; θ))

        F = f(X)

        Y|F ~ p(Y|F)

    Adds functionality to compile various predictions. Inheriting 
    classes must define `.build_predict()`, which is then used by this 
    class's methods to provide various predictions. The mean and 
    variance are pushed through the likelihood to obtain the means and 
    variances of held out data.

    """
    @abstractmethod
    def build_prior_mean_var(self, test_points, num_latent, full_cov=False):
        """Builds an op for the mean and variance of the prior(s).
        
        In the returned tensors, the last index should always be the 
        latent function index.

        Args:
            test_points (tf.Tensor): The points from the sample
                space for which to predict means and variances
                of the prior distribution(s). The shape should be
                `[m, point_dims]`.
            num_latent (tf.int32): The number of latent functions of 
                the GP.
            full_cov (bool): If `False`, return an array of variances
                at the test points. If `True`, return the full
                covariance matrix of the posterior distribution.
            
        Returns:
            (tf.Tensor, tf.Tensor): A tensor that calculates the mean
            at the test points with shape `[m, num_latent]`, a tensor
            that calculates either the variances at the test points 
            (shape `[m, num_latent]`) or the full covariance matrix 
            (shape `[m, m, num_latent]`).
            Both tensors have the same dtype.

        """
        NotImplemented

    @abstractmethod
    def build_posterior_mean_var(self, X, Y, test_points, full_cov=False):
        """Builds an op for the mean and variance of the posterior(s).

        In the returned tensors, the last index should always be the 
        latent function index.

        Args:
            X (tf.Tensor): The training inputs, shape `[n, point_dims]`
            Y (tf.Tensor): The training outputs, shape `[n, num_latent]`
            test_points (tf.Tensor): The points from the sample
                space for which to predict means and variances
                of the posterior distribution(s), shape 
                `[m, point_dims]`.
            full_cov (bool): If `False`, return an array of variances
                at the test points. If `True`, return the full
                covariance matrix of the posterior distribution.

        Returns:
            (tf.Tensor, tf.Tensor): A tensor that calculates the mean
            at the test points with shape `[m, num_latent]`, a tensor
            that calculates either the variances at the test points 
            (shape `[m, num_latent]`) or the full covariance matrix 
            (shape `[m, m, num_latent]`).
            Both tensors have the same dtype.

        """
        NotImplemented

    @autoflow((tf.float64, [None, None]), (tf.int32, []))
    def compute_prior_mean_var(self, test_points, num_latent):
        """Computes the means and variances of the prior(s).

        This is just an autoflowed version of 
        `.build_prior_mean_var(test_points, num_latent)`.

        Args:
            test_points (np.ndarray): The points from the sample
                space for which to predict means and variances
                of the prior distribution(s). The shape should be
                `[m, point_dims]`.
            num_latent (int): The number of latent functions of the GP.
            
        Returns:
            (np.ndarray, np.ndarray): the mean at the test points 
            (shape `[m, num_latent]`), the variances at the test 
            points (shape `[m, num_latent]`).

        """
        return self.build_prior_mean_var(test_points, num_latent, False)

    @autoflow((tf.float64, [None, None]), (tf.int32, []))
    def compute_prior_mean_cov(self, test_points, num_latent):
        """Computes the means and full covariance matrices.

        This is just an autoflowed version of 
        `.build_prior_mean_var(test_points, num_latent, True)`.

        Args:
            test_points (np.ndarray): The points from the sample
                space for which to predict means and variances
                of the prior distribution(s). The shape should be
                `[m, point_dims]`.
            num_latent (int): The number of latent functions of the GP.

        Returns:
            (np.ndarray, np.ndarray): The means at the test points
            (shape `[m, num_latent]`), the full covariance 
            matri(x|ces) for the prior distribution(s) (shape
            `[m, m, num_latent]`.

        """
        return self.build_prior_mean_var(test_points, num_latent, True)

    @autoflow((tf.float64, [None, None]), (tf.int32, []), (tf.int32, []))
    def compute_prior_samples(self, test_points, num_latent, num_samples): 
        """Computes samples from the prior distribution(s).

        Args:
            test_points (np.ndarray): The points from the sample
                space for which to predict means and variances
                of the posterior distribution(s), shape 
                `[m, point_dims]`.
            num_latent (int): The number of latent functions of the GP.
            num_samples (int): The number of samples to take.

        Returns:
            (np.ndarray): An array of samples from the prior
            distributions, with shape `[num_samples, m, num_latent]`

        Examples:
            For testing purposes, we create an example model whose
            likelihood is always `0` and whose `.build_predict()`
            returns mean `0` and variance `1` for every test point,
            or an independent covariance matrix.

            >>> from overrides import overrides
            >>> from gptf import ParamAttributes, tfhacks
            >>> class Example(GPModel, ParamAttributes):
            ...     def __init__(self, dtype):
            ...         super().__init__()
            ...         self.dtype = dtype
            ...     @property
            ...     def dtype(self):
            ...         return self._dtype
            ...     @dtype.setter
            ...     def dtype(self, value):
            ...         self.clear_cache()
            ...         self._dtype = value
            ...     @tf_method()
            ...     @overrides
            ...     def build_log_likelihood(self):
            ...         NotImplemented
            ...     @tf_method()
            ...     @overrides
            ...     def build_prior_mean_var\\
            ...             (self, test_points, num_latent, full_cov=False):
            ...         n = tf.shape(test_points)[0]
            ...         mu = tf.zeros([n, 1], self.dtype)
            ...         mu = tf.tile(mu, (1, num_latent))
            ...         if full_cov:
            ...             var = tf.expand_dims(tfhacks.eye(n, self.dtype), 2)
            ...             var = tf.tile(var, (1, 1, num_latent))
            ...         else:
            ...             var = tf.ones([n, 1], self.dtype)
            ...             var = tf.tile(var, (1, num_latent))
            ...         return mu, var
            ...     @tf_method()
            ...     @overrides
            ...     def build_posterior_mean_var\\
            ...             (self, X, Y, test_points, full_cov=False):
            ...         NotImplemented
            >>> m = Example(tf.float64)  # ignore the likelihood
            >>> test_points = np.array([[0.], [1.], [2.], [3.]])

            The shape of the returned array is `(a, b, c)`, where `a`
            is the number of samples, `b` is the number of test points
            and `c` is the number of latent functions.

            >>> samples = m.compute_prior_samples(test_points, 1, 2)
            >>> samples.shape
            (2, 4, 1)

            `.compute_prior_samples()` respects the dtype of the tensors
            returned by `.build_predict()`.

            >>> samples.dtype
            dtype('float64')
            >>> m.dtype = tf.float32
            >>> samples = m.compute_prior_samples(test_points, 1, 2)
            >>> samples.dtype
            dtype('float32')
            
        """
        mu, var = self.build_prior_mean_var(test_points, num_latent, True)
        jitter = tfhacks.eye(tf.shape(mu)[0], var.dtype) * 1e-06
        L = tf.batch_cholesky(tf.transpose(var, (2, 0, 1)) + jitter)
        V_shape = [tf.shape(L)[0], tf.shape(L)[1], num_samples]
        V = tf.random_normal(V_shape, dtype=L.dtype)
        samples = tf.expand_dims(tf.transpose(mu), -1) + tf.batch_matmul(L, V)
        return tf.transpose(samples)
        
    @autoflow((tf.float64, [None, None]), (tf.float64, [None, None]),
              (tf.float64, [None, None]))
    def compute_posterior_mean_var(self, X, Y, test_points):
        """Computes the means and variances of the posterior(s).

        This is just an autoflowed version of 
        `.build_posterior_mean_var(X, Y, test_points)`.

        Args:
            X (np.ndarray): The training inputs, shape `[n, point_dims]`
            Y (np.ndarray): The training outputs, shape `[n, num_latent]`
            test_points (np.ndarray): The points from the sample
                space for which to predict means and variances
                of the posterior distribution(s), shape 
                `[m, point_dims]`.

        Returns:
            (np.ndarray, np.ndarray): The means at the test points
            (shape `[m, num_latent]`), the variances at the test points
            (shape `[m, num_latent]`).

        """
        return self.build_posterior_mean_var(X, Y, test_points, full_cov=False)

    @autoflow((tf.float64, [None, None]), (tf.float64, [None, None]),
              (tf.float64, [None, None]))
    def compute_posterior_mean_cov(self, X, Y, test_points):
        """Computes the means and full covariance matrices.

        This is just an autoflowed version of 
        `.build_predict(X, Y, test_points, full_cov=True)`.

        Args:
            X (np.ndarray): The training inputs, shape `[n, point_dims]`
            Y (np.ndarray): The training outputs, shape `[n, num_latent]`
            test_points (np.ndarray): The points from the sample
                space for which to predict means and variances
                of the posterior distribution(s), shape 
                `[m, point_dims]`.

        Returns:
            (np.ndarray, np.ndarray): The means at the test points
            (shape `[m, num_latent]`), the full covriance 
            matri(x|ces) for the posterior distribution(s)
            (shape `[m, m, num_latent]`).

        """
        return self.build_posterior_mean_var(X, Y, test_points, full_cov=True)

    @autoflow((tf.float64, [None, None]), (tf.float64, [None, None]), 
              (tf.float64, [None, None]), (tf.int32, []))
    def compute_posterior_samples(self, X, Y, test_points, num_samples): 
        """Computes samples from the posterior distribution(s).

        Args:
            X (np.ndarray): The training inputs, shape `[n, point_dims]`
            Y (np.ndarray): The training outputs, shape `[n, num_latent]`
            test_points (np.ndarray): The points from the sample
                space for which to predict means and variances
                of the posterior distribution(s), shape 
                `[m, point_dims]`.
            num_samples (int): The number of samples to take.

        Returns:
            (np.ndarray): An array of samples from the posterior
            distributions, with shape `[num_samples, m, num_latent]`

        Examples:
            For testing purposes, we create an example model whose
            likelihood is always `0` and whose `.build_predict()`
            returns mean `0` and variance `1` for every test point,
            or an independent covariance matrix.

            >>> from overrides import overrides
            >>> from gptf import ParamAttributes, tfhacks
            >>> class Example(GPModel, ParamAttributes):
            ...     def __init__(self, dtype):
            ...         super().__init__()
            ...         self.dtype = dtype
            ...     @property
            ...     def dtype(self):
            ...         return self._dtype
            ...     @dtype.setter
            ...     def dtype(self, value):
            ...         self.clear_cache()
            ...         self._dtype = value
            ...     @tf_method()
            ...     @overrides
            ...     def build_log_likelihood(self):
            ...         NotImplemented
            ...     @tf_method()
            ...     @overrides
            ...     def build_prior_mean_var\\
            ...             (self, test_points, num_latent, full_cov=False):
            ...         NotImplemented
            ...     @tf_method()
            ...     @overrides
            ...     def build_posterior_mean_var\\
            ...             (self, X, Y, test_points, full_cov=False):
            ...         n = tf.shape(test_points)[0]
            ...         num_latent = tf.shape(Y)[1]
            ...         mu = tf.zeros([n, 1], self.dtype)
            ...         mu = tf.tile(mu, (1, num_latent))
            ...         if full_cov:
            ...             var = tf.expand_dims(tfhacks.eye(n, self.dtype), 2)
            ...             var = tf.tile(var, (1, 1, num_latent))
            ...         else:
            ...             var = tf.ones([n, 1], self.dtype)
            ...             var = tf.tile(var, (1, num_latent))
            ...         return mu, var
            >>> m = Example(tf.float64)
            >>> X = np.array([[.5]])
            >>> Y = np.array([[.3]])
            >>> test_points = np.array([[0.], [1.], [2.], [3.]])

            The shape of the returned array is `(a, b, c)`, where `a`
            is the number of samples, `b` is the number of test points
            and `c` is the number of latent functions.

            >>> samples = m.compute_posterior_samples(X, Y, test_points, 2)
            >>> samples.shape
            (2, 4, 1)

            `.compute_posterior_samples()` respects the dtype of the tensors
            returned by `.build_predict()`.

            >>> samples.dtype
            dtype('float64')
            >>> m.dtype = tf.float32
            >>> samples = m.compute_posterior_samples(X, Y, test_points, 2)
            >>> samples.dtype
            dtype('float32')
            
        """
        mu, var = self.build_posterior_mean_var(X, Y, test_points, True)
        jitter = tfhacks.eye(tf.shape(mu)[0], var.dtype) * 1e-06
        L = tf.batch_cholesky(tf.transpose(var, (2, 0, 1)) + jitter)
        V_shape = [tf.shape(L)[0], tf.shape(L)[1], num_samples]
        V = tf.random_normal(V_shape, dtype=L.dtype)
        samples = tf.expand_dims(tf.transpose(mu), -1) + tf.batch_matmul(L, V)
        return tf.transpose(samples)
        #samples = []
        #for i in range(self.num_latent_functions):
        #    L = tf.cholesky(var[:, :, i] + jitter)
        #    V = tf.random_normal([tf.shape(L)[0], num_samples], dtype=L.dtype)
        #    samples.append(mu[:, i:i + 1] + tf.matmul(L, V))  # broadcast
        #return tf.transpose(tf.pack(samples))

    @autoflow((tf.float64, [None, None]))
    def predict_y(self, test_points):
        """Computes the mean and variance of held-out data."""
        NotImplemented

    @autoflow((tf.float64, [None, None]), (tf.float64, [None, None]))
    def predict_density(self, test_points, test_values):
        """Computes the (log) density of the test values at the test points."""
        NotImplemented
