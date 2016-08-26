"""Provides base classes for models of all kinds."""
from builtins import super, range
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod

import tensorflow as tf
from tensorflow.contrib.opt import ScipyOptimizerInterface
from scipy.optimize import OptimizeResult

from .params import Parameterized
from .wrappedtf import WrappedTF


class Model(with_metaclass(ABCMeta, Parameterized)):
    """Base class for models. 
    
    Inheriting classes must define `.build_log_likelihood(self)`.

    """
    @abstractmethod
    def build_log_likelihood(self):
        """Builds the log likelihood of the model w.r.t. the data.

        Returns:
            (tf.Tensor): A tensor that, when run, calculates the log
            likelihood of the model.

        """
        NotImplemented

    @Parameterized.autoflow()
    def compute_log_likelihood(self):
        """Computes the likelihood of the model w.r.t. the data.
        
        Returns:
            (np.ndarray): The log likelihood of the model.

        """
        return self.build_log_likelihood()

    @Parameterized.autoflow()
    def compute_log_prior(self):
        NotImplemented

    def optimize(self, method='L-BFGS-B', callback=None, maxiter=1000, **kw):
        """Optimize the model by maximising the log likelihood.

        Maximises the sum of the log likelihood and any priors with 
        respect to any free variables.

        Args:
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
                them in the order the `Param`s are returned by `.params`.
            maxiter (int): The maximum number of iterations of the optimizer.
            **kw: Additional keyword arguments are passed through to the
                optimizer.

        Returns:
            (scipy.OptimizeResult) The result of the optimisation.

        Examples:
            Let's construct a very simple model for demonstration 
            purposes. It has two (scalar) parameters, `.a` and `.b`, 
            which are constrained to be positive, and its likelihood is
            `10 - a - b`.
            >>> import numbers
            >>> import numpy as np
            >>> from overrides import overrides
            >>> from gptf.transforms import Exp
            >>> from gptf.params import Param
            >>> class Example(Model):
            ...     def __init__(self, a, b):
            ...         assert isinstance(a, numbers.Number)
            ...         assert isinstance(b, numbers.Number)
            ...         super().__init__()
            ...         self.a = Param(a, transform=Exp(0.))
            ...         self.b = Param(b, transform=Exp(0.))
            ...     @WrappedTF.tf_method
            ...     @overrides
            ...     def build_log_likelihood(self):
            ...         return 10. - self.a.tensor - self.b.tensor

            TensorFlow optimizers
            ---------------------

            We can optimise the parameters of the model using a TensorFlow
            optimizer like so:
            >>> m = Example(3., 4.)
            >>> opt = tf.train.GradientDescentOptimizer(learning_rate=1)
            >>> m.optimize(opt)
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
            >>> m.optimize(opt)
            message: 'Finished iterations.'
            success: True
                  x: array([...])
            >>> print("m.a: {:.3f}".format(np.asscalar(m.a.value)))
            m.a: 0.001
            >>> print("m.b: {:.3f}".format(np.asscalar(m.b.value)))
            m.b: 1.000

            SciPy optimizers
            ----------------

            We can optimise the parameters of the model using a SciPy
            optimizer by provided a string value for `method`:
            >>> m = Example(3., 4.)
            >>> m.optimize('L-BFGS-B', disp=False, ftol=.0001)
            message: 'Finished iterations.'
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
            >>> m.optimize('L-BFGS-B', disp=False, ftol=.0001)
            message: 'Finished iterations.'
            success: True
                  x: array([...])
            >>> print("m.a: {:.3f}".format(np.asscalar(m.a.value)))
            m.a: 0.000
            >>> print("m.b: {:.3f}".format(np.asscalar(m.b.value)))
            m.b: 1.000

        """
        if "_Model__loss" not in self.cache:
            self.cache["_Model__loss"] = self._compile_loss()
        loss = self.cache["_Model__loss"]
        variables = [p.free_state for p in self.params if not p.fixed]
        free_state = tf.concat(0, [tf.reshape(v, [-1]) for v in variables])
        sess = self.get_session()

        try:
            if type(method) is str:
                options = {'maxiter': maxiter, 'disp': True}
                options.update(kw)
                optimizer = ScipyOptimizerInterface(loss, var_list=variables, 
                        method=method, options=options)
                optimizer.minimize(self.get_session(), self.feed_dict, 
                        step_callback=callback)
            else:
                # treat method as TensorFlow optimizer.
                opt_step = method.minimize(loss, var_list=variables, **kw)
                for _ in range(maxiter):
                    sess.run(opt_step, feed_dict=self.feed_dict)
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
                , message="Finished iterations."
                )

    def _compile_loss(self):
        return -self.build_log_likelihood()

