"""GP regression with Gaussian noise."""
from builtins import super

from overrides import overrides
import tensorflow as tf

from gptf import GPModel, ParamAttributes, tf_method
from gptf import likelihoods, densities, meanfunctions
from gptf import tfhacks


#TODO: Write tests.
class GPR(GPModel, ParamAttributes):
    """Gaussian process regression with Gaussian noise.
    
    Attributes:
        inputs (DataHolder): The input data, size `N`x`D`. By default,
            this is set to recompile the model if the shape changes.
        values (DataHolder): The input data, size `N`x`D`. By default,
            this is set to recompile the model if the shape changes.
        kernel (gptf.kernels.Kernel): The kernel of the GP.
        meanfunc (gptf.meanfunctions.MeanFunctions): The mean function
            of the GP.
        likelihood (gptf.likelihoods.Gaussian): The likelihood of the GP

    Examples:
        >>> import numpy as np
        >>> from gptf import kernels
        >>> gp = GPR(kernels.RBF(variance=10.))
        >>> gp.fallback_name = "gp"
        >>> gp.likelihood.variance = .25 # reduce noise
        >>> print(gp.summary(fmt='plain'))
        Parameterized object gp
        <BLANKLINE>
        Params:
            name                   | value  | transform | prior
            -----------------------+--------+-----------+------
            gp.kernel.lengthscales | 1.000  | +ve (Exp) | nyi
            gp.kernel.variance     | 10.000 | +ve (Exp) | nyi
            gp.likelihood.variance | 0.250  | +ve (Exp) | nyi
        <BLANKLINE>

        To generate some sample training outputs, we'll compute a 
        sample from the prior with one latent function at our
        training inputs.
        >>> X = np.random.uniform(0., 5., (100, 1)) # 500 unique 1d points
        >>> Y = gp.compute_prior_samples(X, 1, 1)[0]
        
        Then we'll mess with the value of the parameters. When
        we optimise the model, they should return to something close
        to their original state.
        >>> gp.kernel.variance = 5.
        >>> gp.kernel.lengthscales = 5.
        >>> gp.likelihood.variance = 5.
        >>> gp.optimize(X, Y, disp=False)
        message: 'SciPy optimizer completed successfully.'
        success: True
              x: array([...,...,...])
        >>> abs(gp.kernel.lengthscales.value - 1.) < 1
        True
        >>> abs(gp.kernel.variance.value - 1.) < 1
        True
        >>> abs(gp.likelihood.variance.value - .25) < .1
        True

    """
    def __init__(self, kernel, meanfunction=meanfunctions.Zero(),
                 noise_variance=1.):
        """Initializer.

        Args:
            kernel (gptf.kernels.Kernel): The kernel.
            meanfunction (gptf.meanfunctions.MeanFunction):
                The mean function.
            noise_variance (float | gptf.Param): The variance of the
                noise. This will become self.likelihood.variance.

        """
        super().__init__()
        self.likelihood = likelihoods.Gaussian(noise_variance)
        self.kernel = kernel
        self.meanfunction = meanfunction
    
    @tf_method()
    @overrides
    def build_log_likelihood(self, X, Y):
        noise_variance = self.likelihood.variance.tensor
        K = self.kernel.K(X)
        # Add gaussian noise to kernel
        K += tfhacks.eye(tf.shape(X)[0], X.dtype) * noise_variance
        L = tf.cholesky(K)
        m = self.meanfunction(X)

        return densities.multivariate_normal(Y, m, L)

    @tf_method()
    @overrides
    def build_prior_mean_var(self, test_points, num_latent, full_cov=False):
        noise_var = self.likelihood.variance.tensor
        X = test_points
        fmean = self.meanfunction(X)
        fmean += tf.zeros([1, num_latent], fmean.dtype)  # broadcast mu
        if full_cov:
            fvar = self.kernel.K(X)
            fvar = tf.tile(tf.expand_dims(fvar, 2), (1, 1, num_latent))
        else:
            fvar = self.kernel.Kdiag(X)
            fvar = tf.tile(tf.expand_dims(fvar, 1), (1, num_latent))
        return fmean, fvar

    @tf_method()
    @overrides
    def build_posterior_mean_var(self, X, Y, test_points, full_cov=False):
        noise_var = self.likelihood.variance.tensor
        Kx = self.kernel.K(X, test_points)
        K = self.kernel.K(X)
        K += tfhacks.eye(tf.shape(X)[0], X.dtype) * noise_var
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, Y - self.meanfunction(X))
        fmean = tf.matmul(A, V, transpose_a=True)
        fmean += self.meanfunction(test_points)
        if full_cov:
            fvar = self.kernel.K(test_points) - tf.matmul(A, A, transpose_a=1)
            fvar = tf.tile(tf.expand_dims(fvar, 2), (1, 1, tf.shape(Y)[1]))
        else:
            fvar = self.kernel.Kdiag(test_points)
            fvar -= tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.expand_dims(fvar, 1), (1, tf.shape(Y)[1]))
        return fmean, fvar
