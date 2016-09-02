"""GP regression with Gaussian noise."""
from builtins import super

from overrides import overrides
import tensorflow as tf

from gptf import GPModel, DataHolder, tf_method
from gptf import likelihoods, densities, meanfunctions
from gptf import tfhacks


#TODO: Write tests.
class GPR(GPModel):
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

    """
    def __init__(self, inputs, values, kernel, 
            meanfunction=meanfunctions.Zero()):
        """Initializer.

        Args:
            inputs (DataHolder | np.ndarray): A data matrix of size 
                `N`x`D`, representing the input data.
            values (DataHolder | np.ndarray): A data matrix of size
                `N`x`R`, representing the observed values of the
                latent function(s) at the inputs.
            kernel (gptf.kernels.Kernel): The kernel.
            meanfunction (gptf.meanfunctions.MeanFunction):
                The mean function.

        """
        super().__init__(likelihoods.Gaussian())
        self.inputs = (inputs if isinstance(inputs, DataHolder) 
                else DataHolder(inputs, on_shape_change='recompile'))
        self.values = (values if isinstance(inputs, DataHolder) 
                else DataHolder(values, on_shape_change='recompile'))
        self.kernel = kernel
        self.meanfunction = meanfunc
    
    @tf_method
    @overrides
    def build_log_likelihood(self):
        X = self.inputs.tensor
        Y = self.values.tensor
        K = self.kernel.K(X) + tfhacks.eye(tf.shape(X)[0], X.dtype)
        K *= self.likelihood.variance.tensor
        L = tf.cholesky(K)
        m = self.meanfunction(X)

        return densities.multivariate_normal(Y, m, L)

    @tf_method
    @overrides
    def build_predict(self, test_points, full_cov=False):
        X = self.inputs.tensor
        Y = self.inputs.tensor
        Kx = self.kernel.K(X, test_points)
        K = self.kernel.K(X) + tfhacks.eye(tf.shape(X)[0], X.dtype)
        K *= self.likelihood.variance.tensor
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, Y - self.meanfunction(X))
        fmean = tf.matmul(A, V, transpose_a=True)
        fmean += self.mean_function(test_points)
        if full_cov:
            fvar = self.kernel.K(test_points) - tf.matmul(A, A, transpose_a=1)
            fvar = tf.tile(tf.expand_dims(fvar, 2), (1, 1, tf.shape(Y)[1]))
        else:
            fvar = self.kernel.Kdiag(test_points)
            fvar -= tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.expand_dims(fvar, 1), (1, tf.shape(Y)[1]))
        return fmean, fvar
