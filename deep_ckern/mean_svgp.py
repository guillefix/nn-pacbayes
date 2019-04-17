"""
SVGP / VGP, but we only optimise the mean and the kernel is precomputed
"""

import gpflow
from gpflow import settings
import numpy as np
import tensorflow as tf
import scipy


class PrecomputedKern:
    def __init__(self, vgp):
        self.vgp = vgp

    def K(self, X, X2=None):
        if X2 is None:
            return self.vgp.X
        elif X is self.vgp.X:
            return tf.transpose(X2[:, 1:])
        elif X2 is self.vgp.X:
            return X[:, 1:]
        else:
            raise NotImplementedError

    def Kdiag(self, X):
        return X[:, 0]


class CachedKernelVGP(gpflow.models.VGP_opper_archambeau):
    def __init__(self, X, Y, likelihood, num_latent=None, **kwargs):
        mean_function = None
        gpflow.models.GPModel.__init__(
            self, X, Y, PrecomputedKern(self), likelihood, mean_function,
            num_latent, **kwargs)
        self.num_data = X.shape[0]
        assert X.shape[0] == X.shape[1], "X has to be the kernel matrix"
        self.q_alpha = gpflow.params.Parameter(
            np.zeros((self.num_data, self.num_latent)),
            dtype=settings.float_type)
        self.q_lambda = gpflow.params.Parameter(
            np.ones((self.num_data, self.num_latent)),
            dtype=settings.float_type, transform=gpflow.transforms.positive)


class MeanVGP(CachedKernelVGP):
    def __init__(self, X, Y, likelihood, n_gpus=0,
                 num_latent=None, **kwargs):
        mean_function = None
        gpflow.models.GPModel.__init__(
            self, X, Y, PrecomputedKern(self), likelihood, mean_function,
            num_latent, **kwargs)
        self.num_data = X.shape[0]
        assert X.shape[0] == X.shape[1], "X has to be the kernel matrix"

        self.q_alpha = gpflow.params.Parameter(
            np.zeros((self.num_data, self.num_latent)),
            dtype=settings.float_type)

        self.q_lambda = gpflow.params.Parameter(
            1.0, dtype=settings.float_type, trainable=False)
        self.fvar = gpflow.params.Parameter(
            np.ones((self.num_data, 1)),
            dtype=settings.float_type, trainable=False)
        self.KL_pre = gpflow.params.Parameter(
            1.0, dtype=settings.float_type, trainable=False)
        # self.U_logdet = gpflow.params.Parameter(
        #     1.0, dtype=settings.float_type, trainable=False)
        # self.trAi = gpflow.params.Parameter(
        #     1.0, dtype=settings.float_type, trainable=False)

        self.n_gpus = n_gpus
        self.lik_per_gpu = []

    def precompute_fvar_and_KL(self):
        q_lambda = self.q_lambda.read_value().astype(np.float64)
        # print("mul")
        M = self.X.read_value().astype(np.float64) * q_lambda**2
        M.flat[::len(M)+1] += 1  # Add I, thus 1 to the diagonal
        v = np.eye(M.shape[0], dtype=np.float64) / q_lambda**2

        posv = scipy.linalg.get_lapack_funcs('posv', (M, v))
        U, Ai, info = posv(M, v, lower=False, overwrite_a=True, overwrite_b=True)
        print("System solved:", info)
        fvar = q_lambda**-2 - np.diag(Ai)[:, None]
        U_logdet = np.sum(np.log(np.diag(U)))
        trAi = np.trace(Ai) * q_lambda**2
        KL = self.num_latent * (
            U_logdet + 0.5*(trAi - self.num_data))  # + mean inner product

        self.fvar.assign(fvar.astype(settings.float_type))
        self.KL_pre.assign(KL.astype(settings.float_type))
        # self.trAi.assign(trAi.astype(settings.float_type) * self.num_latent)
        # self.U_logdet.assign(U_logdet.astype(settings.float_type) * 2 * self.num_latent)
        return fvar, KL

    @gpflow.decors.params_as_tensors
    def _build_likelihood(self):
        if self.n_gpus == 0:
            elems_per_gpu = self.num_data
        else:
            elems_per_gpu = (self.num_data + self.n_gpus - 1) // self.n_gpus

        diag_K = tf.expand_dims(tf.diag_part(self.X), axis=1)
        for i in range(max(1, self.n_gpus)):
            s = slice(i*elems_per_gpu, (i+1)*elems_per_gpu)
            print(i, "th slice:", s)
            with tf.device("{}:{}".format(
                    ('cpu' if self.n_gpus == 0 else 'cpu'), 0)):
                K = self.X[s, :]
                fmean = K_mu = K @ self.q_alpha
                maha = tf.reduce_sum(K_mu * self.q_alpha[s, :])
                v_exp = self.likelihood.variational_expectations(
                    fmean, self.fvar[s, :], self.Y[s, :])
                # return tf.stack([tf.reduce_sum(v_exp), self.U_logdet, self.trAi, maha])
                self.lik_per_gpu.append(tf.reduce_sum(v_exp) - .5*maha)
        return tf.add_n(self.lik_per_gpu, "likelihood_per_gpu") - self.KL_pre

    @gpflow.decors.params_as_tensors
    def objective_gradient(self):
        grads = []
        for i, obj in enumerate(self.lik_per_gpu):
            with tf.device("{}:{}".format(
                    ('cpu' if self.n_gpus == 0 else 'gpu'), i)):
                grads += tf.gradients(tf.negative(obj), self.q_alpha)
        return tf.add_n(grads, "objective_gradient")

    @gpflow.decors.params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        if full_cov:
            raise NotImplementedError
        fmean = self.kern.K(Xnew, self.X) @ self.q_alpha
        fvar = self.kern.Kdiag(Xnew)
        return fmean, fvar
