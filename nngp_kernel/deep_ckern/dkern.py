import gpflow
from gpflow import settings
from typing import List
import numpy as np
import tensorflow as tf
import abc

from .exkern import ElementwiseExKern


class DeepKernelBase(gpflow.kernels.Kernel, metaclass=abc.ABCMeta):
    "General kernel for deep networks"
    def __init__(self,
                 input_shape: List[int],
                 recurse_kern: ElementwiseExKern,
                 active_dims: slice = None,
                 data_format: str = "NCHW",
                 input_type = None,
                 name: str = None):
        input_dim = np.prod(input_shape)
        super(DeepKernelBase, self).__init__(input_dim, active_dims, name=name)

        self.input_shape = list(np.copy(input_shape))
        self.recurse_kern = recurse_kern
        if input_type is None:
            input_type = settings.float_type
        self.input_type = input_type
        self.data_format = data_format

    @gpflow.decors.params_as_tensors
    @gpflow.decors.name_scope()
    def K(self, X, X2=None):
        # Concatenate the covariance between X and X2 and their respective
        # variances. Only 1 variance is needed if X2 is None.
        if X.dtype != self.input_type or (
                X2 is not None and X2.dtype != self.input_type):
            raise TypeError("Input dtypes are wrong: {} or {} are not {}"
                            .format(X.dtype, X2.dtype, self.input_type))
        if X2 is None:
            N = N2 = tf.shape(X)[0]
            var_z_list = [
                tf.reshape(tf.square(X), [N] + self.input_shape),
                tf.reshape(X[:, None, :] * X, [N*N] + self.input_shape)]
            cross_start = N

            @gpflow.decors.name_scope("rec_K_X_X")
            def rec_K(var_a_all, concat_outputs=True):
                var_a_1 = var_a_all[:N]
                var_a_cross = var_a_all[cross_start:]
                vz = [self.recurse_kern.Kdiag(var_a_1),
                      self.recurse_kern.K(var_a_cross, var_a_1, None)]
                if concat_outputs:
                    return tf.concat(vz, axis=0)
                return vz

        else:
            N, N2 = tf.shape(X)[0], tf.shape(X2)[0]
            var_z_list = [
                tf.reshape(tf.square(X), [N] + self.input_shape),
                tf.reshape(tf.square(X2), [N2] + self.input_shape),
                tf.reshape(X[:, None, :] * X2, [N*N2] + self.input_shape)]
            cross_start = N + N2

            @gpflow.decors.name_scope("rec_K_X_X2")
            def rec_K(var_a_all, concat_outputs=True):
                var_a_1 = var_a_all[:N]
                var_a_2 = var_a_all[N:cross_start]
                var_a_cross = var_a_all[cross_start:]
                vz = [self.recurse_kern.Kdiag(var_a_1),
                      self.recurse_kern.Kdiag(var_a_2),
                      self.recurse_kern.K(var_a_cross, var_a_1, var_a_2)]
                if concat_outputs:
                    return tf.concat(vz, axis=0)
                return vz
        inputs = tf.concat(var_z_list, axis=0)
        if self.data_format == "NHWC":
            # Transpose NCHW -> NHWC
            inputs = tf.transpose(inputs, [0, 2, 3, 1])

        inputs = self.specific_K(inputs, rec_K, lambda v: rec_K(v, False)[-1], cross_start)
        return self.postprocess_result(inputs, shape=[N, N2])

    def postprocess_result(self, inputs, shape=None):
        """
        It is possible that `self.specific_K` returns a list of kernel matrices
        rather than a single matrix. We handle both cases here.
        """
        def postprocess(i, result):
            with tf.name_scope("postprocess_{}".format(i)):
                if shape is not None:
                    result = tf.reshape(result, shape)

                if self.input_type != settings.float_type:
                    print("Casting kernel from {} to {}"
                        .format(self.input_type, settings.float_type))
                    return tf.cast(result, settings.float_type, name="cast_result")
                return result

        if isinstance(inputs, list):
            return list(postprocess(i, r) for (i, r) in enumerate(inputs))
        return postprocess(0, inputs)

    @gpflow.decors.params_as_tensors
    @gpflow.decors.name_scope()
    def Kdiag(self, X):
        if X.dtype != self.input_type:
            raise TypeError("Input dtype is wrong: {} is not {}"
                            .format(X.dtype, self.input_type))
        inputs = tf.reshape(tf.square(X), [-1] + self.input_shape)
        if self.data_format == "NHWC":
            # Transpose NCHW -> NHWC
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
        inputs = self.specific_K(
            inputs, self.recurse_kern.Kdiag, self.recurse_kern.Kdiag, 0)
        return self.postprocess_result(inputs)

    @abc.abstractmethod
    def specific_K(self, inputs, rec_K, rec_K_last, cross_start):
        """
        Apply the network that this kernel defines, except the last dense layer.
        The last dense layer is different for K and Kdiag.
        """
        raise NotImplementedError


class DeepKernelTesting(DeepKernelBase):
    """
    Reimplement original DeepKernel to test ResNet
    """
    def __init__(self,
                 input_shape: List[int],
                 hidden_layers: int,
                 kernel_size: int,
                 recurse_kern: ElementwiseExKern,
                 var_weight: float = 1.0,
                 var_bias: float = 1.0,
                 active_dims: slice = None,
                 data_format: str = "NCHW",
                 input_type = None,
                 name: str = None):
        super(DeepKernelTesting, self).__init__(
            input_shape=input_shape, recurse_kern=recurse_kern,
            active_dims=active_dims, data_format=data_format,
            input_type=input_type, name=name)

        if hidden_layers == 0:
            raise NotImplementedError("0 hidden layers not supported")
        self.hidden_layers = hidden_layers
        self.kernel_size = kernel_size
        self.var_weight = gpflow.params.Parameter(
            var_weight, gpflow.transforms.positive, dtype=self.input_type)
        self.var_bias = gpflow.params.Parameter(
            var_bias, gpflow.transforms.positive, dtype=self.input_type)

    @gpflow.decors.params_as_tensors
    @gpflow.decors.name_scope()
    def specific_K(self, inputs, rec_K, rec_K_last, cross_start):
        in_chans = int(inputs.shape[self.data_format.index("C")])
        W_init = tf.fill([self.kernel_size, self.kernel_size, in_chans, 1],
                        self.var_weight / in_chans)  # No dividing by receptive field
        inputs = tf.nn.conv2d(
            inputs, W_init, strides=[1,1,1,1], padding="SAME",
            data_format=self.data_format) + self.var_bias

        W = tf.fill([self.kernel_size, self.kernel_size, 1, 1],
                    self.var_weight)  # No dividing by fan_in
        for _ in range(1, self.hidden_layers):
            inputs = rec_K(inputs)
            inputs = tf.nn.conv2d(
                inputs, W, strides=[1,1,1,1], padding="SAME", data_format=self.data_format)
            inputs = inputs + self.var_bias
        inputs = rec_K_last(inputs)

        inputs = tf.reduce_mean(inputs, axis=(1, 2, 3))
        return self.var_bias + self.var_weight * inputs
