import gpflow
import tensorflow as tf
import numpy as np
from typing import List

from .resnet import conv2d_fixed_padding
from .dkern import DeepKernelBase
from .exkern import ElementwiseExKern

__all__ = ['AllConvKernel']


class AllConvKernel(DeepKernelBase):
    "Kernel equivalent to CIFAR-10 All-Convolutional Network"
    def __init__(self,
                 input_shape: List[int],
                 recurse_kern: ElementwiseExKern,
                 var_weight: float = None,
                 var_bias: float = None,
                 active_dims: slice = None,
                 data_format: str = "NCHW",
                 input_type = None,
                 name: str = None):
        super(AllConvKernel, self).__init__(
            input_shape=input_shape, recurse_kern=recurse_kern,
            active_dims=active_dims, data_format=data_format,
            input_type=input_type, name=name)

        self.var_weight = var_weight
        self.var_bias = var_bias
        self.multiple_outputs = True

    @gpflow.decors.params_as_tensors
    @gpflow.decors.name_scope()
    def specific_K(self, inputs, rec_K, rec_K_last, cross_start):
        """
        Apply the network that this kernel defines, except the last dense layer.
        The last dense layer is different for K and Kdiag.
        """
        # Copy from ./resnet.py

        # kernel, strides, name
        architecture = [
            (3, 1, 'conv_0_0'),
            (3, 1, 'conv_0_1'),
            (3, 2, 'conv_0_2'),
            #
            (3, 1, 'conv_1_0'),
            (3, 1, 'conv_1_1'),
            (3, 2, 'conv_1_2'),
            #
            (3, 1, 'relu_192_3'),
            (1, 1, 'relu_192_1'),
            (1, 1, 'relu_10_1'),
        ]

        out = []
        for kernel, strides, name in architecture:
            inputs = conv2d_fixed_padding(
                inputs=inputs, var=self.var_weight, kernel_size=kernel,
                strides=strides, data_format=self.data_format,
                name=name)
            inputs = rec_K(inputs)

            if self.multiple_outputs:
                x = tf.reduce_mean(inputs, axis=(1, 2, 3), name='dense_'+name)
                x = self.var_bias + self.var_weight * x
                out.append(x[cross_start:])
                x = rec_K_last(x)
                out.append(self.var_bias + self.var_weight*x)

        if self.multiple_outputs:
            return out
        return out[-1]
