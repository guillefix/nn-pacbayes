import gpflow
import tensorflow as tf
import warnings
from typing import List
import numpy as np

from .dkern import DeepKernelBase
from .exkern import ElementwiseExKern

__all__ = ['ResnetKernel']


@gpflow.decors.name_scope()
def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in].
        kernel_size: The kernel to be used in the conv2d or max_pool2d
                     operation. Should be a positive integer.
    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    if pad_total == 0:
        return inputs
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == "NCHW":
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def conv2d_fixed_padding(inputs, var, kernel_size, strides,
                         data_format, name='conv2d_fixed_padding'):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    with tf.name_scope(name):
        if strides > 1:
            inputs = fixed_padding(inputs, kernel_size, data_format)
        chan_idx = data_format.index("C")
        try:
            C = int(inputs.shape[chan_idx])
        except TypeError:
            C = tf.shape(inputs)[chan_idx]
        fan_in = C * (kernel_size * kernel_size)
        W = tf.fill([kernel_size, kernel_size, C, 1], var/fan_in)
        if data_format == "NCHW":
            strides_shape = [1, 1, strides, strides]
        else:
            strides_shape = [1, strides, strides, 1]
        return tf.nn.conv2d(
            input=inputs, filter=W, strides=strides_shape,
            padding=('SAME' if strides == 1 else 'VALID'),
            data_format=data_format)


class ResnetKernel(DeepKernelBase):
    "Kernel equivalent to Resnet V2 (tensorflow/models/official/resnet)"
    def __init__(self,
                 input_shape: List[int],
                 block_sizes: List[int],
                 block_strides: List[int],
                 kernel_size: int,
                 recurse_kern: ElementwiseExKern,
                 var_weight: float = None,
                 var_bias: float = None,
                 conv_stride: int = 1,
                 active_dims: slice = None,
                 data_format: str = "NCHW",
                 input_type = None,
                 name: str = None):
        super(ResnetKernel, self).__init__(
            input_shape=input_shape, recurse_kern=recurse_kern,
            active_dims=active_dims, data_format=data_format,
            input_type=input_type, name=name)

        self.block_sizes = np.copy(block_sizes).astype(np.int32)
        self.block_strides = np.copy(block_strides).astype(np.int32)
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride

        if var_weight is None or var_bias is None:
            warnings.warn("var_weight or var_bias are None. Thus we ignore "
                          "them and output all the layers for architecture "
                          "search.")
            self.multiple_outputs = True
            self.var_weight = 1.
            self.var_bias = 0.
        else:
            self.multiple_outputs = False
            self.var_weight = gpflow.params.Parameter(
                var_weight, gpflow.transforms.positive, dtype=self.input_type)
            self.var_bias = gpflow.params.Parameter(
                var_bias, gpflow.transforms.positive, dtype=self.input_type)

    @gpflow.decors.params_as_tensors
    @gpflow.decors.name_scope()
    def specific_K(self, inputs, rec_K, rec_K_last, cross_start):
        """
        Apply the network that this kernel defines, except the last dense layer.
        The last dense layer is different for K and Kdiag.
        """
        # Copy from resnet_model.py
        inputs = conv2d_fixed_padding(
            inputs=inputs, var=self.var_weight,
            kernel_size=self.kernel_size,
            strides=self.conv_stride,
            data_format=self.data_format,
            name='initial_conv')

        if self.multiple_outputs:
            out = []
            def register_layer(layer, block):
                """
                Returns a function that adds an intermediate layer + a dense
                layer to the output
                """
                name = "dense_layer_{}_block_{}".format(layer, block)
                def f(x):
                    out.append(tf.reduce_mean(x[cross_start:], axis=(1, 2, 3),
                                              name=name))
                    return None
                return f
        else:
            out = []
            def register_layer(layer, block):
                def f(x):
                    pass
                return f

        for i, num_blocks in enumerate(self.block_sizes):
            with tf.name_scope("block_layer_{}".format(i+1)):
                # Only the first block per block_layer uses strides
                # and strides
                inputs = self.block_v2(inputs, True, self.block_strides[i],
                                       rec_K, register_layer(0, i))
                print("First layer of block {}:".format(i), inputs)
                for j in range(1, num_blocks):
                    inputs = self.block_v2(inputs, False, 1, rec_K, register_layer(j, i))
                    print("{}th layer of block {}:".format(j, i), inputs)

        # Dense layers at the end
        inputs = tf.reduce_mean(inputs, axis=(1, 2, 3))
        inputs = self.var_bias + self.var_weight * inputs
        out.append(inputs[cross_start:])
        if self.multiple_outputs:
            return out

        # Can add dense layers later by hand if necessary
        inputs = rec_K_last(inputs)
        inputs = self.var_bias + self.var_weight * inputs
        out.append(inputs)
        return inputs

    @gpflow.decors.params_as_tensors
    @gpflow.decors.name_scope()
    def block_v2(self, inputs, projection_shortcut, strides, rec_K, register_layer):
        shortcut = inputs
        inputs = rec_K(inputs)
        register_layer(inputs)
        if projection_shortcut:
            # Need to project the inputs to a smaller space and also apply ReLU
            del shortcut
            shortcut = conv2d_fixed_padding(
                inputs=inputs, var=self.var_weight, kernel_size=1,
                strides=strides, data_format=self.data_format,
                name='projection_shortcut')

        inputs = conv2d_fixed_padding(
            inputs=inputs, var=self.var_weight, kernel_size=3, strides=strides,
            data_format=self.data_format)
        inputs = rec_K(inputs)
        inputs = conv2d_fixed_padding(
            inputs=inputs, var=self.var_weight, kernel_size=3, strides=1,
            data_format=self.data_format)
        return inputs + shortcut
