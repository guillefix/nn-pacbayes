import numpy as np
import tensorflow as tf
from gpflow import settings
import deep_ckern as dkern
import tqdm
import pickle_utils as pu
import sys
import os
import gpflow
from save_kernels import compute_big_K,mnist_1hot_all

# filter_sizes=[[5, 5], [2, 2], [5, 5], [2, 2]]
# padding=["VALID", "SAME", "VALID", "SAME"]
# strides=[[1, 1]] * 4
# sigmaw=100.0
# sigmab=100.0

def kernel_matrix(X,X2=None,image_size=28,number_channels=1,filter_sizes=[[5, 5], [2, 2], [5, 5], [2, 2]],padding=["VALID", "SAME", "VALID", "SAME"],strides=[[1, 1]] * 4, sigmaw=1.0,sigmab=1.0, n_gpus=1):
    with tf.device("cpu:0"):
        kern = dkern.DeepKernel(
            [number_channels, image_size, image_size],
            filter_sizes=filter_sizes,
            recurse_kern=dkern.ExReLU(multiply_by_sqrt2=True),
            var_weight=sigmaw**2,
            var_bias=sigmab**2,
            padding=padding,
            strides=strides,
            data_format="NCHW",
            skip_freq=-1,
            )

    # kern


    # N_train=20000; N_vali=1000
    # X, Y, Xv, _, Xt, _ = mnist_1hot_all()
    # # Xv = np.concatenate([X[N_train:, :], Xv], axis=0)[:N_vali, :]
    # X = X[:N_train]
    # Y = Y[:N_train]
    #
    # Y.shape
    #
    # ys = [int((np.argmax(labels)>5))*2.0-1 for labels in Y]

    # sess.close()
    sess = gpflow.get_default_session()

    K=compute_big_K(sess,kern,400,X,X2,n_gpus=n_gpus)
    sess.close()
    return K
