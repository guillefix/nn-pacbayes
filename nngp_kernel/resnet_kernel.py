import numpy as np
import tensorflow as tf
from gpflow import settings
import sys
import os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
nngp_kernel_folder = os.path.join(ROOT_DIR, 'nngp_kernel')
sys.path.append(nngp_kernel_folder)
import deep_ckern as dkern
import deep_ckern.resnet
import tqdm
import pickle_utils as pu
import gpflow
from save_kernels import compute_big_K,mnist_1hot_all,create_array_dataset

# filter_sizes=[[5, 5], [2, 2], [5, 5], [2, 2]]
# padding=["VALID", "SAME", "VALID", "SAME"]
# strides=[[1, 1]] * 4
# sigmaw=100.0
# sigmab=100.0

# depth=32
# image_size=28
# number_channels=1
# n_blocks=3
# sigmaw=1.0
# sigmab=0.0

def kernel_matrix(X,depth=5,image_size=28,number_channels=1,n_blocks=3, sigmaw=1.0,sigmab=1.0,n_gpus=1):
    # resnet_n=5
    block_depth = (depth - 2) // (n_blocks * 2)
    # resnet_n_plain = resnet_n % 100
    with tf.device("cpu:0"):
        kern = dkern.resnet.ResnetKernel(
            input_shape=[number_channels, image_size, image_size],
            # block_sizes=[resnet_n_plain]*depth,
            block_sizes=[block_depth]*n_blocks,
            block_strides=[1, 2, 2, 2, 2, 2, 2][:n_blocks],
            var_weight=sigmaw**2, # scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)**2,
            var_bias=sigmab**2,
            kernel_size=3,
            conv_stride=1,
            recurse_kern=(dkern.ExReLU() if depth < 100 else dkern.ExErf()),
            data_format='NCHW',
            )

        # kern


    #N_train=100; N_vali=1000
    #X, Y, Xv, _, Xt, _ = mnist_1hot_all()
    ## Xv = np.concatenate([X[N_train:, :], Xv], axis=0)[:N_vali, :]
    #X = X[:N_train]
    #Y = Y[:N_train]
        #
        # Y.shape
        #
        # ys = [int((np.argmax(labels)>5))*2.0-1 for labels in Y]

        # sess.close()

    #sess = gpflow.get_default_session()
    #N = X.shape[0]
    #out = create_array_dataset(False, N,N)

    #K=compute_big_K(out,sess,kern,100,X,n_gpus=n_gpus)

    sess = gpflow.get_default_session()

    K=compute_big_K(sess,kern,400,X,n_gpus=n_gpus)

    #K += 1e-6 * np.eye(len(X))
    # print(K)
    return K
