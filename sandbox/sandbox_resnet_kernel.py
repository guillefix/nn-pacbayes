import numpy as np
import tensorflow as tf
from gpflow import settings
import deep_ckern as dkern
import deep_ckern.resnet
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

depth=32
image_size=28
number_channels=1
n_blocks=3
sigmaw=1.0
sigmab=0.0

# def kernel_matrix(X,depth=32,image_size=28,number_channels=1,n_blocks=3, sigmaw=1.0,sigmab=1.0):
block_depth = (depth - 2) // 6
with tf.device("cpu:0"):
    kern = dkern.resnet.ResnetKernel(
        input_shape=[number_channels, image_size, image_size],
        block_sizes=[block_depth]*n_blocks,
        block_strides=[1, 2, 2, 2, 2, 2, 2][:n_blocks],
        var_weight=sigmaw**2, # scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)**2,
        var_bias=sigmab**2,
        kernel_size=3,
        conv_stride=1,
        recurse_kern=dkern.ExReLU(),
        data_format='NCHW',
        )

        # kern


N_train=10000; N_vali=1000
X, Y, Xv, _, Xt, _ = mnist_1hot_all()
# Xv = np.concatenate([X[N_train:, :], Xv], axis=0)[:N_vali, :]
X = X[:N_train]
Y = Y[:N_train]
        #
        # Y.shape
        #
        # ys = [int((np.argmax(labels)>5))*2.0-1 for labels in Y]

        # sess.close()
sess = gpflow.get_default_session()

K=compute_big_K(sess,kern,20,X)
    # print(K)
    # return K

# np.sum(np.isnan(K))

from GP_prob import GP_prob
import pickle
# pickle.dump(K,open("resnet_K_1000.p","wb"))
K = pickle.load(open("resnet_K_1000.p","rb"))
ys2=[int(np.argmax(y)>=5)*2.0-1 for y in Y]
ys=np.array([[int(np.argmax(y)>=5)] for y in Y])

logPU = GP_prob(K,ys2,1e-9,cpu=False)

logPU

X.shape[1]

import imp
import GP_prob_gpy
imp.reload(GP_prob_gpy)
# # import custom_kernel_matrix
# imp.reload(custom_kernel_matrix)
from GP_prob_gpy import GP_prob

import time
time.
GP_prob(K,X,ys)

import GPy

# GPy.likelihoods.Bi

from custom_kernel_matrix.custom_kernel_matrix import CustomMatrix

lik = GPy.likelihoods.Bernoulli()
m = GPy.core.GP(X=X,
                Y=ys,
                kernel=CustomMatrix(X.shape[1],X,K),
                # inference_method=GPy.inference.latent_function_inference.PEP(alpha = 0.5), #only for regression apparently
                # inference_method=GPy.inference.latent_function_inference.expectation_propagation.EP(),
                likelihood=lik)
# m.likelihood = lik
m.inference_method = GPy.inference.latent_function_inference.PEP(alpha = 0.5)

# m.inference_method

m.log_likelihood()

imp.reload(custom_kernel_matrix.custom_kernel_matrix)

from custom_kernel_matrix.custom_kernel_matrix import CustomMatrix
import custom_kernel_gpflow,imp
imp.reload(custom_kernel_gpflow)
from custom_kernel_gpflow import CustomMatrix

# CustomMatrix(X.shape[1],K)
#
import gpflow

# K

# X.shape

m = gpflow.models.VGP(X, ys,
                      kern=CustomMatrix(X.shape[1],X,K),
                      likelihood=gpflow.likelihoods.Bernoulli())

m = gpflow.models.VGP_opper_archambeau(X, ys,
                      kern=CustomMatrix(X.shape[1],X,K),
                      likelihood=gpflow.likelihoods.Bernoulli())

# m = gpflow.models.SGPMC(X, ys,
#                       kern=CustomMatrix(X.shape[1],X,K),
#                       likelihood=gpflow.likelihoods.Bernoulli(),
#                       Z=X[::5].copy())
#
# VGP_opper_archambeau
sess = gpflow.get_default_session()

sess.run(m._build_likelihood())
sess.run(m.likelihood_tensor)
# sess.run(m.compute_log_likelihood())
# from cnn_kernel import kernel_matrix
#
# kernel_matrix(X,X)
