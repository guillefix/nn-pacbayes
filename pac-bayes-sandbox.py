import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import tensorflow as tf

data_folder = "data/"
arch_folder = "archs/"
kernel_folder = "kernels/"
########## learning with GP

network = "cnn"
dataset = "mnist"
total_samples = 1000
whitening = False
number_layers = 4

import h5py
h5f = h5py.File(data_folder+network+"_"+dataset+"_"+str(total_samples)+("_whitening" if whitening else "")+"_data.h5",'r')
train_images = h5f['train_images'][:]
ys = h5f['ys'][:]
ys = [[y] for y in ys]
test_images = h5f['test_images'][:]
test_ys = h5f['test_ys'][:]
h5f.close()
input_dim = train_images.shape[1]
num_channels = train_images.shape[-1]
tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
flat_train_images = np.transpose(train_images, tp_order)  # NHWC -> NCHW # this is because the cnn GP kernels assume this
flat_train_images = np.array([train_image.flatten() for train_image in flat_train_images])
train_images = tf.constant(train_images)


K = np.load(open(kernel_folder+"K_"+str(total_samples)+"_"+dataset+"_"+network+"_"+str(number_layers)+("_whitening" if whitening else "")+".npy","rb"))

import GPy

from custom_kernel_matrix.custom_kernel_matrix import CustomMatrix
link_fun = GPy.likelihoods.link_functions.Heaviside()
lik = GPy.likelihoods.Bernoulli(gp_link=link_fun)

inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=True)
X = flat_train_images
Y = np.array(ys)
# Y
m = GPy.core.GP(X=X,
                Y=Y,
                kernel=CustomMatrix(X.shape[1],X,K),
                inference_method=inference_method,
                likelihood=lik)


m.log_likelihood()
