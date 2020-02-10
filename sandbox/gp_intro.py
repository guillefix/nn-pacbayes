import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

'''SETUP FLAGS'''

data_folder = "data/"
arch_folder = "archs/"
kernel_folder = "kernels/"

FLAGS = {}
FLAGS['m'] = 100
FLAGS['dataset'] =  "mnist"
FLAGS['network'] =  "fc"
FLAGS['number_inits'] = 1
FLAGS['label_corruption'] =  0.0
FLAGS['confusion'] = 0.0
FLAGS['binarized'] =  True
FLAGS['number_layers'] =  2
FLAGS['layer_widths'] =  "512"
FLAGS['activations'] =  "relu"
FLAGS['pooling'] =  "none"
FLAGS['intermediate_pooling'] =  "0"*FLAGS['number_layers']
FLAGS['intermediate_pooling_type'] =  "max"
FLAGS['init_dist'] =  "gaussian"
FLAGS['sigmaw'] =  1.41
FLAGS['sigmab'] =  0.0
FLAGS['prefix'] =  "test_"
FLAGS['whitening'] =  False
FLAGS['random_labels'] =  True
FLAGS['training'] =  True
FLAGS['no_training'] =  False
FLAGS['whitening'] =  False
FLAGS['centering'] =  False
FLAGS['channel_normalization'] =  False
FLAGS['random_labels'] =  True
FLAGS['training'] =  True
FLAGS['no_training'] =  False
FLAGS['threshold'] =  -1
FLAGS['oversampling'] =  False
FLAGS['oversampling2'] =  False
FLAGS['nn_random_labels'] =  False
FLAGS['nn_random_regression_outputs'] =  False

#%%

'''GET DATA'''

from utils import preprocess_flags
FLAGS = preprocess_flags(FLAGS)
globals().update(FLAGS)

from utils import load_data,load_model,load_kernel
train_images,flat_train_images,ys,test_images,test_ys = load_data(FLAGS)
input_dim = train_images.shape[1]
num_channels = train_images.shape[-1]

test_images = test_images[:50]
test_ys = test_ys[:50]

plt.imshow(train_images[0].reshape(28,28))

X = train_images
Xfull =  np.concatenate([train_images,test_images])
ys2 = [[y] for y in ys]
ysfull = ys2 + [[y] for y in test_ys]
Yfull = np.array(ysfull)
Y = np.array(ys2)

#%%

'''COMPUTE KERNEL'''

from nngp_kernel.fc_kernel import kernel_matrix
Kfull = kernel_matrix(Xfull,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab)

K = Kfull[0:m,0:m]





#%%

import GPy


import numpy as np
X = np.random.uniform(-3.,3.,(20,1))
Y = np.sin(X) + np.random.randn(20,1)*0.05
kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=2.)

# m = GPy.models.GPRegression(X,Y,kernel)
# m.parameters


inference_method = GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference()
lik=GPy.likelihoods.gaussian.Gaussian(variance=0.002)
m = GPy.core.GP(X=X,Y=Y,kernel=kernel,inference_method=inference_method, likelihood=lik)

fig = m.plot()
# GPy.plotting.show(fig, filename='basic_gp_regression_notebook')

#%%




inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=parallel_updates)
m = GPy.core.GP(X=X,
                Y=Y,
                kernel=CustomMatrix(X.shape[1],X,K),
                inference_method=inference_method,
                likelihood=lik)
