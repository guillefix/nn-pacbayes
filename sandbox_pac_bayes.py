import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import tensorflow as tf

data_folder = "data/"
arch_folder = "archs/"
kernel_folder = "kernels/"
########## learning with GP

# network = "cnn"
# dataset = "mnist"
# total_samples = 1000
# whitening = False
# number_layers = 4

# python3 generate_inputs_sample.py --m 10 --dataset mnist --sigmaw 10.0 --sigmab 10.0 --network fc --prefix test --random_labels --training --number_layers 1
FLAGS = {}
FLAGS['m'] = 10
FLAGS['number_inits'] = 1
FLAGS['label_corruption'] =  0.0
FLAGS['confusion'] = 0.0
FLAGS['dataset'] =  "mnist"
FLAGS['binarized'] =  True
FLAGS['number_layers'] =  1
FLAGS['pooling'] =  "none"
FLAGS['intermediate_pooling'] =  "0000"
FLAGS['sigmaw'] =  10.0
FLAGS['sigmab'] =  10.0
FLAGS['network'] =  "fc"
FLAGS['prefix'] =  "test"
FLAGS['whitening'] =  False
FLAGS['random_labels'] =  True
FLAGS['training'] =  True
FLAGS['no_training'] =  False

from utils import preprocess_flags
FLAGS = preprocess_flags(FLAGS)
globals().update(FLAGS)

from utils import load_data,load_model,load_kernel
train_images,flat_train_images,ys,test_images,test_ys = load_data(FLAGS)
input_dim = train_images.shape[1]
num_channels = train_images.shape[-1]
# tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
# train_images = tf.constant(train_images)

test_images = test_images[:500]
test_ys = test_ys[:500]

#%%

X = train_images
Xfull =  np.concatenate([train_images,test_images])
ys2 = [[y] for y in ys]
ysfull = ys2 + [[y] for y in test_ys]
Yfull = np.array(ysfull)
Y = np.array(ys2)
#
# from fc_kernel import kernel_matrix
# Kfull = kernel_matrix(Xfull,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab)


FLAGS["m"] = 1500
Kfull = load_kernel(FLAGS)
K = Kfull[0:m,0:m]

# filename=kernel_folder
# for flag in ["network","dataset","m","confusion","label_corruption","binarized","whitening","random_labels","number_layers","sigmaw","sigmab"]:
#     filename+=str(FLAGS[flag])+"_"
# filename += "kernel.npy"
# np.save(open(filename,"wb"),Kfull)
#

### trying gpflow now
#%%

import gpflow

import custom_kernel_gpflow
import imp; imp.reload(custom_kernel_gpflow)
from custom_kernel_gpflow import CustomMatrix

# m = gpflow.models.VGP(X, Y,
#                       kern=CustomMatrix(X.shape[1],X,K),
#                       likelihood=gpflow.likelihoods.Bernoulli())
#
# # VGP_opper_archambeau
# m = gpflow.models.VGP_opper_archambeau(X, Y,
#                       kern=CustomMatrix(X.shape[1],X,K),
#                       likelihood=gpflow.likelihoods.Bernoulli())
 # m = gpflow.models.SGPMC(X[:,None], Y[:,None],

# kk= gpflow.kernels.RBF(28**2)
#
# kk
# kk.compute_K(X[:10],X[:10])

# import imp; import custom_kernel_gpflow; imp.reload(custom_kernel_gpflow); from custom_kernel_gpflow import CustomMatrix
# kkk = CustomMatrix(X.shape[1],X,K)

# X[:10].shape
# kkk.compute_K(X[:10],X[:10])

# tf.reset_default_graph()
m = gpflow.models.GPMC(X.astype(np.float64), Y,
    kern=CustomMatrix(X.shape[1],X,K),
    # kern=gpflow.kernels.RBF(28*28),
    likelihood=gpflow.likelihoods.Bernoulli(),)
    # Z=X[::5].copy())


print(m)

# next(m.parameters)

#### MCMC
#%%

m.compile()
o = gpflow.train.AdamOptimizer(0.01)
o.minimize(m, maxiter=15) # start near MAP

s = gpflow.train.HMC()
# for i in range(2):
samples = s.sample(m, 20, epsilon=1e-4, lmax=15, lmin=5, thin=5, logprobs=False)#, verbose=True)

samples["GPMC/V"][18]
# samples_of_V = samples["GPMC/V"]
# sess = gpflow.get_default_session()
# m.V.read_value()
# loglik_samples = [sess.run(m._build_likelihood(), {m.V.constrained_tensor: v}) for v in samples_of_V]

# m.anchor(m.enquire_session())
loglik_samples = []
ps = []
for i, V in samples.iterrows():
    m.assign(V)
    loglik_samples.append(m.compute_log_likelihood())
    p = m.predict_y(test_images)[0].squeeze()
    ps.append(p)

loglik_samples

logPU = np.mean(loglik_samples)
logPU
ps = np.array(ps)
ps.shape
p = np.mean(ps,axis=0)

# sess.run(m._build_likelihood())
# sess.run(m.likelihood_tensor)

# m.compute_log_likelihood()

# m.predict_y(test_images[0])


#%%
# import imp; import utils; imp.reload(utils); from utils import load_kernel
# K = load_kernel(FLAGS)


import GPy
# import custom_kernel_matrix
# import imp
# imp.reload(custom_kernel_matrix)
from custom_kernel_matrix.custom_kernel_matrix import CustomMatrix
# link_fun = GPy.likelihoods.link_functions.Heaviside()
# lik = GPy.likelihoods.Bernoulli(gp_link=link_fun)
lik = GPy.likelihoods.Bernoulli()

inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=True)
inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(delta=0.1)
# inference_method = GPy.inference.mcmc.hmc.
# GPy.inference.latent_function_inference.InferenceMethodList()
# inference_method = inference_method=GPy.inference.latent_function_inference.PEP(alpha = 0.5)

m2 = GPy.core.GP(X=X,
                Y=Y,
                kernel=CustomMatrix(Xfull.shape[1],Xfull,Kfull),
                inference_method=inference_method,
                likelihood=lik)

m2.log_likelihood()
print(m2.log_likelihood()) #-417.66

mean, A = m2._raw_predict(test_images)


##################### generror bounds

#%%

# lik.predictive_values(mean,var)
# mean, var = m._raw_predict(Xfull[-2:-1])

p = m2.predict(test_images)[0].squeeze()
p = m.predict_y(test_images)[0].squeeze()

pdiscrete = p>0.5

mean_errors = (p**(1-test_ys))*((1-p)**(test_ys))
mean_error = np.mean(mean_errors)
mean_error #0.20

mean_errors = (pdiscrete**(1-test_ys))*((1-pdiscrete)**(test_ys))
mean_error = np.mean(mean_errors)
mean_error #0.14

from GP_prob_gpy import GP_prob
logPU = GP_prob(K,X,Y)

delta = 2**-10
bound = (-logPU+2*np.log(total_samples)+1-np.log(delta))/total_samples
# bound = (-logPU)/total_samples
bound = 1-np.exp(-bound)
bound

#%%

import matplotlib.pyplot as plt
%matplotlib
plt.matshow(test_images[1].reshape((28,28)))
plt.matshow(Xfull[-500].reshape((28,28)))
test_ys[1]


#%%

import torchvision
import numpy as np
from math import ceil

dataset = torchvision.datasets.KMNIST("./datasets",download=True)
dataset = torchvision.datasets.EMNIST("./datasets",download=True,split="byclass")
m=ceil(dataset.data.shape[0]*5/6)
(train_images,train_labels),(test_images,test_labels) = (dataset.data[:m], dataset.targets),(dataset.data[m:],dataset.targets)

train_images.shape

# dataset.test_data.shape
# dataset.data.shape

import matplotlib.pyplot as plt
import random
plt.matshow(train_images[random.randint(0,m-1)])

ceil(len(np.unique(train_labels))/2)
