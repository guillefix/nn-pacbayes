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


FLAGS = {}
FLAGS['m'] = 1000
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

from fc_kernel import kernel_matrix
Kfull = kernel_matrix(Xfull,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab)

K = Kfull[0:1000,0:1000]


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

m = GPy.core.GP(X=X,
                Y=Y,
                kernel=CustomMatrix(Xfull.shape[1],Xfull,Kfull),
                inference_method=inference_method,
                likelihood=lik)

m.log_likelihood()
print(m.log_likelihood())

mean, A = m._raw_predict(test_images)

### trying gpflow now

import gpflow

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

kk= gpflow.kernels.RBF(28**2)

kk
kk.compute_K(X[:10],X[:10])

import imp; import custom_kernel_gpflow; imp.reload(custom_kernel_gpflow); from custom_kernel_gpflow import CustomMatrix
kkk = CustomMatrix(X.shape[1],X,K)

X[:10].shape
kkk.compute_K(X[:10],X[:10])

m = gpflow.models.GPMC(X.astype(np.float64), Y,
    kern=CustomMatrix(X.shape[1],X,K),
    # kern=gpflow.kernels.RBF(28*28),
    likelihood=gpflow.likelihoods.Bernoulli(),)
    # Z=X[::5].copy())


print(m)

# next(m.parameters)


#%%

m.compile()

# o = gpflow.train.AdamOptimizer(0.01)
# o.minimize(m, maxiter=15) # start near MAP

s = gpflow.train.HMC()
samples = s.sample(m, 100, epsilon=1e-2, lmax=5, lmin=1, thin=5, logprobs=False)#, verbose=True)

samples["GPMC/V"][7]

sess = gpflow.get_default_session()

samples_of_V = samples["GPMC/V"]

m.anchor(m.enquire_session())

# m.V.read_value()

loglik_samples = [sess.run(m._build_likelihood(), {m.V.constrained_tensor: v}) for v in samples_of_V]

loglik_samples

np.mean(loglik_samples)

sess.run(m._build_likelihood())
# sess.run(m.likelihood_tensor)

m.compute_log_likelihood()

m.predict_y(test_images[0])

##################### generror bounds

#%%

# lik.predictive_values(mean,var)
# mean, var = m._raw_predict(Xfull[-2:-1])

p = m.predict(test_images)[0].squeeze()

pdiscrete = p>0.5

mean_errors = (p**(1-test_ys))*((1-p)**(test_ys))
mean_error = np.mean(mean_errors)
mean_error

mean_errors = (pdiscrete**(1-test_ys))*((1-pdiscrete)**(test_ys))
mean_error = np.mean(mean_errors)
mean_error

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
