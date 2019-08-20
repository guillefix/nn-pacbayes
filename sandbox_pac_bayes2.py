import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib
import tensorflow as tf

data_folder = "data/"
arch_folder = "archs/"
kernel_folder = "kernels/"

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

X = train_images
Xfull =  np.concatenate([train_images,test_images])
ys2 = [[y] for y in ys]
ysfull = ys2 + [[y] for y in test_ys]
Yfull = np.array(ysfull)
Y = np.array(ys2)

#%%

from fc_kernel import kernel_matrix
Kfull = kernel_matrix(Xfull,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab)
filename=kernel_folder
FLAGS["m"] = 1500
for flag in ["network","dataset","m","confusion","label_corruption","binarized","whitening","random_labels","number_layers","sigmaw","sigmab"]:
    filename+=str(FLAGS[flag])+"_"
filename += "kernel.npy"
np.save(open(filename,"wb"),Kfull)
K = Kfull[0:m,0:m]

#%%

Kfull = load_kernel(FLAGS)
K = Kfull[0:m,0:m]
#%%

import GPy

from custom_kernel_matrix.custom_kernel_matrix import CustomMatrix
# link_fun = GPy.likelihoods.link_functions.Heaviside()
# lik = GPy.likelihoods.Bernoulli(gp_link=link_fun)
lik = GPy.likelihoods.Bernoulli()

inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=True)
# X = flat_train_images
# Y = np.array(ys)
# Y
model = GPy.core.GP(X=X,
                Y=Y,
                kernel=CustomMatrix(X.shape[1],X,K),
                inference_method=inference_method,
                likelihood=lik)


#%%

# m.posterior_samples(X, size=20)
mean, cov = model._raw_predict(X, full_cov=True)
# mean = mean.squeeze()
# cov.shape

# from scipy.stats import multivariate_normal
# var = multivariate_normal(mean=mean, cov=cov)
# var.pdf(f)

# np.matmul(np.linalg.inv(cov), (f-mean).T)

num_post_samples = int(1e7)
sample = model.posterior_samples_f(X, size=num_post_samples)
# f = sample[:,:,0]
normalization1 = (np.sqrt(np.power(2*np.pi,len(X)) * np.linalg.det(cov)))
normalization2 = (np.sqrt(np.power(2*np.pi,len(X)) * np.linalg.det(K)))
covinv = np.linalg.inv(cov)
Kinv = np.linalg.inv(K)
tot = 0
shift = m*np.log(2)*0.3
for i in range(num_post_samples):
    f = sample[:,:,i]
    # print(f.shape)
    # Q = np.exp(-0.5*np.matmul( (f-mean).T, np.matmul(covinv, (f-mean)) ))/normalization1
    # P = np.exp(-0.5*np.matmul( (f).T, np.matmul(Kinv, (f)) ))/normalization2
    PQratio = np.exp(shift-0.5*(np.matmul(f.T, np.matmul(Kinv, f)) - np.matmul((f-mean).T, np.matmul(covinv, (f-mean))) ))*normalization1/normalization2
    # PQratio = 0.5*(np.matmul(f.T, np.matmul(Kinv, f)) - np.matmul((f-mean).T, np.matmul(covinv, (f-mean))) ) - np.log(normalization1/normalization2)
    if np.prod((f.T>0) == Y.T):
        tot += PQratio

PU = tot/num_post_samples

np.log(PU) - shift

#%%

exact_samples = np.random.multivariate_normal(np.zeros(m),K,int(1e7))>0

count = 0
for i in range(len(exact_samples)):
    # print(sum(exact_samples[i,:]))
    if np.prod(exact_samples[i,:] == Y.T):
        count += 1

PU = count/1e7
PU
np.log(PU)
# m.likelihood.log_predictive_density(X,sample[:,:,0])

# m.log_likelihood()
