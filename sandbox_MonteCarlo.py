import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib
import tensorflow as tf

data_folder = "data/"
arch_folder = "archs/"
kernel_folder = "kernels/"

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

# python3 generate_inputs_sample.py --m 500 --dataset mnist --training --number_layers 1

from utils import preprocess_flags
FLAGS = preprocess_flags(FLAGS)
globals().update(FLAGS)

from utils import load_data,load_model,load_kernel
train_images,flat_train_images,ys,test_images,test_ys = load_data(FLAGS)
input_dim = train_images.shape[1]
num_channels = train_images.shape[-1]
# tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
# train_images = tf.constant(train_images)

X = train_images
ys2 = [[y] for y in ys]
Y = np.array(ys2)

#%%
print("Loading kernel")
from os import path
# FLAGS["m"] = m+500
filename=kernel_folder
for flag in ["network","dataset","m","confusion","label_corruption","binarized","whitening","random_labels","number_layers","sigmaw","sigmab"]:
    filename+=str(FLAGS[flag])+"_"
filename += "kernel.npy"
if path.exists(filename):
    K = load_kernel(FLAGS)
else:
    from fc_kernel import kernel_matrix
    K = kernel_matrix(X,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab)
    np.save(open(filename,"wb"),K)

print("Loaded kernel")
#%%

import GPy

from custom_kernel_matrix.custom_kernel_matrix import CustomMatrix
# link_fun = GPy.likelihoods.link_functions.Heaviside()
# lik = GPy.likelihoods.Bernoulli(gp_link=link_fun)
lik = GPy.likelihoods.Bernoulli()

inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=True)
# inference_method = GPy.inference.latent_function_inference.laplace.Laplace()

model = GPy.core.GP(X=X,
                Y=Y,
                kernel=CustomMatrix(X.shape[1],X,K),
                inference_method=inference_method,
                likelihood=lik)
#%%

mean, cov = model._raw_predict(X, full_cov=True)
mean *= 1
mean = mean.flatten()
cov /= 3
num_post_samples = int(1e5)
# sample = model.posterior_samples_f(X, size=num_post_samples)

# np.prod((sample[:,0,np.random.randint(num_post_samples)].T>0) == Y.T)
# np.prod((mean.T>0) == Y.T)

# normalization1 = (np.sqrt(np.power(2*np.pi,len(X)) * np.linalg.det(cov)))
# normalization2 = (np.sqrt(np.power(2*np.pi,len(X)) * np.linalg.det(K)))
# norm_ratio = np.sqrt(np.linalg.det(cov) / np.linalg.det(K))
log_norm_ratio = np.sum(np.log((np.linalg.eigh(cov)[0] / np.linalg.eigh(K)[0])))/2
covinv = np.linalg.inv(cov)
Kinv = np.linalg.inv(K)
tot = 0
shift = m*np.log(2)*0.3
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)
# num_inits_per_task = 1
num_tasks = num_post_samples
num_tasks_per_job = num_tasks//size
tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))
if rank < num_tasks%size:
    tasks.append(size*num_tasks_per_job+rank)

sample = np.random.multivariate_normal(mean, cov, len(tasks)).T

import sys
for i in range(len(tasks)):
    # print(i)
    if (i%(len(tasks)/100)) == (len(tasks)/100)-1:
        print(str(int(100*i/len(tasks)))+"%")
    f = sample[:,i]
    # Q = np.exp(-0.5*np.matmul( (f-mean).T, np.matmul(covinv, (f-mean)) ))/normalization1
    # P = np.exp(-0.5*np.matmul( (f).T, np.matmul(Kinv, (f)) ))/normalization2
    PQratio = np.exp(shift-0.5*(np.matmul(f.T, np.matmul(Kinv, f)) - np.matmul((f-mean).T, np.matmul(covinv, (f-mean))) ) + log_norm_ratio)
    # PQratio = np.exp(shift-0.5*(np.matmul(f.T, np.matmul(Kinv, f)) - np.matmul((f-mean).T, np.matmul(covinv, (f-mean))) ))*norm_ratio
    # PQratio = 0.5*(np.matmul(f.T, np.matmul(Kinv, f)) - np.matmul((f-mean).T, np.matmul(covinv, (f-mean))) ) - np.log(normalization1/normalization2)
    if np.prod((f.T>0) == Y.T):
        print(PQratio)
        tot += PQratio

tots = comm.gather(tot,root=0)
if rank == 0:
    tot = sum(tots)
    PU = tot/num_post_samples

    # logPU = np.log(PU) - shift
    logPU = np.log(PU)
    print(logPU)

    #%%

    from GP_prob_gpy import GP_prob
    logPU = GP_prob(K,X,Y)
    print(logPU)


# exact_samples = np.random.multivariate_normal(np.zeros(m),K,int(1e7))>0
#
# count = 0
# for i in range(len(exact_samples)):
#     # print(sum(exact_samples[i,:]))
#     if np.prod(exact_samples[i,:] == Y.T):
#         count += 1
#
# PU = count/1e7
# PU
# np.log(PU)
# m.likelihood.log_predictive_density(X,sample[:,:,0])

# m.log_likelihood()
