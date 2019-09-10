import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

#RUN this if no data file found
# python3 generate_inputs_sample.py --m 100 --dataset mnist --sigmaw 10.0 --sigmab 10.0 --network fc --prefix test --random_labels --training --number_layers 1
FLAGS = {}
FLAGS['m'] = 50
FLAGS['number_inits'] = 24
FLAGS['label_corruption'] =  0.0
FLAGS['confusion'] = 0.0
FLAGS['dataset'] =  "mnist"
# FLAGS['dataset'] =  "EMNIST"
FLAGS['binarized'] =  True
FLAGS['number_layers'] =  1
FLAGS['pooling'] =  "none"
FLAGS['intermediate_pooling'] =  "0000"
FLAGS['intermediate_pooling_type'] =  "max"
FLAGS['sigmaw'] =  2.0
FLAGS['sigmab'] =  0.0
FLAGS['network'] =  "fc"
FLAGS['prefix'] =  "test_"
FLAGS['whitening'] =  False
FLAGS['centering'] =  False
FLAGS['channel_normalization'] =  False
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
test_set_size = 50
test_images = test_images[:test_set_size]
test_ys = test_ys[:test_set_size]


X = train_images
Xfull =  np.concatenate([train_images,test_images])
ys2 = [[y] for y in ys]
ysfull = ys2 + [[y] for y in test_ys]
Yfull = np.array(ysfull)
Y = np.array(ys2)

Xfull.shape

from nngp_kernel.fc_kernel import kernel_matrix
number_layers = 6
Kfull = kernel_matrix(Xfull,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab)


# FLAGS["m"] = 1500
# Kfull = load_kernel(FLAGS)
K = Kfull[0:m,0:m]

# filename=kernel_folder
# for flag in ["network","dataset","m","confusion","label_corruption","binarized","whitening","random_labels","number_layers","sigmaw","sigmab"]:
#     filename+=str(FLAGS[flag])+"_"
# filename += "kernel.npy"
# np.save(open(filename,"wb"),Kfull)
#

### GP exact inference for classification tasks! ###

#%%

import os

os.environ["CUDA_PATH"] = "/usr/local/cuda-10.0"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-10.0/lib64:/usr/local/cuda-8.0/lib64::/usr/local/lib:/usr/local/cuda-10.0/lib64"

import cupy as cp
# import numpy as cp

Yfull = np.array(ysfull)
Y = np.array(ys2)
Y = cp.array(Y)
Yfull = cp.array(Yfull)

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

#%%

generrors=[]
# for i in range(1000):
while len(generrors)<100:
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    exact_samples = cp.random.multivariate_normal(cp.zeros(m+test_set_size),Kfull,int(1e5),dtype=np.float32)>0
    # exact_samples = cp.random.multivariate_normal(cp.zeros(m+test_set_size),Kfull,int(1e6))>0


    # Y_extended = np.concatenate([Y.T[0,:],np.ones(50)])==1
    fits_data = cp.prod(~(exact_samples[:,:m]^(Y.T==1)),1)

    # indices = cp.where(fits_data)
    indices = cp.where(fits_data)[0]

    generrors += (cp.sum(~(exact_samples[indices,:][:,m:]^(Yfull.T[0,m:]==1)),1)/test_set_size*1.0).tolist()

# np.mean(np.concatenate(generrors))
print(len(generrors), generrors)
np.mean(generrors)

#%%

freq = 0
for i in range(1000):
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    exact_samples = cp.random.multivariate_normal(cp.zeros(m),K,int(1e5),dtype=np.float32)>0
    # exact_samples = cp.random.multivariate_normal(cp.zeros(m+test_set_size),Kfull,int(1e6))>0


    # Y_extended = np.concatenate([Y.T[0,:],np.ones(50)])==1
    fits_data = cp.prod(~(exact_samples[:,:m]^(Y.T==1)),1)

    # indices = cp.where(fits_data)
    indices = cp.where(fits_data)[0]
    freq += len(indices)

    # generrors += (cp.sum(~(exact_samples[indices,:][:,m:]^(Yfull.T[0,m:]==1)),1)/test_set_size*1.0).tolist()

print(freq/1e8)
# print(len(generrors), generrors)
# np.mean(generrors)

###############################

#%%

### trying TF prob now
y_tensor = tf.squeeze(tf.constant(Y,dtype=tf.float64))
import tensorflow_probability as tfp

tfd = tfp.distributions

init_f = tf.zeros((m,),dtype=tf.float64)
# init_f = tf.zeros((m,),dtype=tf.float32)
P = tfd.MultivariateNormalFullCovariance(loc=tf.zeros_like(init_f, dtype=tf.float64),covariance_matrix=K)

# L=tfd.Logistic(loc=tf.zeros_like(init_f, dtype=tf.float64),scale=tf.ones_like(init_f, dtype=tf.float64))
# L=tfd.Logistic(loc=[0]*m,scale=[1.0]*m)
# why doesn't logistic work with negative values???


# init_f.shape
# errors = tf.math.equal(tf.dtypes.cast(init_f > 0, tf.float64), y_tensor)
# # errors
# sess=tf.Session()
# with sess.as_default():
#     thing = tf.math.log(tf.dtypes.cast(errors,tf.float32)).eval()
    # thing = unnormalized_posterior_log_prob(init_f).eval()
    # print(((2*y_tensor-1)*(init_f+1)*10).eval())
    # thing=-tf.math.log(1+tf.math.exp(-(2*y_tensor-1)*(init_f))).eval()
    # thing = L.log_prob((2*y_tensor-1)*(init_f+1)*10).eval()
    # thing = L.prob(init_f+10).eval()

#
# P.log_prob(init_f)
# thing
# np.log(thing)

beta = 5
def unnormalized_posterior_log_prob(f):
    # errors = tf.math.equal(tf.dtypes.cast(f > 0,tf.float64), y_tensor)
    # return P.log_prob(f) + tf.math.reduce_sum(tf.math.log(tf.dtypes.cast(errors,tf.float64)))
    # return P.log_prob(f) - tf.math.reduce_sum(tf.math.log(1+tf.math.exp(-beta*(2*y_tensor-1)*f)))
    return P.log_prob(f)

sample_chain = tfp.mcmc.sample_chain

Nsamples = 10000
states, kernel_results = sample_chain(
        num_results=Nsamples,
        num_burnin_steps=500,
        current_state=(
            init_f,
        ),
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_posterior_log_prob,
            step_size=0.5,
            num_leapfrog_steps=2))

with tf.Session() as sess:
    states, is_accepted_ = sess.run([states, kernel_results.is_accepted])
    # accepted = np.sum(np.prod(is_accepted_,1))
    accepted = np.sum(is_accepted_)
    print("Acceptance rate: {}".format(accepted / Nsamples))

states[0].shape
states[0]

np.prod(states[0][0,:] > 0)
((states[0][0,:] > 0) == Y).all()

np.prod((1/(1+np.exp(-beta*(2*Y-1)*states[0][0,:]))))

# totprob = 0
totlogprob = 0
for f in states[0]:
    # totprob += ((states[0][0,:] > 0) == Y).all()
    # totprob += np.prod((1/(1+np.exp(-beta*(2*Y-1)*states[0][0,:]))))
    totlogprob += np.sum(-np.log(1+np.exp(-beta*(2*Y-1)*states[0][0,:])))

## importance sampling?

##############
### now trying pyro
## N U T S

import torch, pyro
from pyro.distributions import MultivariateNormal, Bernoulli
from pyro.infer.mcmc import MCMC, NUTS
# import torch.nn.functional as F

# y_tensor = torch.ByteTensor(Y).squeeze()
y_tensor = torch.Tensor(Y).squeeze()

def model():
    prior = MultivariateNormal(torch.zeros(m),torch.Tensor(K))
    fs = pyro.sample("fs",prior)
    likelihood = Bernoulli(probs = (fs > 0).float())
    # softprobs = torch.sigmoid(fs)
    # likelihood = Bernoulli(probs = softprobs)
    ys = pyro.sample("ys",likelihood)
    return ys
    # dataprob = torch.prod((fs > 0) == y_tensor)
    # return (fs > 0).float()

# likelihood = Bernoulli(probs = (fs > 0).float())
# fs.unsqueeze(-1)
# softprobs = torch.sigmoid(fs)
# likelihood = Bernoulli(probs = softprobs)
#
# torch.prod((fs > 0) == torch.ByteTensor(Y).squeeze())

# conditioned_model = pyro.condition(model, data={"ys":y_tensor})
conditioned_model = pyro.condition(model, data={})

thing = model(); thing
thing.shape
# fs.shape

# likelihood.sample()

# def model(data):

# nuts_kernel = NUTS(model, adapt_step_size=True)
nuts_kernel = NUTS(conditioned_model, adapt_step_size=True)

mcmc_run = MCMC(nuts_kernel, num_samples=500, warmup_steps=300).run()

posterior = pyro.infer.abstract_infer.EmpiricalMarginal(mcmc_run, 'fs')

posterior.


from pyro.infer.abstract_infer import EmpiricalMarginal
import pyro.distributions as dist
true_coefs = torch.tensor([1., 2., 3.])
data = torch.randn(2000, 3)
dim = 3
labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()
def model(data):
 coefs_mean = torch.zeros(dim)
 coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))
 # y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
 y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)))
 return y
nuts_kernel = NUTS(model, adapt_step_size=True)
mcmc_run = MCMC(nuts_kernel, num_samples=500, warmup_steps=300).run(data)
posterior = EmpiricalMarginal(mcmc_run, 'beta')
posterior.mean

##################################

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
    kern=CustomMatrix(Xfull.shape[1],Xfull,Kfull),
    # kern=gpflow.kernels.RBF(28*28),
    likelihood=gpflow.likelihoods.Bernoulli(),)
    # Z=X[::5].copy())


print(m)

# next(m.parameters)

#### MCMC
#%%

m.compile()
o = gpflow.train.AdamOptimizer(0.01)
o.minimize(m, maxiter=5) # start near MAP

s = gpflow.train.HMC()
# for i in range(2):
samples = s.sample(m, 500, epsilon=2e-3, lmax=20, lmin=5, thin=25, logprobs=False)#, verbose=True)

samples["GPMC/V"][9]
samples["GPMC/V"][0]
# samples_of_V = samples["GPMC/V"]
# sess = gpflow.get_default_session()
# m.V.read_value()
# loglik_samples = [sess.run(m._build_likelihood(), {m.V.constrained_tensor: v}) for v in samples_of_V]

# m.anchor(m.enquire_session())
loglik_samples = []
ps = []
# for i, V in samples.iterrows():
#     print(V)
for i, V in samples.iterrows():
    m.assign(V)
    loglik_samples.append(m.compute_log_likelihood())
    p = m.predict_y(test_images)[0].squeeze()
    ps.append(p)

#wrongggg v

print(loglik_samples)

logPU = np.mean(loglik_samples)
print(logPU)
ps = np.array(ps)
print(ps)
ps.shape
p = np.mean(ps,axis=0)
print(p)

pdiscrete = p>0.5

mean_errors = (p**(1-test_ys))*((1-p)**(test_ys))
mean_error = np.mean(mean_errors)
mean_error #0.20

mean_errors = (pdiscrete**(1-test_ys))*((1-pdiscrete)**(test_ys))
mean_error = np.mean(mean_errors)
mean_error #0.14

from GP_prob_gpy import GP_prob
# logPU = GP_prob(K,X,Y)

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
