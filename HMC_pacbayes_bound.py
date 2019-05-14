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

# python3 generate_inputs_sample.py --m 10 --dataset mnist --sigmaw 10.0 --sigmab 10.0 --network fc --prefix test --random_labels --training --number_layers 1
FLAGS = {}
FLAGS['m'] = 100
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

test_images = test_images[:50]
test_ys = test_ys[:50]


X = train_images
Xfull =  np.concatenate([train_images,test_images])
ys2 = [[y] for y in ys]
ysfull = ys2 + [[y] for y in test_ys]
Yfull = np.array(ysfull)
Y = np.array(ys2)

from fc_kernel import kernel_matrix
Kfull = kernel_matrix(Xfull,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab)


# FLAGS["m"] = 1500
#Kfull = load_kernel(FLAGS)
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
samples = s.sample(m, 100, epsilon=1e-2, lmax=15, lmin=5, thin=5, logprobs=False)#, verbose=True)

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

print(loglik_samples)

logPU = np.mean(loglik_samples)
print(logPU)
ps = np.array(ps)
print(ps)
ps.shape
p = np.mean(ps,axis=0)
print(p)
