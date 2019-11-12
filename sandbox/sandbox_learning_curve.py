
import numpy as np
import pandas as pd
net ="resnet50"
filename = "newer_arch_sweep_ce_sgd__"+net+"_EMNIST_1000_0.0_0.0_True_False_False_False_-1_True_False_False_data.h5"

from utils import load_data_by_filename
train_images,flat_data,ys,test_images,test_ys = load_data_by_filename("data/"+filename)
#%%

input_dim = train_images.shape[1]
num_channels = train_images.shape[-1]
# tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
# train_images = tf.constant(train_images)
tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
flat_data = np.transpose(train_images, tp_order)  # NHWC -> NCHW # this is because the cnn GP kernels assume this (tho we are not calculating kernels here so meh)
X = np.stack([x.flatten() for x in train_images])
X_test = np.stack([x.flatten() for x in test_images])

test_images = test_images[:500]
test_ys = test_ys[:500]

Xfull =  np.concatenate([X,X_test])
ys2 = [[y] for y in ys]
ysfull = ys2 + [[y] for y in test_ys]
Yfull = np.array(ysfull)
Y = np.array(ys2)

#%%

# filename = net+"_KMNIST_1000_0.0_0.0_True_False_True_4_3.0_0.0_None_0000_max_kernel.npy"
filename = "newer_arch_sweep_ce_sgd__"+str(net)+"_EMNIST_1000_0.0_0.0_True_False_True_4_1.414_0.0_None_0000_max_kernel.npy"

from utils import load_kernel_by_filename
Kfull = load_kernel_by_filename("kernels/"+filename)
# m = 1000
# Kfull.max()

# K = Kfull/Kfull.max()
# K *= 1000

eigs = np.linalg.eigh(K)[0]

eigs

m=1000
epsilon = 1
sigma=1e-1
for i in range(1000):
    epsilon = np.sum(eigs/(1+eigs*m/(sigma**2+epsilon)))

epsilon
