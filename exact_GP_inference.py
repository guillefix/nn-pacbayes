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

test_images = test_images[:50]
test_ys = test_ys[:50]


X = train_images
Xfull =  np.concatenate([train_images,test_images])
ys2 = [[y] for y in ys]
ysfull = ys2 + [[y] for y in test_ys]
Yfull = np.array(ysfull)
Y = np.array(ys2)

Xfull.shape

from nngp_kernel.fc_kernel import kernel_matrix
number_layers = 2
Kfull = kernel_matrix(Xfull,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab)


# FLAGS["m"] = 1500
# Kfull = load_kernel(FLAGS)
K = Kfull[0:m,0:m]

#%%

exact_samples = np.random.multivariate_normal(np.zeros(m+50),Kfull,int(1e7))>0

# Y_extended = np.concatenate([Y.T[0,:],np.ones(50)])==1
fits_data = np.prod(~(exact_samples[:,:50]^(Y.T==1)),1)

indices = np.where(fits_data)

generrors = np.sum(~(exact_samples[indices,:][0,:,50:]^(Yfull.T[0,50:]==1)),1)/50.0
np.mean(generrors)
