
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

#%%
