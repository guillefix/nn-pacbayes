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
FLAGS['number_inits'] = 24
FLAGS['label_corruption'] =  0.0
FLAGS['confusion'] = 0.0
# FLAGS['dataset'] =  "mnist"
FLAGS['dataset'] =  "EMNIST"
FLAGS['binarized'] =  True
FLAGS['number_layers'] =  1
FLAGS['pooling'] =  "none"
FLAGS['intermediate_pooling'] =  "0000"
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

#%%

x = flat_train_images
x.shape
# x[0].min()

import matplotlib.pyplot as plt
C=np.matmul(x.T,x)

# x.mean(axis=0).reshape(28,28)[15,14]
# x.mean(axis=0).max()
plt.matshow(x.mean(axis=0).reshape(28,28))

plt.matshow(C)

plt.matshow(C[100:140,100:140])
