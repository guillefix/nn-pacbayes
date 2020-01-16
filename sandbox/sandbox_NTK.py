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
FLAGS['layer_widths'] =  "1024"
FLAGS['pooling'] =  "none"
FLAGS['intermediate_pooling'] =  "0000"
FLAGS['sigmaw'] =  10.0
FLAGS['sigmab'] =  10.0
FLAGS['network'] =  "fc"
FLAGS['prefix'] =  "test"
FLAGS['whitening'] =  False
FLAGS['centering'] =  False
FLAGS['centering'] =  False
FLAGS['random_labels'] =  True
FLAGS['training'] =  True
FLAGS['no_training'] =  False
FLAGS['nn_random_labels'] =  False
FLAGS['nn_random_regression_outputs'] =  False
FLAGS['activations'] =  "relu"

from utils import preprocess_flags
FLAGS = preprocess_flags(FLAGS)
globals().update(FLAGS)
#%%

# net="resnet50"
net="mobilenetv2"
# net="nasnet"
# net="vgg19"
# net="densenet121"
# net="densenet169"
things = []
# for net in ["densenet121","densenet169","densenet201","mobilenetv2","nasnet","resnet50","vgg16","vgg19"]:
filename = net+"_KMNIST_1000_0.0_0.0_True_False_False_False_-1_True_False_False_data.h5"
# filename = "fc_boolean_50_0.0_0.0_True_False_False_False_1_True_False_False_84.0_data.h5"

from utils import load_data_by_filename
train_images,flat_data,ys,test_images,test_ys = load_data_by_filename("data/"+filename)
#%%

# from utils import load_data,load_model,load_kernel
# train_images,flat_train_images,ys,test_images,test_ys = load_data(FLAGS)
input_dim = train_images.shape[1]
num_channels = train_images.shape[-1]
# tp_order = np.concatenate([[0,len(train_images.shape)-1], np.arange(1, len(train_images.shape)-1)])
# train_images = tf.constant(train_images)
X = np.stack([x.flatten() for x in train_images])
X_test = np.stack([x.flatten() for x in test_images])

test_images = test_images[:500]
test_ys = test_ys[:500]

#%%

Xfull =  np.concatenate([X,X_test])
ys2 = [[y] for y in ys]
ysfull = ys2 + [[y] for y in test_ys]
Yfull = np.array(ysfull)
Y = np.array(ys2)
#
# from fc_kernel import kernel_matrix
# Kfull = kernel_matrix(Xfull,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab)




'''CODE FOR EMPIRICAL NTK CALCULATION'''

#%%
# net="resnet50"
net="mobilenetv2"
# net="nasnet"
# net="vgg19"
# net="densenet121"
filename = net+"_True_4_None_0000_max_gaussian_model"
json_string_filename = filename
arch_json_string = open("archs/"+filename, "r") .read()
from tensorflow.keras.models import model_from_json
model = model_from_json(arch_json_string)
model.compile("sgd",loss=lambda target, pred: pred)
import tensorflow.keras.backend as K

num_layers = len(model.trainable_weights)
trainable_weights = model.trainable_weights
# num_layers

fs = []

num_chunks = 5
layers_per_chunk = num_layers//num_chunks
for layer in range(num_chunks):
    print(layer)
    grads = model.optimizer.get_gradients(model.total_loss, trainable_weights[layer*layers_per_chunk:(layer+1)*layers_per_chunk])
    # symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets)
    f = K.function(symb_inputs, grads)
    # x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    fs.append(f)
if num_layers%num_chunks != 0:
    num_chunks += 1
    grads = model.optimizer.get_gradients(model.total_loss, trainable_weights[(num_chunks-1)*layers_per_chunk:])
    # symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets)
    f = K.function(symb_inputs, grads)
    # x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    fs.append(f)

# grads = model.optimizer.get_gradients(model.total_loss, trainable_weights)
# # symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
# symb_inputs = (model._feed_inputs + model._feed_targets)
# f = K.function(symb_inputs, grads)
# # x, y, sample_weight = model._standardize_user_data(inputs, outputs)


# def get_weight_grad(model, inputs, outputs):
def get_weight_grad(model, inputs, outputs, layer_chunk_index):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    # output_grad = f(x + y + sample_weight)
    x, y, _= model._standardize_user_data(inputs, outputs)
    output_grad = fs[layer_chunk_index](x + y)
    # output_grad = f(x + y)
    output_grad = np.concatenate([x.flatten() for x in output_grad])
    return output_grad

# chunk1 = 10
# chunk2 = 10 # it's benefitial to chunk in j2 too, in orden to reduce the python for loop. Even though we do more on numpy/pytorch (by reducing the chunking on j1, we do more grad computaiotns), python is much slower than those, and so tradeoff is worth it I think
# gradient = get_weight_grad(model, train_images[0*chunk1+0:0*chunk1+0+1], Y[0*chunk1+0:0*chunk1+0+1])
#
# gradient = np.concatenate([x.flatten() for x in gradient])

params_per_layer = [np.prod(x.shape) for x in trainable_weights]
if num_layers%num_chunks != 0:
    params_per_chunk = [sum(params_per_layer[layer*layers_per_chunk:(layer+1)*layers_per_chunk]) for layer in range(num_chunks-1)] + [sum(params_per_layer[(num_chunks-1)*layers_per_chunk:])]
else:
    params_per_chunk = [sum(params_per_layer[layer*layers_per_chunk:(layer+1)*layers_per_chunk]) for layer in range(num_chunks)]
# gradient.shape
tot_parameters = np.sum(params_per_layer)
# np.concatenate([x.flatten() for x in model.trainable_weights]).shape

#%%

# X.shape
# X = X[:100,:]
# import cupy as cp
NTK = np.zeros((len(X),len(X)))
# NTK = cp.zeros((len(X),len(X)))
chunk1 = 500
chunk2 = 500 # it's benefitial to chunk in j2 too, in orden to reduce the python for loop. Even though we do more on numpy/pytorch (by reducing the chunking on j1, we do more grad computaiotns), python is much slower than those, and so tradeoff is worth it I think
# for layer in range(number_layers):
for layer_chunk_index in range(num_chunks):
    print(layer_chunk_index)
    jac1 = np.zeros((chunk1,params_per_chunk[layer_chunk_index]))
    jac2 = np.zeros((chunk2,params_per_chunk[layer_chunk_index]))
    # jac1 = np.zeros((chunk1,tot_parameters))
    # jac2 = np.zeros((chunk2,tot_parameters))
    for j1 in range(len(X)//chunk1):
        for i in range(chunk1):
            # print(i)
            # gradient = get_weight_grad(model, train_images[j1*chunk1+i:j1*chunk1+i+1], Y[j1*chunk1+i:j1*chunk1+i+1])
            gradient = get_weight_grad(model, train_images[j1*chunk1+i:j1*chunk1+i+1], Y[j1*chunk1+i:j1*chunk1+i+1], layer_chunk_index)
            jac1[i,:] = gradient
        for j2 in range(j1,len(X)//chunk2):
            print(j1,j2)
            for i in range(chunk2):
                # print(i)
                # gradient = get_weight_grad(model, train_images[j2*chunk2+i:j2*chunk2+i+1], Y[j2*chunk2+i:j2*chunk2+i+1])
                gradient = get_weight_grad(model, train_images[j2*chunk2+i:j2*chunk2+i+1], Y[j2*chunk2+i:j2*chunk2+i+1], layer_chunk_index)
                jac2[i,:] = gradient
            # NTK[j1*chunk1:(j1+1)*chunk1,j2*chunk2:(j2+1)*chunk2] += cp.matmul(jac1,jac2.T)
            NTK[j1*chunk1:(j1+1)*chunk1,j2*chunk2:(j2+1)*chunk2] += np.matmul(jac1,jac2.T)


NTK = (NTK+NTK.T)/2

filename = net+"_KMNIST_1000_0.0_0.0_True_False_False_False_-1_True_False_False_NTK.npy"
np.save(filename, NTK)
"hi"
#%%
##############################


import numpy as np
np.random.rand(6000000,100).shape
