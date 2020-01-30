import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.keras as keras
import pickle
import os
from math import ceil

from utils import preprocess_flags, save_kernel
from utils import load_data,load_model,load_model_json,load_kernel
from utils import data_folder,kernel_folder,arch_folder

import sys
THIS_DIR = "./"
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
nngp_kernel_folder = os.path.join(ROOT_DIR, 'nngp_kernel')
sys.path.append(nngp_kernel_folder)
import nngp_kernel.deep_ckern as dkern
import tqdm
import pickle_utils as pu
import gpflow
from nngp_kernel.save_kernels import compute_big_K,mnist_1hot_all

def kernel_matrix(X,X2=None,image_size=28,number_channels=1,filter_sizes=[[5, 5], [2, 2], [5, 5], [2, 2]],padding=["VALID", "SAME", "VALID", "SAME"],strides=[[1, 1]] * 4, sigmaw=1.0,sigmab=1.0, n_gpus=1):
    with tf.device("cpu:0"):
        kern = dkern.DeepKernel(
            #[number_channels, image_size, image_size],
            ([number_channels, image_size, image_size] if n_gpus>0 else [image_size,image_size,number_channels]),
            filter_sizes=filter_sizes,
            recurse_kern=dkern.ExReLU(multiply_by_sqrt2=False),
            var_weight=sigmaw**2,
            var_bias=sigmab**2,
            padding=padding,
            strides=strides,
            #data_format="NCHW",
            data_format=("NCHW" if n_gpus>0 else "NHWC"), #but don't need to change inputs dkern transposes the inputs itself apparently :P
            skip_freq=-1, # no residual connections
            )

    return kern

#%%
f = tf.compat.v1.app.flags

from utils import define_default_flags

define_default_flags(f)

FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()

FLAGS.keys()
FLAGS["network"] = "cnn"
FLAGS["prefix"] = "test_"
FLAGS["number_layers"] = 4
FLAGS["dataset"] = "mnist"
FLAGS["m"] = 100
FLAGS["sigmaw"] = 1.41
FLAGS["sigmab"] = 0.1
FLAGS["pooling"] = "none"

FLAGS = preprocess_flags(FLAGS)
globals().update(FLAGS)
#%%

if n_gpus>0:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(1)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

set_session = keras.backend.set_session
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

train_images,flat_train_images,ys,_,_ = load_data(FLAGS)
image_size = train_images.shape[1]
number_channels = train_images.shape[-1]
input_dim = flat_train_images.shape[1]
#%%

kern = kernel_matrix(flat_train_images,image_size=image_size,number_channels=number_channels,filter_sizes=filter_sizes,padding=padding,strides=strides,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)

kern
flat_train_images[0].shape
train_images[0].shape
train_images[0].transpose((2,0,1)).shape
kern.input_shape

test_case = gpflow.test_util.GPflowTestCase()
# test_case.test_context()
from gpflow import settings
s = settings.get_settings()

####### compare output
# type(train_images)
with test_case.test_context() as sess, settings.temp_settings(s):
    with tf.device("cpu:0"):
        kern = kernel_matrix(flat_train_images,image_size=image_size,number_channels=number_channels,filter_sizes=filter_sizes,padding=padding,strides=strides,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)
        X=tf.convert_to_tensor(train_images[0:1].transpose((0,3,1,2)).astype(np.float64))
        tf_y_bnn = kern.equivalent_BNN(X,n_samples=1,n_filters=512)
        output = sess.run(tf_y_bnn)

output

from utils import load_data,load_model,load_kernel,entropy
model = load_model(FLAGS)

model.input_shape
model.predict(train_images[0:1])

####compare per layer

# sess = test_case.test_context().__enter__()
# settings.temp_settings(s).__enter__()
# tf.device("cpu:0").__enter__()

intermediate_outputs = []
intermediate_inputs = []

with test_case.test_context() as sess, settings.temp_settings(s), tf.device("cpu:0"):
    kern = kernel_matrix(flat_train_images,image_size=image_size,number_channels=number_channels,filter_sizes=filter_sizes,padding=padding,strides=strides,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)
    X=tf.convert_to_tensor(train_images[0:1].transpose((0,3,1,2)).astype(np.float64))
    # tf_y_bnn = kern.equivalent_BNN(X,n_samples=1,n_filters=512)
    # output = sess.run(tf_y_bnn)
    n_samples = 1
    n_filters = 512
    self = kern

    if list(map(int, X.shape)) != [1] + self.input_shape:
        raise NotImplementedError("Can only deal with 1 input image")

    # Unlike the kernel, this function operates in NHWC. This is because of
    # the `extract_image_patches` function
    tp_order = np.concatenate([[0], np.arange(2, len(X.shape)), [1]])
    X = tf.transpose(X, tp_order)  # NCHW -> NHWC

    # The name of the first dimension of the einsum. In the first linear
    # transform, it should be "a", to broadcast the "n" dimension of
    # samples of parameters along it. In all other iterations it should be
    # "n".
    first = 'a'
    batch_dim = 1

    for i in range(self.n_layers):
        intermediate_inputs.append(sess.run(X))
        if len(self.filter_sizes[i]) == 0:
            Xp = X
        elif len(self.filter_sizes[i]) == 2:
            h, w = self.filter_sizes[i]
            sh, sw = self.strides[i]
            Xp = tf.extract_image_patches(
                X, [1, h, w, 1], [1, sh, sw, 1], [1, 1, 1, 1],
                self.padding[i])
        else:
            raise NotImplementedError("convolutions other than 2d")

        W, b = self.get_Wb(i, X.shape, n_samples, n_filters)
        equation = "{first:}{dims:}i,nij->n{dims:}j".format(
            first=first, dims="dhw"[4-len(self.input_shape):])

        # We're explicitly doing the convolution by extracting patches and
        # a multiplication, so this flatten is needed.
        W_flat_in = tf.reshape(W, [n_samples, -1, W.shape[-1]])
        X = self.recurse_kern.nlin(tf.einsum(equation, Xp, W_flat_in) + b)
        first = 'n'  # Now we have `n_samples` in the batch dimension
        batch_dim = n_samples
        # print(sess.run(X))
        intermediate_outputs.append(sess.run(X))

    intermediate_inputs.append(sess.run(X))
    W, b = self.get_Wb(self.n_layers, X.shape, n_samples, 1)
    X_flat = tf.reshape(X, [batch_dim, -1])
    Wx = tf.einsum("{first:}i,nij->nj".format(first=first), X_flat, W)
    intermediate_outputs.append(Wx+b)
# return Wx + b

# sess.close()
# sess = test_case.test_context().__exit__(None,None,None)
# settings.temp_settings(s).__exit__(None,None,None)
# tf.device("cpu:0").__exit__(None,None,None)

intermediate_outputs[0].shape

len(model.layers[5].get_weights())

from tensorflow.keras import backend as K
intermediate_outputs_mine = []
for i in range(number_layers+2):
    func = K.function(model.input,model.layers[i].output)
    intermediate_outputs_mine.append(func(train_images[0:1]))
# model.layers[0].output
intermediate_outputs_mine[0].shape

assert len(intermediate_outputs) == len(intermediate_outputs_mine)

for i in range(number_layers+1):
    assert intermediate_outputs_mine[i].shape == intermediate_outputs[i].shape

intermediate_outputs_mine[5].shape
intermediate_outputs[4].shape


###### give them the same weights

model.compile

intermediate_outputs1 = []
intermediate_outputs2 = []
new_weights = []
with test_case.test_context() as sess, settings.temp_settings(s), tf.device("cpu:0"):
    K.set_session(sess)
    model = load_model(FLAGS)
    kern = kernel_matrix(flat_train_images,image_size=image_size,number_channels=number_channels,filter_sizes=filter_sizes,padding=padding,strides=strides,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)
    X=tf.convert_to_tensor(train_images[0:1].transpose((0,3,1,2)).astype(np.float64))
    # W0, b0 = (list(t[0] for t in t_list) for t_list in [kern._W, kern._b])
    # for i in range(number_layers+1):
    #     if i < number_layers:
    #         W,b=kern.get_Wb(i,X_shape=intermediate_inputs[i].shape,n_samples=1,n_filters=512)
    #     else: # last layer has one output
    #         W,b=kern.get_Wb(i,X_shape=intermediate_inputs[i].shape,n_samples=1,n_filters=1)
    #     print(intermediate_inputs[i].shape)
    #     # new_weights += list(sess.run([W,b]))
    #     print(W,b)
    #     b = tf.squeeze(b)
    #     if len(b.shape) == 0:
    #         b = tf.expand_dims(b,0)
    #         print(b)
    #     new_weights += list([sess.run(W[0]),sess.run(b)])
    n_samples = 1
    n_filters = 512
    self = kern

    if list(map(int, X.shape)) != [1] + self.input_shape:
        raise NotImplementedError("Can only deal with 1 input image")

    # Unlike the kernel, this function operates in NHWC. This is because of
    # the `extract_image_patches` function
    tp_order = np.concatenate([[0], np.arange(2, len(X.shape)), [1]])
    X = tf.transpose(X, tp_order)  # NCHW -> NHWC

    # The name of the first dimension of the einsum. In the first linear
    # transform, it should be "a", to broadcast the "n" dimension of
    # samples of parameters along it. In all other iterations it should be
    # "n".
    first = 'a'
    batch_dim = 1

    for i in range(self.n_layers):
        if len(self.filter_sizes[i]) == 0:
            Xp = X
        elif len(self.filter_sizes[i]) == 2:
            h, w = self.filter_sizes[i]
            sh, sw = self.strides[i]
            Xp = tf.extract_image_patches(
                X, [1, h, w, 1], [1, sh, sw, 1], [1, 1, 1, 1],
                self.padding[i])
            # print(Xp)
        else:
            raise NotImplementedError("convolutions other than 2d")

        W, b = self.get_Wb(i, X.shape, n_samples, n_filters)
        # print(W)
        bb = tf.squeeze(b)
        if len(bb.shape) == 0:
            bb = tf.expand_dims(bb,0)
        new_weights += list([sess.run(W[0]),sess.run(bb)])
        equation = "{first:}{dims:}i,nij->n{dims:}j".format(
            first=first, dims="dhw"[4-len(self.input_shape):])

        # We're explicitly doing the convolution by extracting patches and
        # a multiplication, so this flatten is needed.
        # W_flat_in = tf.reshape(W, [n_samples, -1, W.shape[-1]])
        W_flat_in = tf.reshape(new_weights[-2], [n_samples, -1, W.shape[-1]])
        print(W_flat_in)
        X = self.recurse_kern.nlin(tf.einsum(equation, Xp, W_flat_in) + new_weights[-1])
        first = 'n'  # Now we have `n_samples` in the batch dimension
        batch_dim = n_samples
        intermediate_outputs1.append(sess.run(X))
        # print(sess.run(X))
        # break

    W, b = self.get_Wb(self.n_layers, X.shape, n_samples, 1)
    bb = tf.squeeze(b)
    if len(bb.shape) == 0:
        bb = tf.expand_dims(bb,0)
    new_weights += list([sess.run(W[0]),sess.run(bb)])
    X_flat = tf.reshape(X, [batch_dim, -1])
    # Wx = tf.einsum("{first:}i,nij->nj".format(first=first), X_flat, W)
    # W.shape
    # tf.convert_to_tensor(new_weights[-2]).shape
    Wx = tf.einsum("{first:}i,nij->nj".format(first=first), X_flat, tf.convert_to_tensor(np.expand_dims(new_weights[-2],0)))
    # train_images[0:1].shape
    # tf_y_bnn = kern.equivalent_BNN(X,n_samples=1,n_filters=512)
    # output1 = sess.run(Wx+b
    output1 = sess.run(Wx+new_weights[-1])
    # train_images[0:1].shape
    model.set_weights(new_weights)
    for i in range(number_layers+1):
        fun = K.function(model.input,model.layers[i].output)
        intermediate_outputs2.append(fun(train_images[0:1]))
    output2 = model.predict(train_images[0:1])
        # print(sess.run(W).shape)

output1
output2

import imp; import initialization; imp.reload(initialization)
from initialization import get_all_layers, is_normalization_layer, reset_weights

layers = get_all_layers(model)
are_norm = [is_normalization_layer(l) for l in layers for w in l.get_weights()]
initial_weights = model.get_weights()
reset_weights(model, initial_weights, are_norm, sigmaw, sigmab, truncated_init_dist)

model.layers[1].get_config()

np.std(model.layers[1].get_weights()[0])

model.layers[0].get_weights()[0].shape
np.std(new_weights[0])

##########

sess = tf.Session()
K.set_session(sess)
K.set_floatx('float64')
model = load_model(FLAGS)
model.layers[3].get_weights()[1].shape
model.get_weights()[9].shape

model.layers[0].kernel_size

# sess.run(self.recurse_kern.nlin(-10))

# model.layers[0]

intermediate_outputs1[0].shape
intermediate_outputs2[0].shape

intermediate_outputs1[0] == intermediate_outputs2[0]

# type(train_images)
# type(flat_train_images)

# model.dtyle

model.layers[0].set_weights([W,b])
model.layers[0].get_weights()[0]

equation = "{first:}{dims:}i,nij->n{dims:}j".format(
    first="a", dims="dhw"[4-len(self.input_shape):])

# We're explicitly doing the convolution by extracting patches and
# a multiplication, so this flatten is needed.

W == model.layers[0].get_weights()[0]

i=0
h, w = self.filter_sizes[i]
sh, sw = self.strides[i]
X=tf.convert_to_tensor(train_images[0:1].transpose((0,3,1,2)).astype(np.float64))
tp_order = np.concatenate([[0], np.arange(2, len(X.shape)), [1]])
X = tf.transpose(X, tp_order)  # NCHW -> NHWC
Xp = tf.extract_image_patches(
    X, [1, h, w, 1], [1, sh, sw, 1], [1, 1, 1, 1],
    self.padding[i])
W,b = new_weights[0],new_weights[1]
W.shape
W_flat_in = tf.reshape(W, [n_samples, -1, W.shape[-1]])
X = self.recurse_kern.nlin(tf.einsum(equation, Xp, W_flat_in) + b)

# train_images.shape


# first_layer = keras.layers.Conv2D(512,(h,w),(sh,sw),self.padding[i],activation="relu")
# del first_layer

model.layers[0].get_config()

sess.close()
first_layer = keras.layers.Conv2D(512,(h,w),(sh,sw),self.padding[i],activation="relu")

first_layer.get_weights()

first_layer.get_config()

first_layer.set_weights([W,b])

sess.run(first_layer(train_images[0:1]))

model.layers[0].set_weights([W,b])

sess.run(model.layers[0](train_images[0:1]))

fun = K.function(model.input,model.layers[0].output)
fun(train_images[0:1])

sess.run(X)


### fan_in was wrong. I needed to do weight.shape[-2] rather than weight.shape[-1]. This only mattered for the first layer.
