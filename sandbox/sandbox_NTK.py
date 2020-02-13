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
#%%
##
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
########################
######## exploring Jacobians with TF/Keras
#%%


import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian
input = tf.keras.layers.Input(model.input.shape[1:],batch_size=25)
batch_shape = tuple(model.input.shape)
(25,)+batch_shape[1:]
batch_shape[0]
import tensorflow.keras.backend as K
# K.function(input,model.output)
import tensorflow as tf
from tensorflow.keras.models import Model
model2 = Model(input,model(input))
weights = tf.concat([tf.reshape(x,(-1,)) for x in model2.trainable_weights],0)
weights
X.shape
X[:25].shape
output = model2(train_images[:25])
sess = tf.Session()
sess.run(tf.gradients(model2.output,weights),feed_dict={model2._feed_inputs:train_images[:25]})
g = jacobian(model2.output,weights)
g
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# weights = tf.concat([tf.reshape(x,(-1,)) for x in model.trainable_weights],0)
# thing = K.gradients(model2.output,model2.trainable_weights)
# thing = K.gradients(model2.output,model2.trainable_weights)
tf.gra
model2.output
sess.run(tf.global_variables_initializer())
train_images[:25].shape
g = sess.run(thing,feed_dict={model2.input:train_images[:25]})
g[0].shape
len(g)

# g = tf.GradientTape().__enter__()
# model.variables
# model.trainable_weights[0]
# g.watch(model.variables)
# tf.GradientTape().__exit__()

# model.evaluate()
# model.layers[0].layers[6].updates
model.optimizer.get_gradients
# thing.updates

from tensorflow.python.ops import gradients

#alternatively try pytorch and cross validate with TF results
#aaah apparently there's no way, but GradientTape.jacobian does have it, it just doesnt support tf.conds!!! adkjshja
grads = [gradients.gradients(model2.output[i],model2.variables) for i in range(25)]
grads = [gradients.gradients(model(train_images[i:i+1]),model.variables) for i in range(25)]

sess = tf.Session()
sess.run(grads, feed_dict={model2.input:train_images[:25]})

grads = gradients.gradients(tf.expand_dims(tf.tile(model(train_images[:25]),tf.constant([1,25], tf.int32)),0),model.variables, grad_ys=tf.eye(25))

grads

grads[0].shape

tf.app.flags.DEFINE_string('f', '', 'kernel')

model.build()

f = K.function(model.input,model.output,)
model.trainable = False
model.learning_phase
model.layers[0].layers[3]
jacobian(model(train_images[:25]),model.variables)
jacobian(model.predict(train_images[:25]),model.variables)

with tf.GradientTape() as g:
    g.watch(model.variables)
    y = model(train_images[:25])

g.jacobian(y, model.variables)


import tensorflow as tf
from tensorflow.keras.models import Sequential
import tensorflow.keras as keras

class BatchNormLayer(keras.layers.Layer):
    def __init__(self):
        super(BatchNormLayer, self).__init__()
        self.batch_norm_layer = tf.compat.v1.layers.BatchNormalization()
    def call(self, inputs):
        return self.batch_norm_layer(inputs, training=True)

# sess = tf.compat.v1.Session()
# sess.close()
model3 = Sequential()
model3.add(keras.layers.Dense(100))
# model3.add(keras.layers.BatchNormalization())
# model3.add(keras.layers.Lambda(tf.compat.v1.layers.BatchNormalization()))
# model3.add(keras.layers.Lambda(tf.compat.v1.layers.BatchNormalization()))
# model3.add(keras.layers.InputLayer(input_tensor=tf.compat.v1.layers.BatchNormalization(model3.layers[0].output)[0]))
model3.add(BatchNormLayer())
model3.add(keras.layers.Dense(1))

thingy = keras.layers.BatchNormalization()

tf.__version__

foo=tf.compat.v1.layers.batch_normalization(training=True)
# sess.run(foo(x))
eee = tf.compat.v1.layers.batch_normalization(x,training=True)
eee = tf.compat.v1.layers.batch_normalization(x)
eeee = tf.compat.v1.layers.BatchNormalization()
eeee.variables
# np.mean(sess.run(eee),0)
np.mean(sess.run(eeee(x,training=True)),0)

model3.layers[1].get_weights()

thingy.get_weights()

thingy.variables

sess.run(tf.compat.v1.local_variables_initializer())
sess.run(tf.compat.v1.global_variables_initializer())
aaa=tf.compat.v1.layers.BatchNormalization()
# tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, model3.layers[1].updates)
tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.UPDATE_OPS, thingy.updates)
np.mean(sess.run(aaa(x)),0)
# x=tf.random.normal((10,100))
x=tf.constant(tf.random.normal((10,100)))
import numpy as np
x=np.random.randn(10,100).astype(np.float32)
np.mean(x,0)
# model3(x)
np.mean(sess.run(thingy(x)),0)
model3.layers[0].input
model3.layers[1].variables
np.mean(sess.run(model3.layers[1](x)),0)
sess.run(model3.layers[1](x))
sess.run(tf.compat.v1.local_variables_initializer())
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(thingy(x))


y=model3(X[:25])
sess.run(y)

tf.compat.v1.disable_eager_execution()

from tensorflow.python.ops.parallel_for.gradients import jacobian
g=jacobian(model3(X[:25]),model3.variables)
g
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
sess.run([thing for thing in g if thing is not None])


#### ok here we go

isinstance(model.layers[0].layers[3], tf.keras.layers.BatchNormalization)
isinstance(model.layers[0], tf.keras.models.Model)
# import tensorflow.python as tfp
# isinstance(model.layers[0].layers[3],tf.python.keras.layers.normalization.BatchNormalization)

def find_bn_layers(model):
    bn_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            bn_layers.append(layer)
        elif isinstance(layer, tf.keras.models.Model):
            bn_layers += find_bn_layers(layer)
    return bn_layers

find_bn_layers(model)

#%%
import tensorflow.keras as keras
class BatchNormLayer(keras.layers.Layer):
    def __init__(self, name="BatchNormLayer"):
        super(BatchNormLayer, self).__init__(name=name)
        self.batch_norm_layer = tf.compat.v1.layers.BatchNormalization()
    def call(self, inputs):
        return self.batch_norm_layer(inputs, training=True)


from tensorflow.keras.models import Model

#based on https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
def replace_layer(model, layer_class, insert_layer_factory,
                        insert_layer_name=None, position='after', verbose=0):

    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                if layer.name not in network_dict['input_layers_of'][layer_name]:
                    network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer

    # Iterate over all layers after the input
    input_tensor = None
    if isinstance(model.layers[0],keras.layers.InputLayer):
        layers_to_iterate = model.layers[1:]
        network_dict['new_output_tensor_of'].update(
                {model.layers[0].name: model.input})
        input_tensor = model.input
    else:
        layers_to_iterate = model.layers
        input_tensor = keras.Input(model.input.shape[1:])
    for layer in layers_to_iterate:
        if verbose == 1:
            # print(layer)
            print(network_dict)

        # Determine input tensors
        if layer.name in network_dict['input_layers_of']:
            layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                    for layer_aux in network_dict['input_layers_of'][layer.name]]
        else:
            # print("hi")
            # layer_input = keras.Input(model.input.shape)
            layer_input = [input_tensor]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Insert layer if name matches the regular expression
        if isinstance(layer, layer_class):
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            elif position == 'before':
                pass
            else:
                raise ValueError('position must be: before, after or replace')

            new_layer = insert_layer_factory(layer.name)
            # if insert_layer_name:
            #     new_layer.name = insert_layer_name
            # else:
            #     new_layer.name = '{}_{}'.format(layer.name,
            #                                     new_layer.name)
            # new_layer.name = layer.name
            x = new_layer(x)
            # if position == 'after':
            #     print('Layer {} inserted after layer {}'.format(new_layer.name,
            #                                                 layer.name))
            # else:
            #     print('Layer {} replacing layer {}'.format(new_layer.name,
                                                            # layer.name))
            if position == 'before':
                x = layer(x)
        elif isinstance(layer, tf.keras.models.Model):
            new_layer = replace_layer(layer, layer_class, insert_layer_factory, insert_layer_name=insert_layer_name, position=position,verbose=0)
            x = new_layer(layer_input)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer)
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=input_tensor, outputs=x, name=model.name)

#%%

model.layers[0].name
model.layers[0] == model.input
isinstance(model.layers[0].layers[0],tf.keras.layers.InputLayer)
model.layers[0].layers[0] == model.layers[0].input

new_model = replace_layer(model,keras.layers.BatchNormalization,BatchNormLayer,position="replace", verbose=1)


model.layers[0].layers[0].outbound_nodes[0].outbound_layer.name
model.layers[0].layers[1].inbound_nodes[0].inbound_layers

y=model(train_images[:25])
y=new_model(train_images[:25])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(y)

import tensorflow as tf
import pydot as pyd
from IPython.display import SVG

import keras
keras.utils.pydot = pyd
keras.utils.dot = dot
import dot
keras.utils.plot_model(model, to_file="model.png")

tf.keras.utils.model_to_dot(model)
from IPython.display import SVG
from keras.utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

model.layers[0].layers

#####

from tensorflow.python.ops.parallel_for.gradients import jacobian
tf.app.flags.DEFINE_string('f', '', 'kernel')
g=jacobian(new_model(train_images[:25]),new_model.variables)


sess.run(g)
