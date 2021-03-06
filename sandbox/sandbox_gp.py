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
# python3 generate_inputs_sample.py --m 1000 --dataset mnist --sigmaw 1.41 --sigmab 0.0 --network fc --prefix test --training --number_layers 2
FLAGS = {}
FLAGS['m'] = 10000
FLAGS['dataset'] =  "mnist"
FLAGS['network'] =  "fc"
FLAGS['number_inits'] = 1
FLAGS['label_corruption'] =  0.0
FLAGS['confusion'] = 0.0
FLAGS['binarized'] =  True
FLAGS['number_layers'] =  2
FLAGS['layer_widths'] =  "512"
FLAGS['activations'] =  "relu"
FLAGS['pooling'] =  "none"
FLAGS['intermediate_pooling'] =  "0"*FLAGS['number_layers']
FLAGS['intermediate_pooling_type'] =  "max"
FLAGS['init_dist'] =  "gaussian"
FLAGS['sigmaw'] =  1.41
FLAGS['sigmab'] =  0.0
FLAGS['prefix'] =  "test"
FLAGS['whitening'] =  False
FLAGS['random_labels'] =  True
FLAGS['training'] =  True
FLAGS['no_training'] =  False
FLAGS['whitening'] =  False
FLAGS['centering'] =  False
FLAGS['channel_normalization'] =  False
FLAGS['random_labels'] =  True
FLAGS['training'] =  True
FLAGS['no_training'] =  False
FLAGS['threshold'] =  -1
FLAGS['oversampling'] =  False
FLAGS['oversampling2'] =  False
FLAGS['nn_random_labels'] =  False
FLAGS['nn_random_regression_outputs'] =  False
FLAGS['n_gpus'] =  0

#%%

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

plt.imshow(train_images[0].reshape(28,28))

# Y.shape
Xfull.shape

X = train_images
Xfull =  np.concatenate([train_images,test_images])
ys2 = [[y] for y in ys]
ysfull = ys2 + [[y] for y in test_ys]
Yfull = np.array(ysfull)
Y = np.array(ys2)

from nngp_kernel.fc_kernel import kernel_matrix
Kfull = kernel_matrix(Xfull,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab)

# FLAGS["m"] = 1500
#Kfull = load_kernel(FLAGS)
K = Kfull[0:m,0:m]

#%%

# filename=kernel_folder
# for flag in ["network","dataset","m","confusion","label_corruption","binarized","whitening","random_labels","number_layers","sigmaw","sigmab"]:
#     filename+=str(FLAGS[flag])+"_"
# filename += "kernel.npy"
# np.save(open(filename,"wb"),Kfull)
#

import GPy
from GP_prob.custom_kernel_matrix.custom_kernel_matrix import CustomMatrix
lik = GPy.likelihoods.Bernoulli()
inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=False)
m = GPy.core.GP(X=X,
                Y=Y,
                kernel=CustomMatrix(Xfull.shape[1],Xfull,Kfull),
                inference_method=inference_method,
                likelihood=lik)


# m.predict(test_images[0:1])
m.predict(test_images)[0]>0.5

inference_method = GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference()
lik=GPy.likelihoods.gaussian.Gaussian(variance=0.002)
m = GPy.core.GP(X=X,
                Y=Y,
                kernel=CustomMatrix(Xfull.shape[1],Xfull,Kfull),
                inference_method=inference_method,
                likelihood=lik)

# mean, cov = m.predict(test_images,full_cov=True)
#
# inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=False)
# linkfun = GPy.likelihoods.link_functions.Heaviside()
# lik = GPy.likelihoods.Bernoulli(linkfun)
# m = GPy.core.GP(X=test_images,
#                 Y=np.array([[y] for y in test_ys]),
#                 kernel=CustomMatrix(test_images.shape[1],test_images,cov),
#                 inference_method=inference_method,
#                 mean_function=CustomMean(test_images,mean),
#                 likelihood=lik)
#
# class CustomMean(GPy.core.Mapping):
#     def __init__(self,X,means):
#         GPy.core.Mapping.__init__(self, input_dim=X.shape[1], output_dim=1, name="custom_means")
#         self.X = X
#         self.means = means
#         # self.link_parameter(GPy.core.parameterization.Param('means', means))
#     def f(self,X):
#         indices = np.concatenate([np.nonzero(np.prod(self.X == x,1))[0] for x in X])
#         if np.all(np.isin(X,self.X)):
#             if len(indices) != X.shape[0]:
#                 raise NotImplementedError("Some elements of X appear more than once in self.X")
#             else:
#                 return self.means[indices]
#         else:
#             raise NotImplementedError("Some elements of X are not in self.X")
#
#     def update_gradients(self, dL_dF, X):
#         # self.means.gradient = dL_dF.sum(0)
#         pass
#
#     def gradients_X(self, dL_dF, X):
#         return np.zeros_like(X)
#
# m.log_likelihood()

from GP_prob.nngp_mse_heaviside_posterior import nngp_mse_heaviside_posteror_logp

import imp; import GP_prob; imp.reload(GP_prob.nngp_mse_heaviside_posterior)

nngp_mse_heaviside_posteror_logp(X,Y,test_images,np.array([[y] for y in test_ys]),10000*Kfull)

# m.predict(test_images[0:1])
m.predict(test_images)[0]>0.5

########################################################

#%%

from utils import load_data,load_model,load_kernel

arch_json_string = load_model(FLAGS)

from tensorflow.keras.models import model_from_json
model = model_from_json(arch_json_string) # this resets the weights (makes sense as the json string only has architecture)

reset_weights(model)
np.std(model.layers[0].layers[2].get_weights()[0])

model.layers[0].layers[2].kernel_initializer

np.unique([str(type(l)) for l in get_all_layers(model)])
np.unique([str(l.kernel_initializer) for l in get_all_layers(model) if hasattr(l,"kernel_initializer")])

weights2 = sum([l.get_weights() for l in get_all_layers(model)],[])

for i,w in enumerate(weights):
    if np.all(w != weights2[i]):
        print("fail")
        break

weights = model.get_weights()

model.layers[0].layers[3].get_weights()[0]

tf.python.keras.layers.normalization.

isinstance(model.layers[0].layers[3],tf.python.keras.layers.normalization)

isinstance(model.layers[0].layers[0],tf.python.keras.engine.training.Model)

def get_all_layers(model):
    layers = []
    for layer in model.layers:
        if isinstance(layer,tf.python.keras.engine.training.Model):
            layers += get_all_layers(layer)
        else:
            layers += [layer]
    return layers

def is_normalization_layer(l):
    return isinstance(l,tf.python.keras.layers.normalization.BatchNormalization) or isinstance(l,tf.python.keras.layers.normalization.LayerNormalization)

sum([l.get_weights() for l in get_all_layers(model) if not is_normalization_layer(l)],[])

model.layers[0].layers[3].get_weights()

[w.shape for w in weights]

np.std(initialize_var(model.get_weights()[0].shape))

np.std((np.sqrt(2)/np.sqrt(np.prod(model.get_weights()[0].shape[:-1])))*truncnorm.rvs(-1.4,1.4,size=model.get_weights()[0].shape))

from scipy.stats import truncnorm
def reset_weights(model):
    initial_weights = model.get_weights()
    def initialize_var(shape):
        if len(shape) == 1:
            #return tf.random.normal(shape,stddev=sigmab).eval(session=sess)
            return np.random.normal(0,sigmab,shape)
        else:
            #return tf.random.normal(shape,stddev=1.0/np.sqrt(np.prod(shape[:-1]))).eval(session=sess)
            #return np.random.normal(0,1.0/np.sqrt(np.prod(shape[:-1])),shape)
            #return np.random.normal(0,sigmaw/np.sqrt(shape[-2]),shape) #assumes NHWC so that we divide by number of channels as in GP limit
            return (sigmaw/np.sqrt(np.prod(shape[:-1])))*truncnorm.rvs(-np.sqrt(2),np.sqrt(2),size=shape) #assumes NHWC so that we divide by number of channels as in GP limit


    for l in get_all_layers(model):
        if is_normalization_layer(l):
            # new_weights += l.get_weights()
            pass
        else:
            new_weights = []
            for w in l.get_weights():
                new_weights.append(initialize_var(w.shape))
            l.set_weights(new_weights)
    # new_weights = [initialize_var(w.shape) for w in initial_weights]
    # new_weights = [k_eval(lecun_normal()(w.shape)) for w in initial_weights]
    # model.set_weights(new_weights)
