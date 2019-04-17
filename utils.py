import numpy as np
import tensorflow as tf
from math import ceil

def load_data(FLAGS):
    import h5py
    data_folder = "data/"
    filename=data_folder
    for flag in ["network","dataset","m","confusion","label_corruption","binarized","whitening","random_labels"]:
        filename+=str(FLAGS[flag])+"_"
    filename += "data.h5"
    h5f = h5py.File(filename,'r')
    train_images = h5f['train_images'][:]
    if FLAGS["training"]:
        ys = h5f['ys'][:]
        test_images = h5f['test_images'][:]
        test_ys = h5f['test_ys'][:]
    h5f.close()
    data = train_images
    tp_order = np.concatenate([[0,len(data.shape)-1], np.arange(1, len(data.shape)-1)])
    flat_data = np.transpose(data, tp_order)  # NHWC -> NCHW # this is because the cnn GP kernels assume this
    flat_data = np.array([train_image.flatten() for train_image in flat_data])
    return train_images,flat_data,ys,test_images,test_ys

def load_model(FLAGS):
    arch_folder = "archs/"
    filename=arch_folder
    for flag in ["network","binarized","number_layers","pooling","intermediate_pooling"]:
        filename+=str(FLAGS[flag])+"_"
    filename += "model"
    json_string_filename = filename
    arch_json_string = open(filename, "r") .read()
    return arch_json_string

def reset_weights(model):
    initial_weights = model.get_weights()
    def initialize_var(shape):
        if len(shape) == 1:
           #return tf.random.normal(shape).eval(session=sess)
           return np.random.normal(0,1,shape)
        else:
            return np.random.normal(0,1.0/np.sqrt(np.prod(shape[:-1])),shape)
    new_weights = [initialize_var(w.shape) for w in initial_weights]
    model.set_weights(new_weights)

def load_kernel(FLAGS):
    kernel_folder = "kernels/"
    filename=kernel_folder
    for flag in ["network","dataset","m","confusion","label_corruption","binarized","whitening","random_labels","number_layers","sigmaw","sigmab"]:
        filename+=str(FLAGS[flag])+"_"
    filename += "kernel.npy"
    K = np.load(filename,"r")
    return K

def define_default_flags(f):
    f.DEFINE_integer('m',None,"Number of training examples")
    f.DEFINE_float('label_corruption', 0.0, "Fraction of corrupted labels")
    f.DEFINE_float('confusion',0.0,"Number of confusion samples to add to training data, as a fraction of m")
    f.DEFINE_string('dataset', None, "The dataset to use")
    f.DEFINE_string('network', None, "The type of network to use")
    f.DEFINE_integer('number_layers', None, "The number of layers in the network")
    f.DEFINE_boolean('binarized', True, "Whether to convert classification labels to binary")
    f.DEFINE_string('pooling', "none", "The pooling type to use")
    f.DEFINE_string('intermediate_pooling', "0000", "Whether invidiaual layers have a local maxpooling after them; 1 is maxpool; 0 no maxpool")
    # f.DEFINE_integer('number_inits',1,"Number of initializations")
    f.DEFINE_float('sigmaw', 1.0, "The variance parameter of the weights; their variance will be sigmaw/sqrt(number of inputs to neuron")
    f.DEFINE_float('sigmab', 1.0, "The variance of the biases")
    f.DEFINE_boolean('compute_bound', False, "Whether to compute the PAC-Bayes bound or just generate the training data")
    #f.DEFINE_boolean('compute_kernel', False, "Whether to compute the kernel or just generate the training data")
    f.DEFINE_boolean('whitening', False, "Whether to perform ZCA whitening and normalization on the training data")
    f.DEFINE_boolean('no_training', False, "Whether to also generate labels and test data")
    f.DEFINE_boolean('use_empirical_K', False, "Whether to use the empirical kernel matrix (from sampling) or the analytical one")
    f.DEFINE_integer('n_gpus', 1, "Number of GPUs to use")
    f.DEFINE_float('n_samples_repeats', 1.0, "Number of samples to compute empirical kernel, as a multiple of training set size, m")
    f.DEFINE_boolean('random_labels', True, "Whether the confusion data is constructed by randomizing the labels, or by taking a wrong label")
    f.DEFINE_string('prefix', "", "A prefix to use for the result files")

def preprocess_flags(FLAGS):

    if FLAGS["intermediate_pooling"] == "0000":
        FLAGS["intermediate_pooling"] = "0"*FLAGS["number_layers"]

    if FLAGS["number_layers"] != len(FLAGS["intermediate_pooling"]):
        raise ValueError("length of intermediate_pooling binary string should be the same as the number of layers; you are providing whether invidiaual layers have a local maxpooling after them; 1 is maxpool; 0 no maxpool")

    # if FLAGS["compute_bound"]:
    #     FLAGS["compute_kernel"] = True
    #     FLAGS["training"] = True

    number_layers = FLAGS["number_layers"]
    intermediate_pooling = FLAGS["intermediate_pooling"]
    confusion = FLAGS["confusion"]
    m = FLAGS["m"]
    FLAGS["filter_sizes"] = [[5,5],[2,2]]*10
    FLAGS["filter_sizes"] = FLAGS["filter_sizes"][:number_layers]
    FLAGS["padding"]=["VALID", "SAME"]*10
    FLAGS["padding"]= FLAGS["padding"][:number_layers]
    FLAGS["pooling_in_layer"] = [x=="1" for x in intermediate_pooling]
    FLAGS["strides"]=[[1, 1]] * 10
    FLAGS["strides"]= FLAGS["strides"][:number_layers]
    FLAGS["num_filters"] = 100
    if m is not None: FLAGS["total_samples"] = ceil(m*(1.0+confusion))
    FLAGS["training"] = not FLAGS["no_training"]

    return FLAGS


def binary_crossentropy_from_logits(y_true,y_pred):
    return tf.keras.backend.binary_crossentropy(y_true, y_pred,from_logits=True)

from tensorflow import keras
Callback = keras.callbacks.Callback
import warnings
class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='val_acc', value=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

# FLAGS['m'] = 1000
# FLAGS['number_inits'] = 1
# FLAGS['label_corruption'] =  0.0
# FLAGS['confusion'] = 0.0
# FLAGS['dataset'] =  "cifar"
# FLAGS['binarized'] =  True
# FLAGS['number_layers'] =  4
# FLAGS['pooling'] =  "none"
# FLAGS['intermediate_pooling'] =  "0000"
# FLAGS['sigmaw'] =  1.0
# FLAGS['sigmab'] =  1.0
# FLAGS['network'] =  "cnn"
# FLAGS['prefix'] =  "test"
