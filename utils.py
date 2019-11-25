import numpy as np
import tensorflow as tf
from math import ceil
import h5py
import os
from tensorflow import keras

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))+"/"
data_folder = ROOT_FOLDER+"data/"
datasets_folder = ROOT_FOLDER+"datasets/"
arch_folder = ROOT_FOLDER+"archs/"
kernel_folder = ROOT_FOLDER+"kernels/"
results_folder = ROOT_FOLDER+"results/"
if not os.path.isdir(data_folder):
    os.mkdir(data_folder)
if not os.path.isdir(datasets_folder):
    os.mkdir(datasets_folder)
if not os.path.isdir(arch_folder):
    os.mkdir(arch_folder)
if not os.path.isdir(kernel_folder):
    os.mkdir(kernel_folder)
if not os.path.isdir(results_folder):
    os.mkdir(results_folder)

'''DATA FUNCTIONS'''
def data_filename(FLAGS):
    filename=data_folder
    for flag in ["prefix", "network","dataset","m","confusion","label_corruption","binarized","whitening","centering","channel_normalization","threshold","random_labels", "oversampling", "oversampling2"]:
        filename+=str(FLAGS[flag])+"_"
    if FLAGS["dataset"] == "boolean" and FLAGS["boolfun_comp"] is not None:
        filename+=str(FLAGS["boolfun_comp"])+"_"
    filename += "data.h5"
    return filename

def save_data(train_images,ys,test_images,test_ys,FLAGS):
    filename = data_filename(FLAGS)
    h5f = h5py.File(filename,"w")
    h5f.create_dataset('train_images', data=train_images)

    # ys = [y[0] for y in ys]
    h5f.create_dataset('ys', data=ys)
    if FLAGS["training"]:
        h5f.create_dataset('test_images', data=test_images)
        h5f.create_dataset('test_ys', data=test_ys)

    h5f.close()

def load_data(FLAGS):
    filename = data_filename(FLAGS)
    return load_data_by_filename(filename, training=FLAGS["training"])

def load_data_by_filename(filename, training=True):
    h5f = h5py.File(filename,'r')
    train_images = h5f['train_images'][:]
    ys = h5f['ys'][:]
    if training:
        test_images = h5f['test_images'][:]
        test_ys = h5f['test_ys'][:]
    else:
        test_images = test_ys = []
    h5f.close()
    data = train_images
    tp_order = np.concatenate([[0,len(data.shape)-1], np.arange(1, len(data.shape)-1)])
    flat_data = np.transpose(data, tp_order)  # NHWC -> NCHW # this is because the cnn GP kernels assume this
    flat_data = np.array([train_image.flatten() for train_image in flat_data])
    return train_images,flat_data,ys,test_images,test_ys

'''ARCHITECTURE FUNCTIONS'''
def arch_filename(FLAGS):
    filename=arch_folder
    for flag in ["network","binarized","number_layers","pooling","intermediate_pooling","intermediate_pooling_type","init_dist"]:
        filename+=str(FLAGS[flag])+"_"
    filename += "model"
    return filename


def save_arch(json_string,FLAGS):
    filename = arch_filename(FLAGS)
    with open(filename, "w") as f:
        f.write(json_string)

def load_model_json(FLAGS):
    filename = arch_filename(FLAGS)
    json_string_filename = filename
    arch_json_string = open(filename, "r") .read()
    return arch_json_string

def load_model(FLAGS):
    CauchyInit = cauchy_init_class_wrapper(FLAGS["sigmaw"])
    ShiftedInit = shifted_init_class_wrapper(FLAGS["sigmab"],FLAGS["shifted_init_shift"])
    custom_objects = {'cauchy_init': CauchyInit, 'shifted_init':ShiftedInit}
    arch_json_string = load_model_json(FLAGS)
    from tensorflow.keras.models import model_from_json
    model = model_from_json(arch_json_string, custom_objects=custom_objects)
    return model


def kernel_filename(FLAGS):
    filename=kernel_folder
    for flag in ["prefix","network","dataset","m","confusion","label_corruption","binarized","whitening","random_labels","number_layers","sigmaw","sigmab","pooling","intermediate_pooling","intermediate_pooling_type"]:
        filename+=str(FLAGS[flag])+"_"
    if FLAGS["use_empirical_NTK"]: filename += "NTK_"
    filename += "kernel.npy"
    return filename

def save_kernel(K,FLAGS):
    filename = kernel_filename(FLAGS)
    np.save(open(filename,"wb"),K)

def load_kernel(FLAGS):
    filename = kernel_filename(FLAGS)
    return load_kernel_by_filename(filename)

def load_kernel_by_filename(filename):
    K = np.load(filename,"r")
    return K

def define_default_flags(f):
    f.DEFINE_integer('m',None,"Number of training examples")
    f.DEFINE_float('label_corruption', 0.0, "Fraction of corrupted labels")
    f.DEFINE_float('confusion',0.0,"Number of confusion samples to add to training data, as a fraction of m")
    f.DEFINE_string('boolfun_comp', "none", "The LZ complexity of the Boolean function to use is using boolean dataset")
    f.DEFINE_string('boolfun', "none", "The Boolean function to use if using boolean dataset")
    f.DEFINE_string('dataset', None, "The dataset to use")
    f.DEFINE_string('network', None, "The type of network to use")
    f.DEFINE_integer('number_layers', None, "The number of layers in the network")
    f.DEFINE_string('layer_widths', "512", "Number of hidden neurons (fc) or filters (CNN)")
    f.DEFINE_string('activations', "relu", "Number of hidden neurons (fc) or filters (CNN)")
    f.DEFINE_boolean('binarized', True, "Whether to convert classification labels to binary")
    f.DEFINE_boolean('oversampling', False, "Whether to oversample the minority class from the data distribution (a standard technique when dealing with class imbalance)")
    f.DEFINE_boolean('oversampling2', False, "Whether to oversample the minority class from the empirical distribution out of an original samplo of size m (a standard technique when dealing with class imbalance)")
    f.DEFINE_string('pooling', None, "The pooling type to use (none/avg/max)")
    f.DEFINE_string('intermediate_pooling', "0000", "Whether individual layers have a local maxpooling after them; 1 is maxpool; 0 no maxpool")
    f.DEFINE_string('intermediate_pooling_type', "max", "The type of pooling at intermediate layers (avg/max)")
    # f.DEFINE_integer('number_inits',1,"Number of initializations")
    f.DEFINE_float('sigmaw', 1.0, "The variance parameter of the weights; their variance will be sigmaw/sqrt(number of inputs to neuron")
    f.DEFINE_float('sigmab', 1.0, "The variance of the biases")
    f.DEFINE_string('init_dist', "gaussian", "The distribution to use to initialize parameters")
    f.DEFINE_float('shifted_init_shift',0.0,"Mean of the bias term distribution of the last layer (for some class imbalance experiments)")
    # f.DEFINE_boolean('compute_bound', False, "Whether to compute the PAC-Bayes bound or just generate the training data")
    #f.DEFINE_boolean('compute_kernel', False, "Whether to compute the kernel or just generate the training data")
    f.DEFINE_boolean('use_shifted_init', False, "Whether to use a distribution of the last layer bias term which has non-zero mean; may be useful for learning class-imbalanced data")
    f.DEFINE_boolean('whitening', False, "Whether to perform ZCA whitening and normalization on the training data")
    f.DEFINE_boolean('centering', False, "Whether to substract the mean of the data")
    f.DEFINE_boolean('channel_normalization', False, "Whether to normalize the channel of the images")
    f.DEFINE_boolean('no_training', False, "Whether to also generate labels and test data")
    f.DEFINE_boolean('use_empirical_K', False, "Whether to use the empirical kernel matrix (from sampling) or the analytical one")
    f.DEFINE_boolean('use_empirical_NTK', False, "Whether to calculate the empirical NTK rather than NNGP")
    f.DEFINE_integer('n_gpus', 1, "Number of GPUs to use")
    f.DEFINE_integer('threshold', -1, "Label above or on which to binarze as 1, and below which to binarize as 0")
    f.DEFINE_float('n_samples_repeats', 1.0, "Number of samples to compute empirical kernel, as a multiple of training set size, m")
    f.DEFINE_boolean('random_labels', True, "Whether the confusion data is constructed by randomizing the labels, or by taking a wrong label")
    f.DEFINE_boolean('zero_one', True, "Whether to use 0,1 or -1,1, for binarized labels")
    f.DEFINE_boolean('nn_random_labels', False, "Whether to set the labels by a random sample from the neural network which we are going to us")
    f.DEFINE_string('prefix', "", "A prefix to use for the result files")

def preprocess_flags(FLAGS):

    if FLAGS["intermediate_pooling"] == "0000":
        FLAGS["intermediate_pooling"] = "0"*FLAGS["number_layers"]

    if FLAGS["number_layers"] != len(FLAGS["intermediate_pooling"]):
        raise ValueError("length of intermediate_pooling binary string should be the same as the number of layers; you are providing whether invidiaual layers have a local maxpooling after them; 1 is maxpool; 0 no maxpool")

    number_layers = FLAGS["number_layers"]
    intermediate_pooling = FLAGS["intermediate_pooling"]
    confusion = FLAGS["confusion"]
    m = FLAGS["m"]
    FLAGS["filter_sizes"] = [[5,5],[2,2]]*(number_layers//2) + [[5,5]]*(number_layers%2)
    FLAGS["padding"] = ["VALID", "SAME"]*(number_layers//2) + ["VALID"]*(number_layers%2)
    FLAGS["pooling_in_layer"] = [x=="1" for x in intermediate_pooling]
    FLAGS["strides"]=[[1, 1]] * number_layers
    if len(FLAGS["layer_widths"].split(",")) == 1:
        FLAGS["num_filters"] = int(FLAGS["layer_widths"])
    else:
        raise NotImplementedError("haven't implemented varying number of filternumbers for CNNs")
    if m is not None: FLAGS["total_samples"] = ceil(m*(1.0+confusion))
    FLAGS["training"] = not FLAGS["no_training"]
    if FLAGS["pooling"] == "none": FLAGS["pooling"] = None

    for layer_wise_thing in ["layer_widths","activations"]:
        if layer_wise_thing == "layer_widths":
            FLAGS[layer_wise_thing] = [int(w) for w in FLAGS[layer_wise_thing].split(",")]
        elif layer_wise_thing == "activations":
            FLAGS[layer_wise_thing] = FLAGS[layer_wise_thing].split(",")

        if len(FLAGS[layer_wise_thing]) != number_layers:
            if len(FLAGS[layer_wise_thing]) == 1:
                FLAGS[layer_wise_thing] = FLAGS[layer_wise_thing]*number_layers
            else:
                raise AssertionError("Number of "+layer_wise_thing+" should be the same as number of layers")

    return FLAGS


def binary_crossentropy_from_logits(y_true,y_pred):
    return tf.keras.backend.binary_crossentropy(y_true, y_pred,from_logits=True)

Callback = keras.callbacks.Callback
import warnings
class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='val_accuracy', value=0.98, wait_epochs=0, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.wait_epochs = wait_epochs
        self.first_time = True
        self.epoch_after_first_time = 0
    def on_epoch_end(self, epoch, logs=None):
        #print(logs)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        if current >= self.value:
            if self.first_time:
                self.first_time = False
                self.first_time_epoch = epoch
                if self.wait_epochs == 0:
                    if self.verbose > 0:
                        print("Epoch %05d: early stopping THR" % epoch)
                    self.model.stop_training = True
            else:
                if (epoch - self.first_time_epoch) >= self.wait_epochs:
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

def get_weights(model):
    all_parameters = model.get_weights()
    weights = filter(lambda w: len(w.shape)>1, all_parameters)
    return weights

def get_biases(model):
    all_parameters = model.get_weights()
    biases = filter(lambda w: len(w.shape)==1, all_parameters)
    return biases

def measure_sigmas(model):
    def getsigma(w):
        shape = w.shape
        w_flat = w.flatten()
        return np.dot(w_flat,w_flat)*np.prod(shape[:-1]) #np.prod([])=1.0
    def get_count(w):
        return np.prod(w.shape)
    weights, biases = list(get_weights(model)), list(get_biases(model))
    w_count = np.sum([get_count(w) for w in weights])
    b_count = np.sum([get_count(b) for b in biases])
    varws = np.sum([getsigma(w) for w in weights])/w_count
    varbs = np.sum([getsigma(b) for b in biases])/b_count
    return np.sqrt(varws), np.sqrt(varbs)

def get_rescaled_weights(model):
    def get_rescaled_weight(w):
        shape = w.shape
        if len(shape) == 1:
           #return tf.random.normal(shape).eval(session=sess)
           return w
        else:
            return w*np.sqrt(np.prod(shape[:-1]))
    weights, biases = get_weights(model), get_biases(model)
    ws = np.concatenate([get_rescaled_weight(w).flatten() for w in weights])
    bs = np.concatenate([get_rescaled_weight(w).flatten() for w in biases])
    return ws, bs

from math import log

def log2(x):
    return log(x)/log(2.0)

def entropy(f):
    n0=0
    n=len(f)
    for char in f:
        if char=='0':
            n0+=1
    n1=n-n0
    if n1 > 0 and n0 > 0:
        return log2(n) - (1.0/n)*(n0*log2(n0)+n1*log2(n1))
    else:
        return 0

def cauchy_init_wrapper(sigmaw):
    def cauchy_init(shape, dtype=None):
        return (sigmaw/(np.sqrt(np.prod(shape[:-1]))))*np.random.standard_cauchy(shape)
    return cauchy_init


def shifted_init_wrapper(sigmab, shifted_init_shift):
    def shifted_init(shape, dtype=None):
        return 1.0*np.random.standard_normal(shape)-shifted_init_shift
        #return sigmab*np.random.standard_normal(shape)-shifted_init_shift
    return shifted_init

def cauchy_init_class_wrapper(sigmaw):
    class CauchyInit:
        def __call__(self, shape, dtype=None):
            return cauchy_init_wrapper(sigmaw)(shape, dtype=dtype)

def shifted_init_class_wrapper(sigmab, shifted_init_shift):
    class ShiftedInit:
        def __call__(self, shape, dtype=None):
            return shifted_init_wrapper(sigmab,shifted_init_shift)(shape, dtype=dtype)
#
#

    #from tensorflow.python.client import device_lib
    #local_device_protos = device_lib.list_local_devices()
    #print(local_device_protos)
    #def get_available_gpus():
    #    local_device_protos = device_lib.list_local_devices()
    #    return [x.name for x in local_device_protos if x.device_type == 'GPU']

    #num_gpus = len(get_available_gpus())
    #num_gpus = n_gpus
    #print("num_gpus",num_gpus)
