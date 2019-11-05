import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import pickle
import torchvision
from torchvision import transforms, utils
from math import ceil
import keras_applications
import torch

from utils import preprocess_flags, save_data
from utils import data_folder,datasets_folder

def main(_):

    FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()
    FLAGS = preprocess_flags(FLAGS)
    print(FLAGS)
    globals().update(FLAGS)
    global m, total_samples

    print("Generating input samples", dataset, m)

    from keras.datasets import mnist
    (X_train_full, y_train_full), (X_test_full, y_test_full) = mnist.load_data()

    def data_binariser(i):
        if i%2 == 0:
            return 1
        return 0

    n = m
    X_train = X_train_full[:n].reshape(n,784)/255.0
    y_train = np.asarray([data_binariser(i) for i in y_train_full[:n]]).reshape(n,1)

    n = test_set_size
    X_test = X_test_full[:n].reshape(n,784)/255.0
    y_test = np.asarray([data_binariser(i) for i in y_test_full])[:n].reshape(n,1)

    if not zero_one:
        y_test=2*y_test-1
        y_train=2*y_train-1

    '''SAVING DATA SAMPLES'''
    save_data(tf.cast(X_train,tf.float32),tf.cast(y_train,tf.float32),tf.cast(X_test,tf.float32),tf.cast(y_test,tf.float32),FLAGS)


if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    define_default_flags(f)
    f.DEFINE_boolean('zero_one', True, "Whether to use 0,1 or -1,1, for binarized labels")
    f.DEFINE_integer('test_set_size',-1,"Number of test samples, -1 means all of it")

    tf.compat.v1.app.run()
    #tf.app.run()
    import gc; gc.collect()
