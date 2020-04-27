import numpy as np
import pickle
import tensorflow as tf
from utils import preprocess_flags, save_data

imdb_data_folder="../for_guillermo"

def main(_):

    FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    X = np.load(imdb_data_folder+"X.npy")
    X_test = np.load(imdb_data_folder+"X_test_50.npy")
    y = np.load(imdb_data_folder+"y.npy")
    y_test = np.load(imdb_data_folder+"y_test_50.npy")

    K = pickle.load(open(imdb_data_folder+"K_LSTM_full.p","rb"))

    save_kernel(K,FLAGS)
    save_data(X,y,X_test,y_test,FLAGS)

if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    define_default_flags(f)
    f.DEFINE_boolean('compute_for_GP_train', False, "Whether to add a bit of test set to kernel, to be able to use it for GP training")
    f.DEFINE_integer('empirical_kernel_batch_size', 256, "batch size to use when computing the empirical kernel, larger models need smaller values, but smaller models can use larger values")
    f.DEFINE_boolean('out_of_sample_test_error', True, "Whether to test only on inputs outside of training data, or on whole dataset")
    f.DEFINE_boolean('unnormalized_images', False, "Whether to have the images in range [0,255.0], rather than the standard [0,1]")
    #f.DEFINE_boolean('extended_test_set', True, "Whether to extend the test set by the part of the training set not in the sample")
    f.DEFINE_boolean('random_training_set', True, "Whether to make the training set by sampling random instances from the full training set of the dataset, rather than just taking the m initial samples. Only implemented for images datasets ")
    f.DEFINE_string('booltrain_set', None, "when using the Boolean dataset option, you can provide the training set, encoded as a binary string (1 if input is to be included in training set, 0 otherwise), rather than randomly sampling one")
    f.DEFINE_string('binarization_method', "threshold", "the method to binarize the labels. At the moment we have implemented:  with a threshold, and by their parity (odd/even)")


    tf.compat.v1.app.run()
