import numpy as np
import tensorflow as tf
#import keras
from tensorflow import keras
import pickle
import os

from utils import preprocess_flags
from utils import data_folder,kernel_folder,arch_folder,results_folder

def main(_):

    FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    if init_dist != "gaussian":
        raise NotImplementedError("Initialization distributions other than Gaussian are not implemented for computing pac bayes bounds!")

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)

    if n_gpus>0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str((rank)%n_gpus)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    #tf.enable_eager_execution(config=config)
    set_session = tf.compat.v1.keras.backend.set_session
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    '''GET DATA'''
    from utils import load_data,load_model,load_kernel
    train_images,flat_train_images,ys,test_images,test_ys = load_data(FLAGS)
    print("max val", train_images.max())
    #print("ys", ys)
    #process data to be on the right format for GP
    #test on a smaller sample on test set because otherwise GP would run out of memory
    test_images = test_images[:test_function_size]
    test_ys = test_ys[:test_function_size]
    X = flat_train_images
    data = test_images
    tp_order = np.concatenate([[0,len(data.shape)-1], np.arange(1, len(data.shape)-1)])
    print(data.shape,tp_order)
    flat_data = np.transpose(data, tp_order)  # NHWC -> NCHW # this is because the cnn GP kernels assume this
    flat_test_images = np.array([test_image.flatten() for test_image in flat_data])
    Xtrain = flat_train_images
    Xtest = flat_test_images
    Xfull =  np.concatenate([flat_train_images,flat_test_images])
    ys2 = [[y] for y in ys]
    if test_fun_override is not None:
        ys2test = [[float(x)] for x in test_fun_override]
    else:
        ys2test = [[y] for y in test_ys]
    ysfull = ys2 + ys2test
    Yfull = np.array(ysfull)
    Ytrain = np.array(ys2)
    Ytest = np.array(ys2test)
    image_size = train_images.shape[1]
    number_channels = train_images.shape[-1]
    input_dim = flat_train_images.shape[1]

    print("compute probability and bound", network, dataset)

    # if loss is not "mse":
    #     raise NotImplementedError("Haven't implemented logQ estimate for CE loss yet")

    if using_NTK:
        raise NotImplementedError("Haven't implemented logQ estimate for NTK yet")
        # FLAGS["use_empirical_NTK"] = True
        # theta = load_kernel(FLAGS)
        # print(theta)
        # #if using NTK, the above gets the NTK kernel, but we also need the non-NTK one to compute the bound!
        # FLAGS["use_empirical_NTK"] = False
        # K_pre = load_kernel(FLAGS)
        # print(K_pre)
        # if normalize_kernel:
        #     K_pre = K_pre/K_pre.max()
        # K = kernel_mult*K_pre
        # if theta.shape[0] >= m: #must have compute kernel for GP_train
        #     theta = theta[:m,:m]
        # if K.shape[0] >= m: #must have compute kernel for GP_train
        #     K = K[:m,:m]
    else:
        K_pre = load_kernel(FLAGS)
        print(K_pre)
        if normalize_kernel:
            K_pre = K_pre/K_pre.max()
        Kfull = kernel_mult*K_pre


    #finding log marginal likelihood of data
    if using_EP:
        from GP_prob.nngp_mse_heaviside_posterior import nngp_mse_heaviside_posteror_logp
        logQ = nngp_mse_heaviside_posteror_logp(Xtrain,Ytrain,Xtest,Ytest,Kfull)
    else:
        raise NotImplementedError("Only EP estimation of logQ is implemented")

    if rank == 0:
        print(logQ)
        useful_flags = ["dataset","boolfun_comp","boolfun", "network", "m","label_corruption","confusion", "number_layers", "sigmaw", "sigmab", "binarized", "pooling", "intermediate_pooling", "whitening", "training", "n_gpus", "kernel_mult", "normalize_kernel"]
        with open(results_folder+prefix+"bounds.txt","a") as file:
            file.write("#")
            for key in useful_flags:
                file.write("{}\t".format(key))
            file.write("logQ")
            file.write("\n")
            for key in useful_flags:
                file.write("{}\t".format(FLAGS[key]))
            file.write("{}".format(logQ))
            file.write("\n")

if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    define_default_flags(f)
    f.DEFINE_boolean('using_EP', False, "Whether to use Expectation Propagation method for computing probability")
    f.DEFINE_boolean('using_NTK', False, "Whether  to use the exact relative entropy for MSE GP regression, with NTK posterior")
    f.DEFINE_boolean('normalize_kernel', False, "Whether to normalize the kernel (by dividing by max value) or not")
    f.DEFINE_float('kernel_mult', 1.0, "Factor by which to multiply the kernel before computing approximate marginal likelihood")
    f.DEFINE_string('test_fun_override', None, "If given, it substitutes the y-values of the test set with the labels given in the string")
    f.DEFINE_integer('test_function_size',100,"Number of samples on the test set to use to evaluate the function the network has found")

    tf.compat.v1.app.run()
