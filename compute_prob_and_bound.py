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
    train_images,flat_train_images,ys,_,_ = load_data(FLAGS)
    X = flat_train_images
    ys2 = [[y] for y in ys]
    Y = np.array(ys2)
    image_size = train_images.shape[1]
    number_channels = train_images.shape[-1]
    input_dim = flat_train_images.shape[1]

    print("compute probability and bound", network, dataset)

    if using_NTK:
        FLAGS["use_empirical_NTK"] = True
        K = load_kernel(FLAGS)
        print(K)
        #if using NTK, the above gets the NTK kernel, but we also need the non-NTK one to compute the bound!
        FLAGS["use_empirical_NTK"] = False
        theta_pre = load_kernel(FLAGS)
        print(theta_pre)
        if normalize_kernel:
            theta_pre = theta_pre/theta_pre.max()
        theta = kernel_mult*theta_pre
    else:
        K_pre = load_kernel(FLAGS)
        print(K_pre)
        if normalize_kernel:
            K_pre = K_pre/K_pre.max()
        K = kernel_mult*K_pre


    #finding log marginal likelihood of data
    if using_EP:
        from GP_prob.GP_prob_gpy2 import GP_prob
        logPU = GP_prob(K,X,Y, method="EP", using_exactPB=using_exactPB)
    elif using_Laplace:
        from GP_prob.GP_prob_gpy2 import GP_prob
        # from GP_prob.GP_prob_numpy import GP_prob
        logPU = GP_prob(K,X,Y,method="Laplace", using_exactPB=using_exactPB)
        # logPU = GP_prob(K,np.squeeze(Y))
    elif using_Laplace2:
        # from GP_prob.GP_prob_gpy import GP_prob
        from GP_prob.GP_prob_numpy import GP_prob #this gives different results because it uses a worse implementation of Laplace, by using a more Naive Newton method to find the maximum of the posterior
        # logPU = GP_prob(K,X,Y,method="Laplace")
        logPU = GP_prob(K,np.squeeze(Y))
    elif using_MC:
        from GP_prob.GP_prob_MC import GP_prob
        logPU = GP_prob(K,X,Y,FLAGS)
    elif using_regression:
        from GP_prob.GP_prob_regression import GP_prob
        # logPU = GP_prob(K,X,Y,sigma_noise=np.sqrt(total_samples/2))
        logPU = GP_prob(K,X,Y,sigma_noise=1.0)
    elif using_NTK:
        # from GP_prob.GP_prob_regression import GP_prob
        # logPU = GP_prob(K,X,Y,sigma_noise=np.sqrt(total_samples/2))
        # logPU = GP_prob(K,X,Y,sigma_noise=1.0, posterior="ntk")
        from GP_prob.GP_prob_ntk import GP_prob
        logPU = GP_prob(K,theta,X,Y,t=1e2)

    if rank == 0:
        print(logPU)
        #compute PAC-Bayes bound
        delta = 2**-10
        bound = (-logPU+2*np.log(total_samples)+1-np.log(delta))/total_samples
        bound = 1-np.exp(-bound)
        print("pre-confusion-correction bound: ", bound)
        rho = confusion/(1.0+confusion)
        bound = (bound - 0.5*rho)/(1-rho) #to correct for the confusion changing the training data distribution (in training set, but not in test set)!
        print("Bound: ", bound)
        print("Accuracy bound: ", 1-bound)
        useful_flags = ["dataset","boolfun_comp","boolfun", "network", "m","label_corruption","confusion", "number_layers", "sigmaw", "sigmab", "binarized", "pooling", "intermediate_pooling", "whitening", "training", "n_gpus", "kernel_mult", "normalize_kernel"]
        with open(results_folder+prefix+"bounds.txt","a") as file:
            file.write("#")
            for key in useful_flags:
                file.write("{}\t".format(key))
            file.write("bound")
            file.write("\t")
            file.write("logP")
            file.write("\n")
            for key in useful_flags:
                file.write("{}\t".format(FLAGS[key]))
            file.write("{}".format(bound))
            file.write("\t")
            file.write("{}".format(logPU))
            file.write("\n")

if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    define_default_flags(f)
    f.DEFINE_boolean('using_EP', False, "Whether to use Expectation Propagation method for computing probability")
    f.DEFINE_boolean('using_Laplace', False, "Whether to use Laplace method for computing probability")
    f.DEFINE_boolean('using_Laplace2', False, "Whether my numpy implementation of Laplace method for computing probability")
    f.DEFINE_boolean('using_regression', False, "Whether to use the exact relative entropy for MSE GP regression")
    f.DEFINE_boolean('using_NTK', False, "Whether  to use the exact relative entropy for MSE GP regression, with NTK posterior")
    f.DEFINE_boolean('using_exactPB', False, "Whether using exact PAC-Bayes on approximate posterior rather than approximate PAC-Bayes on exact postierior")
    f.DEFINE_boolean('using_MC', False, "Whether to use Monte Carlo method for computing probability")
    f.DEFINE_boolean('normalize_kernel', False, "Whether to normalize the kernel (by dividing by max value) or not")
    f.DEFINE_integer('num_post_samples', int(1e5), "Number of approximate EP posterior samples in importance-sampling-based Monte Carlo estimation of marginal likelihood")
    f.DEFINE_float('cov_mult', 1.0, "Factor by which to multiply the variance of the approximate posterior, to focus the importance sampling more in the non-zero likelihood region, at the risk of biasing away from true posterior.")
    f.DEFINE_float('kernel_mult', 1.0, "Factor by which to multiply the kernel before computing approximate marginal likelihood")
    f.DEFINE_float('mean_mult', 1.0, "Factor by which to multiply the mean of the approximate posterior, to focus the importance sampling more in the non-zero likelihood region, at the risk of biasing away from true posterior.")

    tf.compat.v1.app.run()
