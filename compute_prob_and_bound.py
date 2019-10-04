import numpy as np
import tensorflow as tf
import keras
import pickle
import os

from utils import preprocess_flags
from utils import data_folder,kernel_folder,arch_folder,results_folder

def main(_):

    FLAGS = tf.app.flags.FLAGS.flag_values_dict()
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)

    if n_gpus>0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str((rank+1)%n_gpus)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #tf.enable_eager_execution(config=config)
    set_session = keras.backend.set_session
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    from utils import load_data,load_model,load_kernel
    train_images,flat_train_images,ys,_,_ = load_data(FLAGS)
    X = flat_train_images
    ys2 = [[y] for y in ys]
    Y = np.array(ys2)

    arch_json_string = load_model(FLAGS)

    image_size = train_images.shape[1]
    number_channels = train_images.shape[-1]
    input_dim = flat_train_images.shape[1]

    print("compute probability and bound", network, dataset)

    K = load_kernel(FLAGS)

    #finding log marginal likelihood of data
    if using_EP:
        from GP_prob.GP_prob_gpy import GP_prob
        logPU = GP_prob(K,X,Y)
    elif using_MC:
        from GP_prob.GP_prob_MC import GP_prob
        logPU = GP_prob(K,X,Y,FLAGS)

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
        useful_flags = ["dataset", "network", "m","label_corruption","confusion", "number_layers", "sigmaw", "sigmab", "binarized", "pooling", "intermediate_pooling", "whitening", "training", "n_gpus"]
        with open(results_folder+prefix+"bounds.txt","a") as file:
            file.write("#")
            for key in useful_flags:
                file.write("{}\t".format(key))
            file.write("bound")
            file.write("\n")
            for key in useful_flags:
                file.write("{}\t".format(FLAGS[key]))
            file.write("{}".format(bound))
            file.write("\n")

if __name__ == '__main__':

    f = tf.app.flags

    from utils import define_default_flags

    define_default_flags(f)
    f.DEFINE_boolean('using_EP', False, "Whether to use Expectation Propagation method for computing probability")
    f.DEFINE_boolean('using_MC', False, "Whether to use Monte Carlo method for computing probability")
    f.DEFINE_integer('num_post_samples', int(1e5), "Number of approximate EP posterior samples in importance-sampling-based Monte Carlo estimation of marginal likelihood")
    f.DEFINE_float('cov_mult', 1.0, "Factor by which to multiply the variance of the approximate posterior, to focus the importance sampling more in the non-zero likelihood region, at the risk of biasing away from true posterior.")

    tf.compat.v1.app.run()
