import numpy as np
import tensorflow as tf
import keras
import pickle
import os

data_folder = "data/"
kernel_folder = "kernels/"
arch_folder = "archs/"


def main(_):

    FLAGS = tf.app.flags.FLAGS.flag_values_dict()
    from utils import preprocess_flags
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)

    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"]=str((rank)%n_gpus)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True

    #tf.enable_eager_execution(config=config)
    set_session = keras.backend.set_session
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    from utils import load_data,load_model,load_kernel
    train_images,flat_train_images,ys,_,_ = load_data(FLAGS)
    #print(train_images)
    #train_images = tf.constant(train_images)
    arch_json_string = load_model(FLAGS)

    image_size = train_images.shape[1]
    number_channels = train_images.shape[-1]
    input_dim = flat_train_images.shape[1]

    print(network, dataset)

    # COMPUTE KERNEL
    if use_empirical_K:
        from empirical_kernel import empirical_K
        from math import ceil
        print("n_samples_repeats",n_samples_repeats)
        print(ceil(int(train_images.shape[0])*n_samples_repeats))
        K = empirical_K(arch_json_string,train_images,ceil(int(train_images.shape[0])*n_samples_repeats),sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)
    if rank == 0:
        if not use_empirical_K:
            if network=="cnn":
                from cnn_kernel import kernel_matrix
                K = kernel_matrix(flat_train_images,image_size=image_size,number_channels=number_channels,filter_sizes=filter_sizes,padding=padding,strides=strides,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)

            elif network=="resnet":
                from resnet_kernel import kernel_matrix
                K = kernel_matrix(flat_train_images,depth=number_layers,image_size=image_size,number_channels=number_channels,n_blocks=3,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)

            elif network == "fc":
                from fc_kernel import kernel_matrix
                K = kernel_matrix(flat_train_images,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)

        print(K)

        '''SAVE KERNEL'''

        filename=kernel_folder
        for flag in ["network","dataset","m","confusion","label_corruption","binarized","whitening","random_labels","number_layers","sigmaw","sigmab"]:
            filename+=str(FLAGS[flag])+"_"
        filename += "kernel.npy"
        np.save(open(filename,"wb"),K)

        #compute PAC-Bayes bound

        if compute_bound:
            from GP_prob_gpy import GP_prob
            ys = [[y] for y in ys]
            logPU = GP_prob(K,flat_train_images,np.array(ys))

            delta = 2**-10
            bound = (-logPU+2*np.log(total_samples)+1-np.log(delta))/total_samples
            bound = 1-np.exp(-bound)
            print("pre-confusion-correction bound: ", bound)
            rho = confusion/(1.0+confusion)
            bound = (bound - 0.5*rho)/(1-rho) #to correct for the confusion changing the training data distribution (in training set, but not in test set)!
            print("Bound: ", bound)
            print("Accuracy bound: ", 1-bound)
            useful_flags = ["dataset", "network", "m","label_corruption","confusion", "number_layers", "sigmaw", "sigmab", "binarized", "pooling", "intermediate_pooling", "whitening", "training", "n_gpus"]
            with open(prefix+"bounds.txt","a") as file:
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

    tf.app.run()
