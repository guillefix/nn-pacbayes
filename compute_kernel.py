import numpy as np
import tensorflow as tf
import keras
import pickle
import os
from math import ceil

from utils import preprocess_flags, save_kernel
from utils import load_data,load_model,load_kernel
from utils import data_folder,kernel_folder,arch_folder

def main(_):

    FLAGS = tf.app.flags.FLAGS.flag_values_dict()
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)

    os.environ["CUDA_VISIBLE_DEVICES"]=str((rank+1)%n_gpus)
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True

    #tf.enable_eager_execution(config=config)
    set_session = keras.backend.set_session
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    train_images,flat_train_images,ys,_,_ = load_data(FLAGS)
    #print(train_images)
    #train_images = tf.constant(train_images)
    arch_json_string = load_model(FLAGS)

    image_size = train_images.shape[1]
    number_channels = train_images.shape[-1]
    input_dim = flat_train_images.shape[1]

    print("compute kernel and bound", network, dataset)

    # COMPUTE KERNEL
    if use_empirical_K:
        from nngp_kernel.empirical_kernel import empirical_K
        print("n_samples_repeats",n_samples_repeats)
        print(ceil(int(train_images.shape[0])*n_samples_repeats))
        K = empirical_K(arch_json_string,train_images,ceil(int(train_images.shape[0])*n_samples_repeats),sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus,sess=sess)
    if rank == 0:
        if not use_empirical_K:
            if network=="cnn":
                from nngp_kernel.cnn_kernel import kernel_matrix
                K = kernel_matrix(flat_train_images,image_size=image_size,number_channels=number_channels,filter_sizes=filter_sizes,padding=padding,strides=strides,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)

            elif network=="resnet":
                from nngp_kernel.resnet_kernel import kernel_matrix
                K = kernel_matrix(flat_train_images,depth=number_layers,image_size=image_size,number_channels=number_channels,n_blocks=3,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)

            elif network == "fc":
                from nngp_kernel.fc_kernel import kernel_matrix
                K = kernel_matrix(flat_train_images,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)

        print(K)

        '''SAVE KERNEL'''
        save_kernel(K,FLAGS)


if __name__ == '__main__':

    f = tf.app.flags

    from utils import define_default_flags

    define_default_flags(f)

    tf.app.run()