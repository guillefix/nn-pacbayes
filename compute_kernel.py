import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.keras as keras
import pickle
import os
from math import ceil

from utils import preprocess_flags, save_kernel, save_kernel_partial
from utils import load_data,load_model,load_model_json,load_kernel
from utils import data_folder,kernel_folder,arch_folder

def main(_):

    FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    if init_dist != "gaussian":
        raise NotImplementedError("Initialization distributions other than Gaussian are not implemented for computing kernels!")

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)

    if n_gpus>0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str((rank)%n_gpus)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    set_session = keras.backend.set_session
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    config.allow_soft_placement = True  # so that it uses any other existing and supported devices, if the requested GPU:0 isn't found
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    train_images,flat_train_images,_,test_images,_ = load_data(FLAGS)
    image_size = train_images.shape[1]
    number_channels = train_images.shape[-1]
    #print("image_size", image_size)
    X = train_images
    flat_X = flat_train_images
    if compute_for_GP_train:
        test_images = test_images[:1000]
        data = test_images
        tp_order = np.concatenate([[0,len(data.shape)-1], np.arange(1, len(data.shape)-1)])
        print(data.shape,tp_order)
        flat_data = np.transpose(data, tp_order)  # NHWC -> NCHW # this is because the cnn GP kernels assume this
        flat_test_images = np.array([test_image.flatten() for test_image in flat_data])
        Xfull =  np.concatenate([flat_train_images,flat_test_images])
        flat_X = Xfull
        X = np.concatenate([train_images,test_images])

    print("compute kernel", network, dataset)

    # COMPUTE KERNEL
    if use_empirical_NTK:
        from nngp_kernel.empirical_ntk import empirical_NTK
        print(ceil(int(X.shape[0])*n_samples_repeats))
        from tensorflow.keras.models import model_from_json
        model = load_model(FLAGS)
        K = empirical_NTK(model,X)#,sess=sess)
    elif use_empirical_K:
        from nngp_kernel.empirical_kernel import empirical_K
        print("n_samples_repeats",n_samples_repeats)
        print(ceil(int(X.shape[0])*n_samples_repeats))
        arch_json_string = load_model_json(FLAGS)
        K = empirical_K(arch_json_string,X,ceil(int(X.shape[0])*n_samples_repeats),sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus,empirical_kernel_batch_size=empirical_kernel_batch_size, sess=sess, truncated_init_dist=truncated_init_dist,data_parallelism=False,store_partial_kernel=store_partial_kernel,partial_kernel_n_proc=partial_kernel_n_proc,partial_kernel_index=partial_kernel_index)
    if rank == 0:
        if not (use_empirical_K or use_empirical_NTK):
            if network=="cnn":
                from nngp_kernel.cnn_kernel import kernel_matrix
                K = kernel_matrix(flat_X,image_size=image_size,number_channels=number_channels,filter_sizes=filter_sizes,padding=padding,strides=strides,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)

            elif network=="resnet":
                from nngp_kernel.resnet_kernel import kernel_matrix
                K = kernel_matrix(flat_X,depth=number_layers,image_size=image_size,number_channels=number_channels,n_blocks=3,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)

            elif network == "fc":
                from nngp_kernel.fc_kernel import kernel_matrix
                K = kernel_matrix(flat_X,number_layers=number_layers,sigmaw=sigmaw,sigmab=sigmab,n_gpus=n_gpus)

        print(K)

        '''SAVE KERNEL'''
        if store_partial_kernel:
            save_kernel_partial(K,FLAGS,partial_kernel_index) 
        else:
            save_kernel(K,FLAGS)


if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    define_default_flags(f)
    f.DEFINE_boolean('compute_for_GP_train', False, "Whether to add a bit of test set to kernel, to be able to use it for GP training")
    f.DEFINE_boolean('store_partial_kernel', False, "Whether to store the kernels partially on a file to free the processes")
    f.DEFINE_integer('empirical_kernel_batch_size', 256, "batch size to use when computing the empirical kernel, larger models need smaller values, but smaller models can use larger values")
    f.DEFINE_integer('partial_kernel_n_proc', 175, "number of processes over which we are parallelizing the when computing partial kernels and saving")
    f.DEFINE_integer('partial_kernel_index', 0, "index of the process when using partial_kernels method")

    tf.compat.v1.app.run()
