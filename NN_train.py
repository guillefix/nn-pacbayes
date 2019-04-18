import numpy as np
import tensorflow as tf
from math import *

import load_dataset
#from gpflow import settings
# import tqdm
#import missinglink
#missinglink_callback = missinglink.KerasCallback()

from utils import binary_crossentropy_from_logits,EarlyStoppingByAccuracy

def main(_):

    FLAGS = tf.app.flags.FLAGS.flag_values_dict()
    from utils import preprocess_flags
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=str(rank)

    from tensorflow import keras
    # weight_initializer = keras.initializers.RandomNormal(stddev=1/np.sqrt(input_dim))
    # bias_initializer = keras.initializers.RandomNormal(stddev=0.5)
    # keras.layers.Flatten(input_shape=(28, 28)),

    callbacks = [
            EarlyStoppingByAccuracy(monitor='val_acc', value=1.0, verbose=1),
            #missinglink_callback,
            # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
            # ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
        ]

    # %%

    '''LOAD DATA & ARCHITECTURE'''

    from utils import load_data,load_model,load_kernel
    train_images,_,ys,test_images,test_ys = load_data(FLAGS)
    input_dim = train_images.shape[1]
    num_channels = train_images.shape[-1]

    sample_weights = np.ones(total_samples)
    sample_weights[m:] = gamma

    arch_json_string = load_model(FLAGS)
    from tensorflow.keras.models import model_from_json
    model = model_from_json(arch_json_string)

    set_session = keras.backend.set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    # config.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    test_accs = []
    train_accs = []
    weightss = []
    if network == "cnn":
        weightss_first_layer = []
    biasess = []
    weights_norms = []
    biases_norms = []
    iterss = []


    for init in range(number_inits//size):
        print(init)
        model = model_from_json(arch_json_string)

        model.compile(optimizer='sgd',#keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=1e-6),#'sgd',#tf.keras.optimizers.SGD(lr=0.01),
                      #loss='binary_crossentropy',
                      loss=binary_crossentropy_from_logits,
                      # loss_weights=[50000],
                      metrics=['accuracy'])

        # model.fit(train_images, ys, verbose=2, epochs=500)
        # print(ys)
        model.fit(train_images, ys, verbose=1, sample_weight=sample_weights, validation_data=(train_images, ys), epochs=3000,callbacks=callbacks)
        #print([w.shape for w in model.get_weights()])
        #print(np.concatenate([w.flatten() for w in model.get_weights()]).shape)

        '''GET DATA: weights, and errors'''
        if network != "resnet": #not tested this part for resnet yet..
            p=model.get_weights()
            if network == "cnn":
                weigths=np.concatenate([w.flatten() for w in p[2::2]])
                weigths_first_layer=np.concatenate([w.flatten() for w in p[0:2:2]])
            if network == "fc":
                weigths=np.concatenate([w.flatten() for w in p[::2]])
            biases=np.concatenate([b.flatten() for b in p[1::2]])
            weights_norm = np.linalg.norm(weigths)*np.sqrt(num_filters)/(np.sqrt(len(weigths)))
            biases_norm = np.linalg.norm(biases)/np.sqrt(len(biases))
            print(weights_norm,biases_norm)

        train_loss, train_acc = model.evaluate(train_images, ys)
        test_loss, test_acc = model.evaluate(test_images, test_ys)

        print('Test accuracy:', test_acc)
        test_accs.append(test_acc)
        train_accs.append(train_acc)
        if network != "resnet":
            weightss.append(weigths)
            if network == "cnn":
                weightss_first_layer.append(weigths_first_layer)
            biasess.append(biases)
            weights_norms.append(weights_norm)
            biases_norms.append(biases_norm)
        iterss.append(model.history.epoch[-1])
        keras.backend.clear_session()

    print("HI")
    test_accs = comm.gather(test_accs, root=0)
    train_accs = comm.gather(train_accs, root=0)

    if network != "resnet":
        weightss = comm.gather(weightss, root=0)
        weights_norms = comm.gather(weights_norms,root=0)
        if network == "cnn":
            weightss_first_layer = comm.gather(weightss_first_layer, root=0)
        biasess = comm.gather(biasess, root=0)
    iterss = comm.gather(iterss, root=0)

    '''PROCESS COLLECTIVE DATA'''
    if rank == 0:
        if network == "cnn" or network == "fc":
            weightss = np.stack(weightss,axis=-1)
            if network == "cnn":
                weightss_first_layer = np.stack(weightss_first_layer,axis=-1)
            biasess = np.stack(biasess,axis=-1)
            weights_std = np.mean(np.std(weightss,axis=-1))*np.sqrt(num_filters)
            if network == "cnn":
                weights_first_layer_std = np.mean(np.std(weightss_first_layer,axis=-1))*np.sqrt(num_channels)
                weights_std = np.mean([weights_first_layer_std,weights_std])
            biases_std = np.mean(np.std(biasess,axis=-1))
            weights_norm_mean = np.mean(weights_norms)
            weights_norm_std = np.std(weights_norms)
            biases_norm_mean = np.mean(biases_norm)
            biases_norm_std = np.std(biases_norm)
        else:
            weights_std = -1
            biases_std = -1
            weights_norm_mean = -1
            weights_norm_std = -1
            biases_norm_mean = -1
            biases_norm_std = -1
        test_acc = np.mean(np.array(test_accs))
        train_acc = np.mean(np.array(train_accs))
        test_acc = np.mean(np.array(test_accs))
        train_acc = np.mean(np.array(train_accs))
        train_acc_std = np.std(np.array(test_accs))
        test_acc_std = np.std(np.array(train_accs))
        mean_iters = np.mean(iterss)

        useful_flags = FLAGS.copy()
        if "data_dir" in useful_flags: del useful_flags["data_dir"]
        if "helpfull" in useful_flags: del useful_flags["helpfull"]
        if "help" in useful_flags: del useful_flags["help"]
        if "helpshort" in useful_flags: del useful_flags["helpshort"]
        if "h" in useful_flags: del useful_flags["h"]
        if "f" in useful_flags: del useful_flags["f"]
        if "prefix" in useful_flags: del useful_flags["prefix"]
        with open(prefix+"nn_training_results.txt","a") as file:
            file.write("#")
            for key, value in sorted(useful_flags.items()):
                file.write("{}\t".format(key))
            file.write("\t".join(["train_acc", "test_error","weights_std","biases_std","weights_norm_mean","weights_norm_std","biases_norm_mean","biases_norm_std","mean_iters","train_acc_std","test_acc_std"]))
            file.write("\n")
            for key, value in sorted(useful_flags.items()):
                file.write("{}\t".format(value))
            file.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:d}\t{:.4f}\t{:.4f}\n".format(train_acc, 1-test_acc,weights_std,biases_std,\
                weights_norm_mean,weights_norm_std,biases_norm_mean,biases_norm_std,int(mean_iters),train_acc_std,test_acc_std)) #normalized to sqrt(input_dim)
    else:
        assert test_accs is None
        assert train_accs is None
    # for i in range(number_inits//20):
        # Parallel(n_jobs=20)(train_net(init,ys,test_ys,train_images,test_images) for init in range(number_inits))


if __name__ == '__main__':

    f = tf.app.flags

    from utils import define_default_flags

    define_default_flags(f)

    f.DEFINE_integer('number_inits',1,"Number of initializations")
    f.DEFINE_float('gamma',1.0,"weight for confusion samples (1.0 weigths them the same as normal samples)")

    tf.app.run()
    import gc; gc.collect()
