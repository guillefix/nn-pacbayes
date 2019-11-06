import numpy as np
import tensorflow as tf
from math import *
import tensorflow_probability as tfp

# import load_dataset
#from gpflow import settings
# import tqdm
#import missinglink
#missinglink_callback = missinglink.KerasCallback()

from utils import binary_crossentropy_from_logits,EarlyStoppingByAccuracy, get_biases, get_weights, measure_sigmas, get_rescaled_weights, results_folder

def main(_):
    MAX_TRAIN_EPOCHS=3000

    FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()
    from utils import preprocess_flags
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    num_tasks_per_job = number_inits//size
    tasks = list(range(int(rank*num_tasks_per_job),int((rank+1)*num_tasks_per_job)))

    if rank < number_inits%size:
        tasks.append(size*num_tasks_per_job+rank)

    import os
    if n_gpus>0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(rank%n_gpus)

    from tensorflow import keras

    def binary_accuracy_for_mse(y_true,y_pred):
        if zero_one:
            return keras.backend.mean(tf.cast(tf.equal(tf.cast(tf.math.greater(y_pred,0.5),tf.float32),y_true), tf.float32))
        else:
            return keras.backend.mean(tf.cast(tf.equal(tf.math.sign(y_pred),y_true), tf.float32))

    print(tf.__version__)
    if loss=="mse":
        callbacks = [EarlyStoppingByAccuracy(monitor='val_binary_accuracy_for_mse', value=1.0, verbose=0, wait_epochs=epochs_after_fit)]
    else:
        #if tf.__version__[:3] == "2.1":
        if tf.__version__[0] == "2":
            print("hi im tf 2")
            callbacks = [EarlyStoppingByAccuracy(monitor='val_accuracy', value=1.0, verbose=0, wait_epochs=epochs_after_fit)]
        else:
            callbacks = [EarlyStoppingByAccuracy(monitor='val_acc', value=1.0, verbose=0, wait_epochs=epochs_after_fit)]

    # callbacks += [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    #               ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
    #              ]

    '''LOAD DATA & ARCHITECTURE'''

    from utils import load_data,load_model,load_kernel, reset_weights
    train_images,_,ys,test_images,test_ys = load_data(FLAGS)
    input_dim = train_images.shape[1]
    num_channels = train_images.shape[-1]
    print(train_images.shape, ys.shape)


    sample_weights = None
    if gamma != 1.0:
        sample_weights = np.ones(len(ys))
        if not oversampling2:
            sample_weights[m:] = gamma
        else:
            raise NotImplementedError("Gamma not equal to 1.0 with oversampling2 not implemented")

    arch_json_string = load_model(FLAGS)
    from tensorflow.keras.models import model_from_json

    '''some custom initalizers and keras setup'''
    def cauchy_init(shape, dtype=None):
        # return keras.backend.variable((sigmaw/(np.sqrt(np.prod(shape[:-1]))))*np.random.standard_cauchy(shape), dtype=dtype)
        return (sigmaw/(np.sqrt(np.prod(shape[:-1]))))*np.random.standard_cauchy(shape)

    def shifted_init(shape, dtype=None):
        return sigmab*np.random.standard_normal(shape)-0.5

    class CauchyInit:
        def __call__(self, shape, dtype=None):
            return cauchy_init(shape, dtype=dtype)

    class ShiftedInit:
        def __call__(self, shape, dtype=None):
            return shifted_init(shape, dtype=dtype)

    custom_objects = {'cauchy_init': CauchyInit, 'shifted_init':ShiftedInit}
    model = model_from_json(arch_json_string,custom_objects=custom_objects)

    set_session = tf.compat.v1.keras.backend.set_session

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    '''TRAINING LOOP'''
    #things to keep track off
    functions = []
    test_accs = []
    train_accs = []
    weights_norms = []
    biases_norms = []
    iterss = []
    funs_filename = results_folder+prefix+"_"+str(rank)+"_nn_train_functions.txt"

    print("Training with",loss,"and optimizer",optimizer)
    if optimizer == "langevin":
        optim = tfp.optimizer.StochasticGradientLangevinDynamics(learning_rate=0.01)
    elif optimizer == "sgd":
        optim = keras.optimizers.SGD(lr=learning_rate)
        #keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=1e-6)
    else:
        optim = optimizer

    model = model_from_json(arch_json_string,custom_objects=custom_objects)
    model.compile(optim,
                  loss=binary_crossentropy_from_logits if loss=="ce" else loss,
                  metrics=([binary_accuracy_for_mse] if loss=="mse" else ['accuracy']))

    for init in tasks:
        funs_file = open(funs_filename,"a")
        print(init)

        #this reinitalizes the net
        reset_weights(model)

        print(train_images.shape,ys.shape)
        model.fit(train_images, ys, verbose=0,\
            sample_weight=sample_weights, validation_data=(train_images, ys), epochs=MAX_TRAIN_EPOCHS,callbacks=callbacks, batch_size=batch_size)

        '''GET DATA: weights, and errors'''
        #weights, biases = get_rescaled_weights(model)
        #weights_norm, biases_norm = measure_sigmas(model)
        weights_norm, biases_norm = -1, -1
        #print(weights_norm,biases_norm)

        train_loss, train_acc = model.evaluate(train_images, ys, verbose=0)
        print(train_acc)
        test_loss, test_acc = model.evaluate(test_images, test_ys, verbose=0)
        preds = model.predict(test_images)[:,0]
        # print(preds)
        # print(preds.shape)
        # test_false_positive_rate = test_fps/(len([x for x in test_ys if x==1]))
        def sigmoid(x):
            return np.exp(x)/(1+np.exp(x))

        #for th in np.linspace(0,1,1000):
        print('Test accuracy:', test_acc)
        if not ignore_non_fit or train_acc == 1.0:
            print("printing function to file", funs_filename)
            function = (model.predict(test_images[:test_function_size], verbose=0))[:,0]
            function = function>0
            function=function.astype(int)
            function = ''.join([str(int(i)) for i in function])
            funs_file.write(function+"\r\n")
            funs_file.close()
            #functions.append(function)
            test_accs.append(test_acc)
            train_accs.append(train_acc)
            weights_norms.append(weights_norm)
            biases_norms.append(biases_norm)
            iterss.append(model.history.epoch[-1])
        #keras.backend.clear_session()

    test_accs = comm.gather(test_accs, root=0)
    train_accs = comm.gather(train_accs, root=0)

    weights_norms = comm.gather(weights_norms,root=0)
    iterss = comm.gather(iterss, root=0)

    '''PROCESS COLLECTIVE DATA'''
    if rank == 0:
        weights_norms = sum(weights_norms,[])
        weights_norm_mean = np.mean(weights_norms)
        weights_norm_std = np.std(weights_norms)
        biases_norm_mean = np.mean(biases_norm)
        biases_norm_std = np.std(biases_norm)

        # functions = sum(functions,[])
        test_acc = np.mean(sum(test_accs,[]))
        train_acc = np.mean(sum(train_accs,[]))
        print('Mean test accuracy:', test_acc)
        print('Mean train accuracy:', train_acc)
        test_acc = np.mean(sum(test_accs,[]))
        train_acc = np.mean(sum(train_accs,[]))
        train_acc_std = np.std(sum(test_accs,[]))
        test_acc_std = np.std(sum(train_accs,[]))
        mean_iters = np.mean(sum(iterss,[]))

        useful_train_flags = ["dataset", "m", "network", "loss", "optimizer", "pooling", "epochs_after_fit", "ignore_non_fit", "test_function_size", "batch_size", "number_layers", "sigmaw", "sigmab", "init_dist","whitening", "centering", "oversampling", "oversampling2", "channel_normalization", "training", "binarized", "confusion","filter_sizes", "gamma", "intermediate_pooling", "label_corruption", "threshold", "n_gpus", "n_samples_repeats", "layer_width", "number_inits", "padding"]
        with open(results_folder+prefix+"nn_training_results.txt","a") as file:
            file.write("#")
            for key in sorted(useful_train_flags):
                file.write("{}\t".format(key))
            file.write("\t".join(["train_acc", "test_error", "test_acc","weights_norm_mean","weights_norm_std","biases_norm_mean","biases_norm_std","mean_iters","train_acc_std","test_acc_std"]))
            file.write("\n")
            for key in sorted(useful_train_flags):
                file.write("{}\t".format(FLAGS[key]))
            file.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:d}\t{:.4f}\t{:.4f}\n".format(train_acc, 1-test_acc,test_acc,\
                weights_norm_mean,weights_norm_std,biases_norm_mean,biases_norm_std,int(mean_iters),train_acc_std,test_acc_std)) #normalized to sqrt(input_dim)
    else:
        assert test_accs is None
        assert train_accs is None


if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    define_default_flags(f)

    f.DEFINE_integer('number_inits',1,"Number of initializations")
    f.DEFINE_integer('batch_size',32,"batch_size")
    f.DEFINE_integer('epochs_after_fit',1,"Number of epochs to wait after it first reacehs 100% accuracy")
    f.DEFINE_float('gamma',1.0,"weight for confusion samples (1.0 weigths them the same as normal samples)")
    f.DEFINE_float('learning_rate',0.01,"learning rate when using SGD")
    f.DEFINE_string('optimizer',"sgd","Which optimizer to use (keras optimizers available)")
    f.DEFINE_string('loss',"ce","Which loss to use (ce/mse/etc)")
    f.DEFINE_boolean('ignore_non_fit', False, "Whether to ignore functions that don't fit data")
    f.DEFINE_integer('test_function_size',100,"Number of samples on the test set to use to evaluate the function the network has found")
    f.DEFINE_boolean('zero_one', True, "Whether to use 0,1 or -1,1, for binarized labels")

    tf.compat.v1.app.run()
    import gc; gc.collect()

#
#
