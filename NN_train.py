import numpy as np
import tensorflow as tf
from math import *
import tensorflow_probability as tfp
import sys
import gc

# import load_dataset
#from gpflow import settings
# import tqdm
#import missinglink
#missinglink_callback = missinglink.KerasCallback()

from utils import binary_crossentropy_from_logits, EarlyStoppingByAccuracy, get_biases, get_weights, measure_sigmas, get_rescaled_weights, results_folder, EarlyStoppingByLoss

def main(_):
    MAX_TRAIN_EPOCHS=5000

    FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()
    from utils import preprocess_flags
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)
    if doing_regression:
        assert loss == "mse"
    global threshold

    if using_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        rank=0
        size=1
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
            return keras.backend.mean(tf.cast(tf.equal(tf.cast(y_pred>0.5,tf.float32),y_true), tf.float32))
        else:
            return keras.backend.mean(tf.cast(tf.equal(tf.math.sign(y_pred),y_true), tf.float32))

    print(tf.__version__)
    if loss=="mse":
        callbacks = [EarlyStoppingByAccuracy(monitor='val_binary_accuracy_for_mse', value=acc_threshold, verbose=0, wait_epochs=epochs_after_fit)]
        if doing_regression:
            callbacks = [EarlyStoppingByLoss(monitor='val_loss', value=1e-2, verbose=0, wait_epochs=epochs_after_fit)]
    else:
        #if tf.__version__[:3] == "2.1":
        if tf.__version__[0] == "2":
            print("hi im tf 2")
            callbacks = [EarlyStoppingByAccuracy(monitor='val_accuracy', value=acc_threshold, verbose=0, wait_epochs=epochs_after_fit)]
        else:
            callbacks = [EarlyStoppingByAccuracy(monitor='val_acc', value=acc_threshold, verbose=0, wait_epochs=epochs_after_fit)]

    # callbacks += [EarlyStopping(monitor='val_loss', patience=2, verbose=0),
    #               ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
    #              ]

    '''LOAD DATA & ARCHITECTURE'''

    from utils import load_data,load_model,load_kernel
    train_images,_,ys,test_images,test_ys = load_data(FLAGS)
    print("max val", train_images.max())
    print("ys", ys)
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

    model = load_model(FLAGS)

    set_session = tf.compat.v1.keras.backend.set_session

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    '''TRAINING LOOP'''
    #things to keep track off
    #functions = []
    test_accs = 0
    test_accs_squared = 0
    test_sensitivities = 0
    test_specificities = 0
    train_accs = 0
    train_accs_squared = 0
    weightss = None
    biasess = None
    weightss_squared = None
    biasess_squared = None
    weights_norms = 0
    biases_norms = 0
    weights_norms_squared = 0
    biases_norms_squared = 0
    iterss = 0
    funs_filename = results_folder+prefix+"_"+str(rank)+"_nn_train_functions.txt"

    print("Training NN with",loss,"and optimizer",optimizer)
    if optimizer == "langevin":
        optim = tfp.optimizer.StochasticGradientLangevinDynamics(learning_rate=0.01)
    elif optimizer == "sgd":
        optim = keras.optimizers.SGD(lr=learning_rate)
        #keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=1e-6)
    else:
        optim = optimizer

    def get_metrics():
        if doing_regression:
            #return [keras.losses.mean_squared_error]
            return []
        elif loss=="mse":
            return [binary_accuracy_for_mse]
        else:
            return ['accuracy']

    print(loss)
    model.compile(optim,
                  loss=binary_crossentropy_from_logits if loss=="ce" else loss,
                  metrics=get_metrics())
                  #metrics=['accuracy',sensitivity])
                  #metrics=['accuracy',tf.keras.metrics.SensitivityAtSpecificity(0.99),\
                            #tf.keras.metrics.FalsePositives()])

    from initialization import get_all_layers, is_normalization_layer, reset_weights, simple_reset_weights
    if network not in ["cnn", "fc"]:
        layers = get_all_layers(model)
        are_norm = [is_normalization_layer(l) for l in layers for w in l.get_weights()]
        initial_weights = model.get_weights()

    local_index = 0
    for init in tasks:
        funs_file = open(funs_filename,"a")
        #print(init)
        #
        #TODO: move to a different file, as this is repeated in GP_train..
        ##if the labels are to be generated by a neural network in parallel
        if nn_random_labels or nn_random_regression_outputs:
            if local_index>0:
                if network in ["cnn", "fc"]:
                    simple_reset_weights(model, sigmaw, sigmab)
                else:
                    reset_weights(model, initial_weights, are_norm, sigmaw, sigmab, truncated_init_dist)
            if nn_random_labels:
                ys = model.predict(train_images)[:,0]>0
                if training:
                    test_ys = model.predict(test_images)[:,0]>0
            else:
                ys = model.predict(train_images)[:,0]
                if training:
                    test_ys = model.predict(test_images)[:,0]
        ##
        if local_index>0 or nn_random_labels or nn_random_regression_outputs:
            if network in ["cnn", "fc"]:
                simple_reset_weights(model, sigmaw, sigmab)
            else:
                reset_weights(model, initial_weights, are_norm, sigmaw, sigmab)

        local_index+=1

        ##this reinitalizes the net
        #model = load_model(FLAGS)
        #model.compile(optim,
        #              loss=binary_crossentropy_from_logits if loss=="ce" else loss,
        #              metrics=get_metrics())

        weights, biases = get_weights(model), get_biases(model)
        weights_norm, biases_norm = measure_sigmas(model)
        #print(weights_norm,biases_norm)

        #batch_size = min(batch_size, m)
        model.fit(train_images.astype(np.float32), ys.astype(np.float32), verbose=0,\
            sample_weight=sample_weights, validation_data=(train_images.astype(np.float32), ys.astype(np.float32)), epochs=MAX_TRAIN_EPOCHS,callbacks=callbacks, batch_size=min(m,batch_size))
        sys.stdout.flush()

        '''GET DATA: weights, and errors'''
        weights, biases = get_rescaled_weights(model)
        weights_norm, biases_norm = measure_sigmas(model) #TODO: make sure it works with archs with norm layers etc
        #print(weights_norm,biases_norm)

        if not doing_regression: # classification
            train_loss, train_acc = model.evaluate(train_images.astype(np.float32), ys.astype(np.float32), verbose=0)
            test_loss, test_acc = model.evaluate(test_images.astype(np.float32), test_ys.astype(np.float32), verbose=0)
        else:
            train_acc = train_loss = model.evaluate(train_images.astype(np.float32), ys, verbose=0)
            test_acc = test_loss = model.evaluate(test_images.astype(np.float32), test_ys, verbose=0)
        preds = model.predict(test_images)[:,0]
        # print(preds)
        # print(preds.shape)
        # test_false_positive_rate = test_fps/(len([x for x in test_ys if x==1]))
        def sigmoid(x):
            return np.exp(x)/(1+np.exp(x))

        #for th in np.linspace(0,1,1000):
        if loss=="mse":
            #NOTE: sensitivity and specificity are not implemented for MSE loss
            test_sensitivity = -1
            test_specificity = -1
        else:
            #print("threshold", threshold)
            #TODO: this is ugly, I should just add a flag that allows to say whether we are doing threshold selection or not!!
            if threshold != -1:
                for th in np.linspace(0,1,1000):
                    test_specificity = sum([(sigmoid(preds[i])>th)==x for i,x in enumerate(test_ys[:100]) if x==0])/(len([x for x in test_ys[:100] if x==0]))
                    if test_specificity>0.99:
                        num_0s = len([x for x in test_ys if x==0])
                        if num_0s > 0:
                            test_specificity = sum([(sigmoid(preds[i])>th)==x for i,x in enumerate(test_ys) if x==0])/(num_0s)
                        else:
                            test_specificity = -1
                        if test_specificity>0.99:
                            num_1s = len([x for x in test_ys if x==1])
                            if num_1s > 0:
                                test_sensitivity = sum([(sigmoid(preds[i])>th)==x for i,x in enumerate(test_ys) if x==1])/(num_1s)
                            else:
                                test_sensitivity = -1
                            break
            else:
                # for th in np.linspace(0,1,5): # low number of thresholds as I'm not exploring unbalanced datasets right now
                #     test_specificity = sum([(sigmoid(preds[i])>th)==x for i,x in enumerate(test_ys) if x==0])/(len([x for x in test_ys if x==0]))
                #     if test_specificity>0.99:
                #         test_sensitivity = sum([(sigmoid(preds[i])>th)==x for i,x in enumerate(test_ys) if x==1])/(len([x for x in test_ys if x==1]))
                #         break
                test_specificity = -1
                test_sensitivity = -1
        #print("Training accuracy", train_acc)
        #print('Test accuracy:', test_acc)
        #print('Test sensitivity:', test_sensitivity)
        #print('Test specificity:', test_specificity)

        if not ignore_non_fit or train_acc >= acc_threshold:
            #print("printing function to file", funs_filename)
            function = (model.predict(test_images[:test_function_size].astype(np.float32), verbose=0))[:,0]
            if loss=="mse" and zero_one:
                function = function>0.5
            else:
                function = function>0
            function=function.astype(int)
            function = ''.join([str(int(i)) for i in function])
            funs_file.write(function+"\r\n")
            funs_file.close()
            #functions.append(function)
            test_accs += test_acc
            test_accs_squared += test_acc**2
            test_sensitivities += test_sensitivity
            test_specificities += test_specificity
            train_accs += train_acc
            train_accs_squared += train_acc**2
            if weightss is None:
                weightss = weights
                biasess = biases
                weightss_squared = weights**2
                biasess_squared = biases**2
            else:
                weightss += weights
                biasess += biases
                weightss_squared += weights**2
                biasess_squared += biases**2
            weights_norms += weights_norm
            weights_norms_squared += weights_norm**2
            biases_norms += biases_norm
            biases_norms_squared += biases_norm**2
            iterss += model.history.epoch[-1]
        #keras.backend.clear_session()
        gc.collect()

    #print("Print functions to file")
    #with open(,"a") as file:
    #    file.write("\r\n".join(functions))
    #    file.write("\r\n")

    # functions = comm.gather(functions, root=0)
    if rank == 0:
        #test_accs_recv = np.empty([size,1],dtype=np.float32)
        #test_accs_squared_recv = np.empty([size,1],dtype=np.float32)
        #test_sensitivities_recv = np.empty([size,1],dtype=np.float32)
        #test_specificities_recv = np.empty([size,1],dtype=np.float32)
        #train_accs_recv = np.empty([size,1],dtype=np.float32)
        #train_accs_squared_recv = np.empty([size,1],dtype=np.float32)

        weights_shape = weightss.flatten().shape[0]
        biases_shape = biasess.flatten().shape[0]
        weightss_recv = np.zeros(weights_shape, dtype=np.float32)
        biasess_recv = np.zeros(biases_shape, dtype=np.float32)
        weightss_squared_recv = np.zeros(weights_shape, dtype=np.float32)
        biasess_squared_recv = np.zeros(biases_shape, dtype=np.float32)
        #weights_norms_recv = np.empty([size,1],dtype=np.float32)
        #weights_norms_squared_recv = np.empty([size,1],dtype=np.float32)
        #biases_norms_recv = np.empty([size,1],dtype=np.float32)
        #biases_norms_squared_recv = np.empty([size,1],dtype=np.float32)
        #iterss_recv = np.empty([size,1],dtype='i')

    else:
        #test_accs_recv = None
        #test_accs_squared_recv = None
        #test_sensitivities_recv = None
        #test_specificities_recv = None
        #train_accs_recv = None
        #train_accs_squared_recv = None

        weightss_recv = None
        weightss_squared_recv = None
        biasess_recv = None
        biasess_squared_recv = None
        #weights_norms_recv = None
        #weights_norms_squared_recv = None
        #biases_norms_recv = None
        #biases_norms_squared_recv = None
        #iterss_recv = None

    if using_mpi:
        test_accs_recv = comm.reduce(test_accs, root=0)
        test_accs_squared_recv = comm.reduce(test_accs_squared, root=0)
        test_sensitivities_recv = comm.reduce(test_sensitivities, root=0)
        test_specificities_recv = comm.reduce(test_specificities, root=0)
        train_accs_recv = comm.reduce(train_accs, root=0)
        train_accs_squared_recv = comm.reduce(train_accs_squared, root=0)

        comm.Reduce(weightss.flatten(), weightss_recv, root=0)
        comm.Reduce(biasess.flatten(), biasess_recv, root=0)
        comm.Reduce(weightss_squared.flatten(), weightss_squared_recv, root=0)
        comm.Reduce(biasess_squared.flatten(), biasess_squared_recv, root=0)
        weights_norms_recv = comm.reduce(weights_norms, root=0)
        weights_norms_squared_recv = comm.reduce(weights_norms_squared, root=0)
        biases_norms_recv = comm.reduce(biases_norms, root=0)
        biases_norms_squared_recv = comm.reduce(biases_norms_squared, root=0)
        iterss_recv = comm.reduce(iterss, root=0)
    else:
        test_accs_recv = test_accs
        test_accs_squared_recv = test_accs_squared
        test_sensitivities_recv = test_sensitivities
        test_specificities_recv = test_specificities
        train_accs_recv = train_accs
        train_accs_squared_recv = train_accs_squared

        weightss_recv=weightss.flatten()
        biasess_recv=biasess.flatten()
        weightss_squared_recv=weightss_squared.flatten()
        biasess_squared_recv=biasess_squared.flatten()
        weights_norms_recv = weights_norms
        weights_norms_squared_recv = weights_norms_squared
        biases_norms_recv = biases_norms
        biases_norms_squared_recv = biases_norms_squared
        iterss_recv = iterss

    '''PROCESS COLLECTIVE DATA'''
    if rank == 0:
        #weightss = np.stack(sum(weightss,[]))
        #weights_norms = sum(weights_norms,[])
        #biasess = np.stack(sum(biasess,[]))
        weights_mean = np.mean(weightss_recv)/number_inits #average over dimension indexing which weight it is (we've already reduced over the number_inits dimension)
        biases_mean = np.mean(biasess_recv)/number_inits
        weights_std = np.mean(weightss_squared_recv)/number_inits - weights_mean**2
        biases_std = np.mean(biasess_squared_recv)/number_inits - biases_mean**2
        weights_norm_mean = weights_norms_recv/number_inits
        weights_norm_std = weights_norms_squared_recv/number_inits - weights_norm_mean**2
        biases_norm_mean = biases_norms_recv/number_inits
        biases_norm_std = biases_norms_squared_recv/number_inits - biases_norm_mean**2

        # functions = sum(functions,[])
        test_acc = test_accs_recv/number_inits
        test_sensitivity = test_sensitivities_recv/number_inits
        test_specificity = test_specificities_recv/number_inits
        train_acc = train_accs_recv/number_inits
        print('Mean test accuracy:', test_acc)
        print('Mean test sensitivity:', test_sensitivity)
        print('Mean test specificity:', test_specificity)
        print('Mean train accuracy:', train_acc)
        test_acc = test_accs_recv/number_inits
        train_acc = train_accs_recv/number_inits
        train_acc_std = train_accs_squared_recv/number_inits - train_acc**2
        test_acc_std = test_accs_squared_recv/number_inits - test_acc**2
        mean_iters = 1.0*iterss_recv/number_inits

        useful_train_flags = ["dataset", "m", "network", "loss", "optimizer", "pooling", "epochs_after_fit", "ignore_non_fit", "test_function_size", "batch_size", "number_layers", "sigmaw", "sigmab", "init_dist","use_shifted_init","shifted_init_shift","whitening", "centering", "oversampling", "oversampling2", "channel_normalization", "training", "binarized", "confusion","filter_sizes", "gamma", "intermediate_pooling", "label_corruption", "threshold", "n_gpus", "n_samples_repeats", "layer_widths", "number_inits", "padding"]
        with open(results_folder+prefix+"nn_training_results.txt","a") as file:
            file.write("#")
            for key in sorted(useful_train_flags):
                file.write("{}\t".format(key))
            file.write("\t".join(["train_acc", "test_error", "test_acc","test_sensitivity","test_specificity","weights_std","biases_std","weights_mean", "biases_mean", "weights_norm_mean","weights_norm_std","biases_norm_mean","biases_norm_std","mean_iters","train_acc_std","test_acc_std"]))
            file.write("\n")
            for key in sorted(useful_train_flags):
                file.write("{}\t".format(FLAGS[key]))
            file.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:d}\t{:.4f}\t{:.4f}\n".format(train_acc, 1-test_acc,test_acc,\
                test_sensitivity,test_specificity,weights_std,biases_std,\
                weights_mean,biases_mean,weights_norm_mean,weights_norm_std,biases_norm_mean,biases_norm_std,int(mean_iters),train_acc_std,test_acc_std)) #normalized to sqrt(input_dim)


if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    define_default_flags(f)

    f.DEFINE_integer('number_inits',1,"Number of initializations")
    f.DEFINE_integer('batch_size',32,"batch_size")
    f.DEFINE_integer('epochs_after_fit',1,"Number of epochs to wait after it first reacehs 100% accuracy")
    f.DEFINE_float('gamma',1.0,"weight for confusion samples (1.0 weigths them the same as normal samples)")
    f.DEFINE_float('learning_rate',0.01,"learning rate when using SGD")
    f.DEFINE_float('acc_threshold',1.0,"the minimum training accuracy after which we early stop (unless combined with wait_for_epochs parameter)")
    f.DEFINE_string('optimizer',"sgd","Which optimizer to use (keras optimizers available)")
    f.DEFINE_string('loss',"ce","Which loss to use (ce/mse/etc)")
    f.DEFINE_boolean('ignore_non_fit', False, "Whether to ignore functions that don't fit data")
    f.DEFINE_integer('test_function_size',100,"Number of samples on the test set to use to evaluate the function the network has found")
    f.DEFINE_boolean('using_mpi', True, "Whether to use MPI or not (don't use if calling this script from another process using MPI, as it would throw error)")

    tf.compat.v1.app.run()
    import gc; gc.collect()
