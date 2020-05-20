import numpy as np
import tensorflow as tf
from math import *
import tensorflow_probability as tfp
import sys
import GPy
from GP_prob.custom_kernel_matrix.custom_kernel_matrix import CustomMatrix

# import load_dataset
#from gpflow import settings
# import tqdm
#import missinglink
#missinglink_callback = missinglink.KerasCallback()

from utils import binary_crossentropy_from_logits, EarlyStoppingByAccuracy, get_biases, get_weights, measure_sigmas, get_rescaled_weights, results_folder, EarlyStoppingByLoss

def cross_entropy_loss(x,p):
    return -x*np.log(p)-(1-x)*np.log(1-p)

def main(_):
    MAX_TRAIN_EPOCHS=5000

    FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()
    from utils import preprocess_flags
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)
    if doing_regression:
        assert loss == "mse"
    global threshold

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

    '''LOAD DATA & ARCHITECTURE'''

    from utils import load_data,load_model,load_kernel
    train_images,flat_train_images,ys,test_images,test_ys = load_data(FLAGS)
    print("max val", train_images.max())
    #print("ys", ys)
    #process data to be on the right format for GP
    #test on a smaller sample on test set because otherwise GP would run out of memory
    test_images = test_images[:1000]
    test_ys = test_ys[:1000]
    X = flat_train_images
    data = test_images
    tp_order = np.concatenate([[0,len(data.shape)-1], np.arange(1, len(data.shape)-1)])
    print(data.shape,tp_order)
    flat_data = np.transpose(data, tp_order)  # NHWC -> NCHW # this is because the cnn GP kernels assume this
    flat_test_images = np.array([test_image.flatten() for test_image in flat_data])
    Xfull =  np.concatenate([flat_train_images,flat_test_images])
    ys2 = [[y] for y in ys]
    ysfull = ys2 + [[y] for y in test_ys]
    Yfull = np.array(ysfull)
    Y = np.array(ys2)


    K_pre = load_kernel(FLAGS)
    print(K_pre)
    if normalize_kernel:
        K_pre = K_pre/K_pre.max()
    Kfull = kernel_mult*K_pre

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
    funs_filename = results_folder+prefix+"_"+str(rank)+"_nn_train_functions.txt"

    if loss=="mse":
        likelihood = "gaussian"
    elif loss=="ce":
        likelihood = "bernoulli"
    print("Training GP with "+likelihood+" likelihood")

    from initialization import get_all_layers, is_normalization_layer, reset_weights, simple_reset_weights
    if nn_random_labels or nn_random_regression_outputs:
        model = load_model(FLAGS)
        model.compile("sgd", loss="mse")
        layers = get_all_layers(model)
        are_norm = [is_normalization_layer(l) for l in layers for w in l.get_weights()]
        initial_weights = model.get_weights()

    if likelihood=="gaussian":
        inference_method = GPy.inference.latent_function_inference.exact_gaussian_inference.ExactGaussianInference()
        lik = GPy.likelihoods.gaussian.Gaussian(variance=0.002)
    if likelihood=="bernoulli":
        lik = GPy.likelihoods.Bernoulli()
        inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(parallel_updates=True)
    kernel = CustomMatrix(Xfull.shape[1],Xfull,Kfull)
    gp_model = GPy.core.GP(X=X,Y=Y,kernel=kernel,inference_method=inference_method, likelihood=lik)

    '''MAIN LOOP'''
    local_index = 0

    from math import ceil
    if len(tasks)>0:
        samples_per_chunk_base=min(len(tasks),10000)
        num_chunks = len(tasks)//samples_per_chunk_base
    else:
        samples_per_chunk_base=1
        num_chunks=0
    remainder = len(tasks)%samples_per_chunk_base
    if remainder > 0:
        num_chunks += 1
    for chunki in range(num_chunks):
        if chunki == num_chunks-1 and remainder>0:
            samples_per_chunk = remainder
        else:
            samples_per_chunk = samples_per_chunk_base
        funs_file = open(funs_filename,"a")
        print(chunki)
        #
        ##if the labels are to be generated by a neural network in parallel
        if nn_random_labels or nn_random_regression_outputs:
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

        local_index+=1

        #preds = model.predict(flat_test_images)[0]
        #dimensions of output of posterior_samples is (number of input points)x(dimension of output Y)x(number of samples)
        preds = gp_model.posterior_samples(flat_test_images,size=samples_per_chunk)[:,0,:].T
        print(preds.shape)
        #preds = np.array([pred[0] for pred in preds])
        if not doing_regression:
            th = 0.5
            train_loss, train_acc = 0, 1.0*samples_per_chunk
            test_loss, test_acc = np.sum(cross_entropy_loss(test_ys,preds))/len(test_ys), np.sum((preds>th)==test_ys)/len(test_ys)
        else:
            train_acc = train_loss = 0
            test_acc = test_loss = np.sum(cross_entropy_loss(test_ys,preds))/len(test_ys)

        #for th in np.linspace(0,1,1000):
        if loss=="mse":
            #NOTE: sensitivity and specificity are not implemented for MSE loss
            test_sensitivity = -1
            test_specificity = -1
        else:
            print("threshold", threshold)
            #TODO: this is ugly, I should just add a flag that allows to say whether we are doing threshold selection or not!!
            if threshold != -1:
                for th in np.linspace(0,1,1000):
                    test_specificity = np.sum(((sigmoid(preds)>th)==test_ys[:100])*(test_ys[:100]==0))/np.sum(test_ys[:100]==0)
                    if test_specificity>0.99:
                        num_0s = np.sum(test_ys==0)
                        if num_0s > 0:
                            test_specificity = np.sum(((sigmoid(preds)>th)==test_ys)*(test_ys==0))/(num_0s)
                        else:
                            test_specificity = -1
                        if test_specificity>0.99:
                            num_1s = np.sum(test_ys==1)
                            if num_1s > 0:
                                test_sensitivity = np.sum(((sigmoid(preds)>th)==test_ys)*(test_ys==1))/(num_1s)
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

        print("Training accuracy", train_acc/samples_per_chunk)
        print('Test accuracy:', test_acc/samples_per_chunk)
        if threshold != -1:
            print('Test sensitivity:', test_sensitivity/samples_per_chunk)
            print('Test specificity:', test_specificity/samples_per_chunk)
        if not ignore_non_fit or train_acc/samples_per_chunk == 1.0:
            print("printing function to file", funs_filename)
            functions = preds[:,:test_function_size]>0.5
            functions=functions.astype(int)
            print(functions.shape)
            functions = [''.join([str(int(x)) for x in function])+"\r\n" for function in functions]
            funs_file.writelines(functions)
            funs_file.close()
            #functions.append(function)
            test_accs += test_acc
            test_accs_squared += test_acc**2
            test_sensitivities += test_sensitivity
            test_specificities += test_specificity
            train_accs += train_acc
            train_accs_squared += train_acc**2

    test_accs_recv = comm.reduce(test_accs, root=0)
    test_accs_squared_recv = comm.reduce(test_accs_squared, root=0)
    test_sensitivities_recv = comm.reduce(test_sensitivities, root=0)
    test_specificities_recv = comm.reduce(test_specificities, root=0)
    train_accs_recv = comm.reduce(train_accs, root=0)
    train_accs_squared_recv = comm.reduce(train_accs_squared, root=0)

    '''PROCESS COLLECTIVE DATA'''
    if rank == 0:
        test_acc = test_accs_recv/number_inits
        test_sensitivity = test_sensitivities_recv/number_inits
        test_specificity = test_specificities_recv/number_inits
        train_acc = train_accs_recv/number_inits
        print('Mean train accuracy:', train_acc)
        print('Mean test accuracy:', test_acc)
        if threshold != -1:
            print('Mean test sensitivity:', test_sensitivity)
            print('Mean test specificity:', test_specificity)
        test_acc = test_accs_recv/number_inits
        train_acc = train_accs_recv/number_inits
        train_acc_std = train_accs_squared_recv/number_inits - train_acc**2
        test_acc_std = test_accs_squared_recv/number_inits - test_acc**2

        useful_train_flags = ["dataset", "m", "network", "pooling", "ignore_non_fit", "test_function_size", "number_layers", "sigmaw", "sigmab", "init_dist","use_shifted_init","shifted_init_shift","whitening", "centering", "oversampling", "oversampling2", "channel_normalization", "training", "binarized", "confusion","filter_sizes", "gamma", "intermediate_pooling", "label_corruption", "threshold", "n_gpus", "n_samples_repeats", "layer_widths", "number_inits", "padding"]
        with open(results_folder+prefix+"nn_training_results.txt","a") as file:
            file.write("#")
            for key in sorted(useful_train_flags):
                file.write("{}\t".format(key))
            file.write("\t".join(["train_acc", "test_error", "test_acc","test_sensitivity","test_specificity","train_acc_std","test_acc_std"]))
            file.write("\n")
            for key in sorted(useful_train_flags):
                file.write("{}\t".format(FLAGS[key]))
            file.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(train_acc, 1-test_acc,test_acc,\
                test_sensitivity,test_specificity,\
                train_acc_std,test_acc_std))


if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    define_default_flags(f)

    f.DEFINE_integer('number_inits',1,"Number of initializations")
    f.DEFINE_float('gamma',1.0,"weight for confusion samples (1.0 weigths them the same as normal samples)")
    f.DEFINE_boolean('ignore_non_fit', False, "Whether to ignore functions that don't fit data")
    f.DEFINE_integer('test_function_size',100,"Number of samples on the test set to use to evaluate the function the network has found")
    f.DEFINE_string('loss',"ce","Which loss to use (ce/mse/etc)")
    f.DEFINE_float('kernel_mult', 1.0, "Factor by which to multiply the kernel before computing approximate marginal likelihood")
    f.DEFINE_boolean('normalize_kernel', False, "Whether to normalize the kernel (by dividing by max value) or not")

    tf.compat.v1.app.run()
    import gc; gc.collect()
