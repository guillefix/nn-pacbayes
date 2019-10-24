import numpy as np
import tensorflow as tf
from math import *
import tensorflow_probability as tfp

# import load_dataset
#from gpflow import settings
# import tqdm
#import missinglink
#missinglink_callback = missinglink.KerasCallback()

from utils import binary_crossentropy_from_logits,EarlyStoppingByAccuracy, get_biases, get_weights, measure_sigmas, get_rescaled_weights

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

    import os
    if n_gpus>0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(rank%n_gpus)

    from tensorflow import keras

    callbacks = [
            #EarlyStoppingByAccuracy(monitor='val_accuracy', value=1.0, verbose=1, wait_epochs=epochs_after_fit),
            EarlyStoppingByAccuracy(monitor='val_acc', value=1.0, verbose=1, wait_epochs=epochs_after_fit),
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

    # from keras.utils.generic_utils import get_custom_objects
    #
    # get_custom_objects().update({'cauchy_init': CauchyInit})

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
    # config.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    test_accs = []
    #test_sensitivities = []
    test_sensitivities_base = []
    test_specificities_base = []
    train_accs = []
    weightss = []
    biasess = []
    weights_norms = []
    biases_norms = []
    iterss = []


    for init in range(number_inits//size):
        print(init)

        model = model_from_json(arch_json_string,custom_objects=custom_objects)

        # import keras.backend as K
        #
        # def sensitivity(y_true, y_pred):
        #     fns=tf.math.reduce_sum((~((y_true==1)^(y_pred>0.5)))&(y_true==1))
        #     return tf.divide(tf.to_float(fns),tf.math.reduce_sum(y_true))
        #     # m=tf.keras.metrics.SensitivityAtSpecificity(0.99)
        #     # # m.update_state(y_true,tf.math.sigmoid(y_pred))
        #     # m.update_state(y_true,y_pred)
        #     # return m.result()
        if optimizer == "langevin":
            #global optimizer
            optimizerr = tfp.optimizer.StochasticGradientLangevinDynamics(learning_rate=0.01)
        else:
            optimizerr = optimizer

        model.compile(optimizer=optimizerr,
                #keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=1e-6),#'sgd',#tf.keras.optimizers.SGD(lr=0.01),
        #model.compile(keras.optimizers.SGD(lr=1e-5),#'sgd',#tf.keras.optimizers.SGD(lr=0.01),
                      #loss='binary_crossentropy',
                      #loss=binary_crossentropy_from_logits,
                      loss=binary_crossentropy_from_logits if loss=="ce" else loss,
                      # loss_weights=[50000],
                      metrics=['accuracy'])
                      #metrics=['accuracy',sensitivity])
                      #metrics=['accuracy',tf.keras.metrics.SensitivityAtSpecificity(0.99),\
                                #tf.keras.metrics.FalsePositives()])

        if network == "fc":
            num_filters = input_dim

        weights, biases = get_weights(model), get_biases(model)
        weights_norm, biases_norm = measure_sigmas(model)
        print(weights_norm,biases_norm)

        model.fit(train_images, ys, verbose=1,\
            sample_weight=sample_weights, validation_data=(train_images, ys), epochs=MAX_TRAIN_EPOCHS,callbacks=callbacks, batch_size=batch_size)

        '''GET DATA: weights, and errors'''
        weights, biases = get_rescaled_weights(model)
        weights_norm, biases_norm = measure_sigmas(model)
        print(weights_norm,biases_norm)

        train_loss, train_acc = model.evaluate(train_images, ys)
        test_loss, test_acc = model.evaluate(test_images, test_ys)
        preds = model.predict(test_images)[:,0]
        # print(preds)
        # print(preds.shape)
        # test_false_positive_rate = test_fps/(len([x for x in test_ys if x==1]))
        def sigmoid(x):
            return np.exp(x)/(1+np.exp(x))
        print(test_ys)
        #for th in np.linspace(0,1,1000):
        for th in np.linspace(0,1,5): # as I'm not exploring unbalanced datasets right now
            test_sensitivity_base = sum([(sigmoid(preds[i])>th)==x for i,x in enumerate(test_ys) if x==1])/(len([x for x in test_ys if x==1]))
            test_specificity_base = sum([(sigmoid(preds[i])>th)==x for i,x in enumerate(test_ys) if x==0])/(len([x for x in test_ys if x==0]))
            if test_specificity_base>0.99:
                break

        # print([preds[i] for i,x in enumerate(test_ys) if x==0 and preds[i]>=1-1e-3])

        print('Test accuracy:', test_acc)
        # print('Test sensitivity (at 99% accuracy):', test_sensitivity)
        print('Test sensitivity:', test_sensitivity_base)
        print('Test specificity:', test_specificity_base)
        # print('Test false positive rate:', test_false_positive_rate)
        test_accs.append(test_acc)
        #test_sensitivities.append(test_sensitivity)
        test_sensitivities_base.append(test_sensitivity_base)
        test_specificities_base.append(test_specificity_base)
        train_accs.append(train_acc)
        weightss.append(weights)
        biasess.append(biases)
        weights_norms.append(weights_norm)
        biases_norms.append(biases_norm)
        iterss.append(model.history.epoch[-1])
        keras.backend.clear_session()

    print("HI")
    test_accs = comm.gather(test_accs, root=0)
    #test_sensitivities = comm.gather(test_sensitivities, root=0)
    test_sensitivities_base = comm.gather(test_sensitivities_base, root=0)
    test_specificities_base = comm.gather(test_specificities_base, root=0)
    train_accs = comm.gather(train_accs, root=0)

    weightss = comm.gather(weightss, root=0)
    biasess = comm.gather(biasess, root=0)
    weights_norms = comm.gather(weights_norms,root=0)
    iterss = comm.gather(iterss, root=0)

    '''PROCESS COLLECTIVE DATA'''
    if rank == 0:
        weightss = np.stack(sum(weightss,[]))
        biasess = np.stack(sum(biasess,[]))
        weights_std = np.mean(np.std(weightss,axis=0)).squeeze()
        biases_std = np.mean(np.std(biasess,axis=0)).squeeze()
        weights_mean = np.mean(weightss)
        biases_mean = np.mean(biasess)
        weights_norm_mean = np.mean(weights_norms)
        weights_norm_std = np.std(weights_norms)
        biases_norm_mean = np.mean(biases_norm)
        biases_norm_std = np.std(biases_norm)

        test_acc = np.mean(np.array(test_accs))
        #test_sensitivity = np.mean(np.array(test_sensitivities))
        test_sensitivity_base = np.mean(np.array(test_sensitivities_base))
        test_specificity_base = np.mean(np.array(test_specificities_base))
        print('Mean test accuracy:', test_acc)
        # print('Mean test sensitivity:', test_sensitivity)
        print('Mean test sensitivity:', test_sensitivity_base)
        print('Mean test specificity:', test_specificity_base)
        train_acc = np.mean(np.array(train_accs))
        print('Mean train accuracy:', train_acc)
        test_acc = np.mean(np.array(test_accs))
        train_acc = np.mean(np.array(train_accs))
        train_acc_std = np.std(np.array(test_accs))
        test_acc_std = np.std(np.array(train_accs))
        mean_iters = np.mean(iterss)

        # useful_flags = FLAGS.copy()
        # if "data_dir" in useful_flags: del useful_flags["data_dir"]
        # if "helpfull" in useful_flags: del useful_flags["helpfull"]
        # if "help" in useful_flags: del useful_flags["help"]
        # if "helpshort" in useful_flags: del useful_flags["helpshort"]
        # if "h" in useful_flags: del useful_flags["h"]
        # if "f" in useful_flags: del useful_flags["f"]
        # if "prefix" in useful_flags: del useful_flags["prefix"]
        useful_flags = ["dataset", "m", "network", "pooling", "number_layers", "sigmaw", "sigmab", "init_dist","whitening", "centering", "oversampling", "oversampling2", "channel_normalization", "training", "binarized", "confusion","filter_sizes", "gamma", "intermediate_pooling", "label_corruption", "threshold", "n_gpus", "n_samples_repeats", "num_filters", "number_inits", "padding"]
        with open(prefix+"nn_training_results.txt","a") as file:
            file.write("#")
            for key in sorted(useful_flags):
                file.write("{}\t".format(key))
            file.write("\t".join(["train_acc", "test_error", "test_acc","test_sensitivity","test_specificity","weights_std","biases_std","weights_mean", "biases_mean", "weights_norm_mean","weights_norm_std","biases_norm_mean","biases_norm_std","mean_iters","train_acc_std","test_acc_std"]))
            file.write("\n")
            for key in sorted(useful_flags):
                file.write("{}\t".format(FLAGS[key]))
            file.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:d}\t{:.4f}\t{:.4f}\n".format(train_acc, 1-test_acc,test_acc,\
                test_sensitivity_base,test_specificity_base,weights_std,biases_std,\
                weights_mean,biases_mean,weights_norm_mean,weights_norm_std,biases_norm_mean,biases_norm_std,int(mean_iters),train_acc_std,test_acc_std)) #normalized to sqrt(input_dim)
    else:
        assert test_accs is None
        assert train_accs is None
    # for i in range(number_inits//20):
        # Parallel(n_jobs=20)(train_net(init,ys,test_ys,train_images,test_images) for init in range(number_inits))


if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    define_default_flags(f)

    f.DEFINE_integer('number_inits',1,"Number of initializations")
    f.DEFINE_integer('batch_size',32,"batch_size")
    f.DEFINE_integer('epochs_after_fit',1,"Number of epochs to wait after it first reacehs 100% accuracy")
    f.DEFINE_float('gamma',1.0,"weight for confusion samples (1.0 weigths them the same as normal samples)")
    f.DEFINE_string('optimizer',"sgd","Which optimizer to use (keras optimizers available)")
    f.DEFINE_string('loss',"ce","Which loss to use (ce/mse/etc)")

    tf.compat.v1.app.run()
    import gc; gc.collect()
