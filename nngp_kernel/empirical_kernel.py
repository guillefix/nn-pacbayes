import numpy as np
import tensorflow as tf
import keras

import os,sys
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.chdir("/users/guillefix/bias/nn_bias/CSR")
#tf.enable_eager_execution()
import h5py
import pickle
from tensorflow.keras import backend as K


arch_folder = "archs/"
data_folder = "data/"
kernel_folder = "kernels/"
results_folder = "results/"

def empirical_K(arch_json_string, data, number_samples,sigmaw=1.0,sigmab=1.0,n_gpus=1,sess=None):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)
    num_tasks = number_samples

    save_freq = 5000
    # we can use checkpoints (TODO: make this work again with nice option and stuff)
    try:
        chkpt = pickle.load(open("checkpoint.p","rb"))
        print("getting checkopoint of "+str(chkpt)+" functions")
    except IOError:
        chkpt = 0
    if rank == 0:
        try:
            fs_init = pickle.load(open("fs.p","rb"))
        except IOError:
            fs_init = []

    num_tasks = num_tasks - chkpt

    num_tasks_per_job = num_tasks//size
    tasks = list(range(int(rank*num_tasks_per_job),int((rank+1)*num_tasks_per_job)))

    if rank < num_tasks%size:
        tasks.append(size*num_tasks_per_job+rank)

    print("Doing task %d of %d" % (rank, size))
    import time
    start_time = time.time()

    from tensorflow.keras.models import model_from_json
    model = model_from_json(arch_json_string) # this resets the weights (makes sense as the json string only has architecture)

    from initialization import get_all_layers, is_normalization_layer, reset_weights

    #fs = []
    covs = np.zeros((len(data),len(data)))
    last_layer = model.layers[-1].input
    print(last_layer.shape)
    func = K.function(model.input,last_layer)
    local_index = 0
    layers = get_all_layers(model)
    are_norm = [is_normalization_layer(l) for l in layers for w in l.get_weights()]
    initial_weights = model.get_weights()
    for index in tasks:
        print("sample for kernel", index)

        # model = model_from_json(arch_json_string) # this resets the weights (makes sense as the json string only has architecture)
        if local_index>0:
            reset_weights(model, initial_weights, are_norm, sigmaw, sigmab)

        X = np.squeeze(func(data))
        print("X",X)
        if len(X.shape)==1:
            X = np.expand_dims(X,0)
        covs += (sigmaw**2/X.shape[1])*np.matmul(X,X.T)+(sigmab**2)*np.ones((X.shape[0],X.shape[0]))
        #outputs = model.predict(data)
        #print(outputs)

        # print(outputs)
        #fs.append(outputs)
        #if index % save_freq == save_freq-1:
        #    fs_tmp = comm.gather(fs,root=0)
        #    if rank == 0:
        #        fs_tmp = sum(fs_tmp, [])
        #        fs_tmp += fs_init
        #        pickle.dump(fs_tmp,open("fs.p","wb"))
        #        pickle.dump(len(fs_tmp),open("checkpoint.p","wb"))
        sys.stdout.flush()
        local_index += 1

    print("--- %s seconds ---" % (time.time() - start_time))

    #fs = comm.gather(fs,root=0)
    covs = comm.gather(covs,root=0)

    if rank == 0:
        #fs = sum(fs, [])
        #covs = sum(covs, [])
        #fs += fs_init
        #fs = np.array(fs)
        #fs = np.squeeze(fs)
        #return np.cov(fs.T)
        return np.sum(covs,axis=0)/number_samples
    else:
        return None
