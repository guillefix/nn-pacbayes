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
import gc


arch_folder = "archs/"
data_folder = "data/"
kernel_folder = "kernels/"
results_folder = "results/"

def empirical_K(arch_json_string, data, number_samples,sigmaw=1.0,sigmab=1.0,n_gpus=1, empirical_kernel_batch_size=256, sess=None, truncated_init_dist=False, data_parallelism=False):

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)
    num_tasks = number_samples

    if data_parallelism:
        num_tasks *= n_gpus

    save_freq = 5000
    # we can use checkpoints (TODO: make this work again with nice option and stuff)
    #try:
    #    chkpt = pickle.load(open("checkpoint.p","rb"))
    #    print("getting checkopoint of "+str(chkpt)+" functions")
    #except IOError:
    #    chkpt = 0
    #if rank == 0:
    #    try:
    #        fs_init = pickle.load(open("fs.p","rb"))
    #    except IOError:
    #        fs_init = []

    #num_tasks = num_tasks - chkpt

    num_tasks_per_job = num_tasks//size
    tasks = list(range(int(rank*num_tasks_per_job),int((rank+1)*num_tasks_per_job)))

    if rank < num_tasks%size:
        tasks.append(size*num_tasks_per_job+rank)

    print("Doing task %d of %d" % (rank, size))
    import time

    from tensorflow.keras.models import model_from_json
    model = model_from_json(arch_json_string) # this resets the weights (makes sense as the json string only has architecture)

    from initialization import get_all_layers, is_normalization_layer, reset_weights

    #fs = []
    covs = np.zeros((len(data),len(data)),dtype=np.float32)
    #last_layer = model.layers[-1].input
    #print(last_layer.shape)
    #func = K.function(model.input,last_layer)
    #from keras.models import Model, Input
    ##model = Model(inputs=model.input, outputs=last_layer)
    #input_tensor = Input(batch_shape=model.input.shape)
    #output_tensor = model(input_tensor)
    #model = Model(inputs=input_tensor, outputs=output_tensor)
    #model.layers.pop(-1)
    #from keras.models import Sequential
    #new_model = Sequential()
    #for layer in model.layers[:-1]:
    #    new_model.add(layer)
    #model = new_model
    model.pop()
    local_index = 0
    layers = get_all_layers(model)
    are_norm = [is_normalization_layer(l) for l in layers for w in l.get_weights()]
    initial_weights = model.get_weights()
    update_chunk = 20000
    num_chunks = covs.shape[0]//update_chunk
    print("num_chunks",num_chunks)
    for index in tasks:
        start_time = time.time()
        print("sample for kernel", index)

        # model = model_from_json(arch_json_string) # this resets the weights (makes sense as the json string only has architecture)
        if local_index>0:
            reset_weights(model, initial_weights, are_norm, sigmaw, sigmab, truncated_init_dist)

        #X = np.squeeze(func(data))
        X = model.predict(data, batch_size=min(empirical_kernel_batch_size, len(data))).astype(np.float32)
        print("X",X)
        if len(X.shape)==1:
            X = np.expand_dims(X,0)
        #covs += (sigmaw**2/X.shape[1])*np.matmul(X,X.T)+(sigmab**2)*np.ones((X.shape[0],X.shape[0]), dtype=np.float32)
        if covs.shape[0] > update_chunk:
            for i in range(num_chunks):
                covs[i*update_chunk:(i+1)*update_chunk] += (sigmaw**2/X.shape[1])*np.matmul(X[i*update_chunk:(i+1)*update_chunk],X.T)+(sigmab**2)*np.ones((update_chunk,X.shape[0]), dtype=np.float32)
            last_bits = slice(update_chunk*num_chunks,covs.shape[0])
            covs[last_bits] += (sigmaw**2/X.shape[1])*np.matmul(X[last_bits],X.T)+(sigmab**2)*np.ones((last_bits.stop-last_bits.start,X.shape[0]), dtype=np.float32)
        else:
            covs += (sigmaw**2/X.shape[1])*np.matmul(X,X.T)+(sigmab**2)*np.ones((X.shape[0],X.shape[0]), dtype=np.float32)
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
        #gc.collect()

        print("--- %s seconds ---" % (time.time() - start_time))

    #fs = comm.gather(fs,root=0)
    if size > 1:
        covs1_recv = None
        covs2_recv = None
        if rank == 0:
            covs1_recv = np.zeros_like(covs[:25000,:])
            covs2_recv = np.zeros_like(covs[25000:,:])
        print(covs[25000:,:])
        comm.Reduce(covs[:25000,:], covs1_recv, op=MPI.SUM, root=0)
        comm.Reduce(covs[25000:,:], covs2_recv, op=MPI.SUM, root=0)

        if rank == 0:
            #fs = sum(fs, [])
            #covs = sum(covs, [])
            #fs += fs_init
            #fs = np.array(fs)
            #fs = np.squeeze(fs)
            #return np.cov(fs.T)
            covs_recv = np.concatenate([covs1_recv,covs2_recv],0)
            return covs_recv/number_samples
        else:
            return None
    else:
        return covs/number_samples
