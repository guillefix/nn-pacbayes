#!/usr/bin/python
import sys
import numpy as np
import tensorflow as tf
import keras
import h5py
import os
#os.chdir("/users/guillefix/bias/nn_bias/CSR")
import pickle


arch_folder = "archs/"
data_folder = "data/"
kernel_folder = "kernels/"
results_folder = "results/"


def main(_):

    FLAGS = tf.app.flags.FLAGS.flag_values_dict()
    from utils import preprocess_flags
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    # total_samples = m

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)
    # num_inits_per_task = 1
    #num_tasks = int(sys.argv[1])
    num_tasks = number_samples

    #from tensorflow.python.client import device_lib
    #
    #def get_available_gpus():
    #    local_device_protos = device_lib.list_local_devices()
    #    return [x.name for x in local_device_protos if x.device_type == 'GPU']
    #
    #num_gpus = len(get_available_gpus())
    num_gpus = n_gpus

    num_tasks_per_job = num_tasks//size
    tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))

    if rank < num_tasks%size:
        tasks.append(size*num_tasks_per_job+rank)

    #config = tf.ConfigProto(device_count={'GPU': rank%num_gpus})
    config = tf.ConfigProto()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(rank%num_gpus)
    config.gpu_options.allow_growth = True
    tf.enable_eager_execution(config=config)

    from utils import load_data,load_model,load_kernel
    data,flat_data,_,_,_ = load_data(FLAGS)
    data = tf.constant(data)
    model = load_model(FLAGS)
    K = load_kernel(FLAGS)

    def lass(model,x,r=0.01):
        pred = tf.sign(model(x))
        alpha=0.5
        #alpha=0.25
        #beta=0.2
        deltax = tf.zeros(x.shape)
        xtilde = x+deltax
        max_iters = 20
        iterr = 0
        while iterr < max_iters:
            with tf.GradientTape() as g:
                g.watch(xtilde)
                y = model(xtilde)
            grads = g.gradient(y,xtilde)
            delta = alpha * tf.sign(-pred*grads) #+ beta*tf.random.normal(x.shape)
            deltax += delta
            deltax = tf.clip_by_value(deltax,-r,r)
            # deltax -= tf.to_float(tf.math.abs(deltax) >= r) * tf.clip_by_value(deltax,-r,r)
            xtilde = x+deltax
            # print(grads)

            if tf.sign(model(xtilde)).numpy()[0] != pred.numpy()[0]:
                return True
            iterr += 1
        return False

    def crit_sample_ratio(model,xs,r=0.01): # is 0.3 fine for a 0-1 scaling, when they say 0-255 what do they mean? Hmm
        crit_samples = 0
        for i in range(int(xs.shape[0])):
            #print(i)
            # print(xs[i:i+1,:,:,:])
            if lass(model,xs[i:i+1,:,:,:],r):
                crit_samples += 1
        return 1.0*crit_samples/int(xs.shape[0])

    #%%

    print("Beginning job %d of %d" % (rank, size))
    import time
    start_time = time.time()
    crit_sample_ratios = []
    #probs = []
    for index in tasks:
        print(index)
        model.load_weights("./sampled_nets/"+str(index)+"_"+json_string_filename+".h5")
        csr = crit_sample_ratio(model,data,r=0.03)
        crit_sample_ratios.append((index, csr))
        with open(results_folder+"CSRs_"+FLAGS["prefix"]+"_"+FLAGS["dataset"]+"_"+FLAGS["network"]+"_"+str(FLAGS["number_layers"])+"_"+FLAGS["pooling"]+"_"+FLAGS["intermediate_pooling"]+".txt","a") as f:
            f.write(str(index)+"\t"+str(csr)+"\n")
        #print(csr)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Finishing job %d of %d" % (rank, size))

    csr_data = comm.gather(crit_sample_ratios,root=0)

    #tf.keras.initializers.glorot_uniform

    if rank == 0:
        csr_data = sum(csr_data, [])
        pickle.dump(csr_data,open(results_folder+"CSRs_"+FLAGS["prefix"]+"_"+FLAGS["dataset"]+"_"+FLAGS["network"]+"_"+str(FLAGS["number_layers"])+"_"+FLAGS["pooling"]+"_"+FLAGS["intermediate_pooling"]+".p","wb"))
        #with open(results_folder+"CSRs_"+FLAGS["prefix"]+"_"+FLAGS["dataset"]+"_"+FLAGS["network"]+"_"+str(FLAGS["number_layers"])+"_"+FLAGS["pooling"]+"_"+FLAGS["intermediate_pooling"]+".txt","w") as f:
        #    for index,csr in csr_data:
        #            f.write(str(index)+"\t"+str(csr)+"\n")

if __name__ == '__main__':

    f = tf.app.flags

    from utils import define_default_flags

    define_default_flags(f)

    tf.app.run()
