
import numpy as np
import tensorflow as tf
import keras

#import sys
import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.chdir("/users/guillefix/bias/nn_bias/CSR")
#tf.enable_eager_execution()
import h5py
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

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)
    # num_inits_per_task = 1
    num_tasks = number_samples

    #from tensorflow.python.client import device_lib

    #def get_available_gpus():
    #    local_device_protos = device_lib.list_local_devices()
    #    return [x.name for x in local_device_protos if x.device_type == 'GPU']

    #num_gpus = len(get_available_gpus())
    num_gpus = n_gpus
    print("num_gpus",num_gpus)

    num_tasks_per_job = num_tasks//size
    tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))

    if rank < num_tasks%size:
        tasks.append(size*num_tasks_per_job+rank)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(rank%num_gpus)

    config = tf.ConfigProto()
    if num_gpus > 0:
        #config = tf.ConfigProto(device_count={'GPU': rank%num_gpus})
        #config.device_count = {'GPU': rank%num_gpus}
        config.gpu_options.allow_growth = True
        #config.gpu_options.visible_device_list = str(rank%num_gpus)

    tf.enable_eager_execution(config=config)

    # total_samples = m

    '''LOAD DATA & ARCHITECTURE'''

    from utils import load_data,load_model,load_kernel
    data,flat_data,_,_,_ = load_data(FLAGS)
    data = tf.constant(data)
    input_dim = data.shape[1]
    num_channels = data.shape[-1]

    arch_json_string = load_model(FLAGS)
    from tensorflow.keras.models import model_from_json
    model = model_from_json(arch_json_string)

    #K = load_kernel(FLAGS)

    #sess = tf.Session()

    #from keras.initializers import lecun_normal  # Or your initializer of choice

    def reset_weights(model):
        initial_weights = model.get_weights()
        def initialize_var(shape):
            if len(shape) == 1:
               #return tf.random.normal(shape).eval(session=sess)
               return np.random.normal(0,1,shape)
            else:
                return np.random.normal(0,1.0/np.sqrt(np.prod(shape[:-1])),shape)
        new_weights = [initialize_var(w.shape) for w in initial_weights]
        model.set_weights(new_weights)

    from GP_prob_gpy import GP_prob

    def calculate_logPU(preds):
        logPU = GP_prob(K,flat_data,preds )
        return logPU

    from math import log

    def log2(x):
        return log(x)/log(2.0)

    def entropy(f):
        n0=0
        n=len(f)
        for char in f:
            if char=='0':
                n0+=1
        n1=n-n0
        if n1 > 0 and n0 > 0:
            return log2(n) - (1.0/n)*(n0*log2(n0)+n1*log2(n1))
        else:
            return 0


    #%%

    print("Doing task %d of %d" % (rank, size))
    import time
    start_time = time.time()

    index_fun_probs = []
    fun_probs = {}

    for index in tasks:
        print(index)
        # index = rank*num_inits_per_task+i
        reset_weights(model)
        #model = model_from_json(arch_json_string) # this resets the weights (makes sense as the json string only has architecture)

        #save weights?
        #model.save_weights("sampled_nets/"+str(index)+"_"+json_string_filename+".h5")

        #predictions = tf.keras.backend.eval(model(data)) > 0
        # if network == "resnet":
        #     predictions = model.predict(data,steps=1) > 0.5 #because resnet is defined with sigmoid as output, rather than logits (because I use a module that requires that :P)
        # else:

        predictions = model.predict(data) > 0
        fstring = "".join([str(int(x[0])) for x in predictions])
        ent = entropy(fstring)

        #if fstring not in fun_probs:
        #    fun_probs[fstring] = calculate_logPU(predictions)
        #index_fun_probs.append((index,ent,fstring,fun_probs[fstring]))
        
        index_fun_probs.append((index,ent,fstring))
        #keras.backend.clear_session()
        #del model

    print("--- %s seconds ---" % (time.time() - start_time))

    index_fun_probs = comm.gather(index_fun_probs,root=0)

    if rank == 0:
        index_fun_probs = sum(index_fun_probs, [])
        with open(results_folder+"index_funs_probs_"+FLAGS["prefix"]+"_"+FLAGS["dataset"]+"_"+FLAGS["network"]+"_"+str(FLAGS["number_layers"])+"_"+FLAGS["pooling"]+"_"+FLAGS["intermediate_pooling"]+".txt","w") as f:
            #for index,ent,fstring,logProb in index_fun_probs:
            for index,ent,fstring in index_fun_probs:
                    #f.write(str(index)+"\t"+fstring+"\t"+str(ent)+"\t"+str(logProb)+"\n")
                    f.write(str(index)+"\t"+fstring+"\t"+str(ent)+"\n")

if __name__ == '__main__':

    f = tf.app.flags

    from utils import define_default_flags

    f.DEFINE_integer('number_samples', None, "Number of samples")
    define_default_flags(f)

    tf.app.run()
