import numpy as np
import tensorflow as tf
import keras
import os
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import h5py
import pickle
from initialization import get_all_layers, is_normalization_layer, reset_weights, simple_reset_weights

arch_folder = "archs/"
data_folder = "data/"
kernel_folder = "kernels/"
results_folder = "results/"

def main(_):

    FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()
    from utils import preprocess_flags
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(rank)
    num_tasks = number_samples

    num_gpus = n_gpus
    print("num_gpus",num_gpus)

    num_tasks_per_job = num_tasks//size
    tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))

    if rank < num_tasks%size:
        tasks.append(size*num_tasks_per_job+rank)

    if num_gpus>0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(rank%num_gpus)

    config = tf.compat.v1.ConfigProto()
    if num_gpus > 0:
        config.gpu_options.allow_growth = True

    tf.compat.v1.enable_eager_execution(config=config)

    '''some custom initalizers and keras setup'''
    from utils import cauchy_init_class_wrapper, shifted_init_class_wrapper
    CauchyInit = cauchy_init_class_wrapper(sigmaw)
    ShiftedInit = shifted_init_class_wrapper(sigmab,shifted_init_shift)
    custom_objects = {'cauchy_init': CauchyInit, 'shifted_init':ShiftedInit}

    '''LOAD DATA & ARCHITECTURE'''

    from utils import load_data,load_model,load_kernel,entropy
    data,flat_data,_,_,_ = load_data(FLAGS)
    data = tf.constant(data)
    input_dim = data.shape[1]
    num_channels = data.shape[-1]

    arch_json_string = load_model(FLAGS)
    from tensorflow.keras.models import model_from_json
    model = model_from_json(arch_json_string, custom_objects=custom_objects)

    #K = load_kernel(FLAGS)
    #from GP_prob.GP_prob_gpy import GP_prob
    #def calculate_logPU(preds):
    #    logPU = GP_prob(K,flat_data,preds )
    #    return logPU

    print("Doing task %d of %d" % (rank, size))
    import time
    start_time = time.time()

    index_fun_probs = []
    fun_probs = {}

    if FLAGS["pooling"] is None:
        pooling_flag = "none"
    else:
        pooling_flag = FLAGS["pooling"]
    outfilename = results_folder+"index_funs_probs_"+str(rank)+"_"+FLAGS["prefix"]+"_"+str(FLAGS["shifted_init_shift"])+"_"+FLAGS["dataset"]+"_"+FLAGS["network"]+"_"+str(FLAGS["number_layers"])+"_"+pooling_flag+"_"+FLAGS["intermediate_pooling"]+".txt"

    if network not in ["cnn", "fc"]:
        layers = get_all_layers(model)
        are_norm = [is_normalization_layer(l) for l in layers for w in l.get_weights()]
        initial_weights = model.get_weights()

    local_index = 0

    '''SAMPLING LOOP'''
    for index in tasks:
        outfile = open(outfilename, "a")
        print(index)
        if local_index>0:
            if network in ["cnn", "fc"]:
                simple_reset_weights(model, sigmaw, sigmab)
            else:
                reset_weights(model, initial_weights, are_norm, sigmaw, sigmab)
        #model = model_from_json(arch_json_string) # this resets the weights (makes sense as the json string only has architecture)

        #save weights?
        #model.save_weights("sampled_nets/"+str(index)+"_"+json_string_filename+".h5")

        #predictions = tf.keras.backend.eval(model(data)) > 0
        predictions = model.predict(data) > 0
        fstring = "".join([str(int(x[0])) for x in predictions])
        n1s = len([x for x in fstring if x == "1"])
        ent = entropy(fstring)

        #if fstring not in fun_probs:
        #    fun_probs[fstring] = calculate_logPU(predictions)
        #index_fun_probs.append((index,ent,fstring,fun_probs[fstring]))
        #index_fun_probs.append((index,ent,fstring))
        outfile.write(str(index)+"\t"+fstring+"\t"+str(ent)+"\t"+str(n1s)+"\n")
        outfile.close()
        #keras.backend.clear_session()
        local_index+=1

    print("--- %s seconds ---" % (time.time() - start_time))

    index_fun_probs = comm.gather(index_fun_probs,root=0)

    #if rank == 0:
    #    index_fun_probs = sum(index_fun_probs, [])
    #    #print(FLAGS["pooling"])
    #    if FLAGS["pooling"] is None:
    #        pooling_flag = "none"
    #    else:
    #        pooling_flag = FLAGS["pooling"]
    #    with open(results_folder+"index_funs_probs_"+FLAGS["prefix"]+"_"+FLAGS["dataset"]+"_"+FLAGS["network"]+"_"+str(FLAGS["number_layers"])+"_"+pooling_flag+"_"+FLAGS["intermediate_pooling"]+".txt","w") as f:
    #        #for index,ent,fstring,logProb in index_fun_probs:
    #        for index,ent,fstring in index_fun_probs:
    #                #f.write(str(index)+"\t"+fstring+"\t"+str(ent)+"\t"+str(logProb)+"\n")
    #                f.write(str(index)+"\t"+fstring+"\t"+str(ent)+"\n")

if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    f.DEFINE_integer('number_samples', None, "Number of samples")
    define_default_flags(f)

    tf.compat.v1.app.run()
