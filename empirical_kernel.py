import numpy as np
import tensorflow as tf
import keras

import os,sys
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# os.chdir("/users/guillefix/bias/nn_bias/CSR")
#tf.enable_eager_execution()
import h5py
import pickle


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

    #from tensorflow.python.client import device_lib
    #local_device_protos = device_lib.list_local_devices()
    #print(local_device_protos)
    #def get_available_gpus():
    #    local_device_protos = device_lib.list_local_devices()
    #    return [x.name for x in local_device_protos if x.device_type == 'GPU']

    #num_gpus = len(get_available_gpus())
    #num_gpus = n_gpus
    #print("num_gpus",num_gpus)

    save_freq = 500
    try:
        chkpt = pickle.load(open("checkpoint.p","rb"))
        print("getting checkopoint of "+str(chkpt)+" functions")
    except FileNotFoundError:
        chkpt = 0
    if rank == 0:
        try:
            fs_init = pickle.load(open("fs.p","rb"))
        except FileNotFoundError:
            fs_init = []

    num_tasks = num_tasks - chkpt

    num_tasks_per_job = num_tasks//size
    tasks = list(range(rank*num_tasks_per_job,(rank+1)*num_tasks_per_job))

    if rank < num_tasks%size:
        tasks.append(size*num_tasks_per_job+rank)

    #os.environ["CUDA_VISIBLE_DEVICES"]=str(rank%n_gpus)
    #print(rank%n_gpus)

#    config = tf.ConfigProto()
#    if n_gpus > 0:
#        #config = tf.ConfigProto(device_count={'GPU': rank%num_gpus})
#        #config.device_count = {'GPU': rank%num_gpus}
#        # config.gpu_options.allow_growth = True
#        #config.gpu_options.per_process_gpu_memory_fraction = 0.5
#        config.gpu_options.visible_device_list = str(rank%n_gpus)
#
#    #tf.enable_eager_execution(config=config)
    #set_session = keras.backend.set_session
    #config.log_device_placement = False  # to log device placement (on which device the operation ran)
    #sess = tf.Session(config=config)
    #set_session(sess)  # set this TensorFlow session as the default session for Keras

    #data = tf.constant(data)

    print("Doing task %d of %d" % (rank, size))
    import time
    start_time = time.time()

    from tensorflow.keras.models import model_from_json
    model = model_from_json(arch_json_string) # this resets the weights (makes sense as the json string only has architecture)

    # from keras.initializers import lecun_normal  # Or your initializer of choice

    # k_eval = lambda placeholder: placeholder.eval(session=keras.backend.get_session())
    initial_weights = model.get_weights()

    def reset_weights(model):
        def initialize_var(shape):
            if len(shape) == 1:
                #return tf.random.normal(shape,stddev=sigmab).eval(session=sess)
                return np.random.normal(0,sigmab,shape)
            else:
                #return tf.random.normal(shape,stddev=1.0/np.sqrt(np.prod(shape[:-1]))).eval(session=sess)
                #return np.random.normal(0,1.0/np.sqrt(np.prod(shape[:-1])),shape)
                return np.random.normal(0,sigmaw/np.sqrt(shape[-2]),shape) #assumes NHWC so that we divide by number of channels as in GP limit
        new_weights = [initialize_var(w.shape) for w in initial_weights]
        # new_weights = [k_eval(lecun_normal()(w.shape)) for w in initial_weights]
        model.set_weights(new_weights)

    fs = []
    for index in tasks:
        print("sample for kernel", index)
        sys.stdout.flush()
        reset_weights(model)
        #model.save_weights("sampled_nets/"+str(index)+"_"+json_string_filename+".h5")
        #outputs = model.predict(data,batch_size=data.shape[0])
        #outputs = model.predict(data,steps=1)
        outputs = model.predict(data)
        #keras.backend.clear_session()
        # print(outputs)
        fs.append(outputs)
        if index % save_freq == save_freq-1:
            fs_tmp = comm.gather(fs,root=0)
            if rank == 0:
                fs_tmp = sum(fs_tmp, [])
                fs_tmp += fs_init
                pickle.dump(fs_tmp,open("fs.p","wb"))
                pickle.dump(len(fs_tmp),open("checkpoint.p","wb"))

    print("--- %s seconds ---" % (time.time() - start_time))

    fs = comm.gather(fs,root=0)

    if rank == 0:
        fs = sum(fs, [])
        fs += fs_init
        fs = np.array(fs)
        fs = np.squeeze(fs)
        # print(fs.shape)
        # print(fs)
        return np.cov(fs.T)
    else:
        return None
