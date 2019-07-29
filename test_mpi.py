import os
import tensorflow as tf

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

n_gpus=1
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False  # to log device placement (on which device the operation ran)
#os.environ["CUDA_VISIBLE_DEVICES"]=str((rank)%n_gpus)
os.environ["CUDA_VISIBLE_DEVICES"]=str(0)

sess = tf.Session(config=config)

import h5py
h5py.File("test","w")

print("hi", rank)
