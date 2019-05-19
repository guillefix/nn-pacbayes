import os
import tensorflow as tf

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

n_gpus=8
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False  # to log device placement (on which device the operation ran)
os.environ["CUDA_VISIBLE_DEVICES"]=str((rank)%n_gpus)

sess = tf.Session(config=config)

print("hi", rank)
