import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.keras as keras
import pickle
import os
from math import ceil

from utils import preprocess_flags, save_kernel, save_kernel_partial, load_kernel_by_filename, save_kernel_partial, find_partial_kernel_filenames, kernel_filename
from utils import load_data,load_model,load_model_json,load_kernel
from utils import data_folder,kernel_folder,arch_folder
import glob
from time import sleep

def main(_):

    FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    print(partial_kernel_n_proc)

    number_completed = 0

    files = find_partial_kernel_filenames(FLAGS)
    for f in files:
        cnt = int(f.split("_")[-2])
        if cnt > partial_kernel_n_proc:
            number_completed = cnt - partial_kernel_n_proc
        if f == kernel_filename(FLAGS):
            number_completed = partial_kernel_n_proc
            #break
    while number_completed < partial_kernel_n_proc:
        files = find_partial_kernel_filenames(FLAGS)
        print(files)
        if len(files) > 1:
            for i,f in enumerate(files):
                print(f)
                if i == 0:
                    cov = load_kernel_by_filename(f)
                    if number_completed == 0:
                        number_completed += 1
                else:
                    cov += load_kernel_by_filename(f)
                    number_completed += 1
            if number_completed >= partial_kernel_n_proc:
                save_kernel(cov/number_completed, FLAGS)
            else:
                save_kernel_partial(cov,FLAGS,partial_kernel_n_proc+number_completed)
            for f in files:
                os.remove(f)
        sleep(100)

if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    define_default_flags(f)
    f.DEFINE_boolean('compute_for_GP_train', False, "Whether to add a bit of test set to kernel, to be able to use it for GP training")
    f.DEFINE_boolean('store_partial_kernel', False, "Whether to store the kernels partially on a file to free the processes")
    f.DEFINE_integer('empirical_kernel_batch_size', 256, "batch size to use when computing the empirical kernel, larger models need smaller values, but smaller models can use larger values")
    f.DEFINE_integer('partial_kernel_n_proc', 175, "number of processes over which we are parallelizing the when computing partial kernels and saving")
    f.DEFINE_integer('partial_kernel_index', 0, "index of the process when using partial_kernels method")

    tf.compat.v1.app.run()
