import numpy as np
import tensorflow as tf
import tensorflow.compat.v1.keras as keras
import pickle
import os
from math import ceil

from utils import preprocess_flags, save_kernel
from utils import load_data,load_model,load_model_json,load_kernel
from utils import data_folder,kernel_folder,arch_folder

#%%
f = tf.compat.v1.app.flags

from utils import define_default_flags

define_default_flags(f)

FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()

FLAGS.keys()
FLAGS["network"] = "densenet121"
FLAGS["prefix"] = "test_"
FLAGS["number_layers"] = 4
FLAGS["dataset"] = "mnist"
FLAGS["m"] = 100
FLAGS["sigmaw"] = 1.41
FLAGS["sigmab"] = 0.1
FLAGS["pooling"] = "none"

FLAGS = preprocess_flags(FLAGS)
globals().update(FLAGS)
#%%

if n_gpus>0:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(1)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

set_session = keras.backend.set_session
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

train_images,flat_train_images,ys,_,_ = load_data(FLAGS)
image_size = train_images.shape[1]
number_channels = train_images.shape[-1]
input_dim = flat_train_images.shape[1]
#%%

model = load_model(FLAGS)

model.layers[0].layers[2].get_config()
