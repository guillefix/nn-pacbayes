import numpy as np
import tensorflow as tf

from utils import preprocess_flags, save_arch
from utils import arch_folder

def main(_):

    FLAGS = tf.compat.v1.app.flags.FLAGS.flag_values_dict()
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)
    print("poolin", pooling)

    print("Generating architecture", network, number_layers)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    import os
    if n_gpus>0:
        os.environ["CUDA_VISIBLE_DEVICES"]=str((rank)%n_gpus)

    from tensorflow import keras
    #import keras
    import keras_applications
    keras_applications._KERAS_BACKEND = keras.backend
    keras_applications._KERAS_LAYERS = keras.layers
    keras_applications._KERAS_MODELS = keras.models
    keras_applications._KERAS_UTILS = keras.utils
    import warnings

    num_classes=2
    #TODO: make code compatible with non-binary

    # %%

    if dataset == "cifar":
            image_size=32
            number_channels=3
    elif dataset == "imagenet":
            image_size=256
            number_channels=3
    elif dataset == "mnist":
            image_size=28
            number_channels=1
    elif dataset == "mnist-fashion":
            image_size=28
            number_channels=1
    elif dataset == "KMNIST":
            image_size=28
            number_channels=1
    elif dataset == "EMNIST":
            image_size=28
            number_channels=1
    elif dataset == "boolean":
        if boolean_input_dim is not None:
            input_dim = boolean_input_dim
        else:
            input_dim = 7
    elif dataset == "calabiyau":
        input_dim = 180
    elif dataset == "ion":
        input_dim = 34
    else:
        raise NotImplementedError

    if not(dataset == "boolean" or dataset == "calabiyau" or dataset == "ion"):
        image_height = image_size
        image_width = image_size
        input_dim = image_height*image_width*number_channels
    set_session = tf.compat.v1.keras.backend.set_session

    from utils import cauchy_init_wrapper,shifted_init_wrapper

    if init_dist == "gaussian":
        bias_initializer = keras.initializers.RandomNormal(stddev=sigmab)
        # weight_initializer = keras.initializers.RandomNormal(stddev=sigmaw/np.sqrt(input_dim))
        weight_initializer = keras.initializers.VarianceScaling(scale=sigmaw**2, mode='fan_in', distribution='normal', seed=None)
        if use_shifted_init:
            bias_initializer_last_layer = shifted_init_wrapper(sigmab,shifted_init_shift)
        else:
            bias_initializer_last_layer = bias_initializer
    elif init_dist == "cauchy":
        bias_initializer = cauchy_init_wrapper(sigmab)
        weight_initializer  = _wrapper(sigmaw)
        bias_initializer_last_layer = bias_initializer
    elif init_dist == "uniform":
        bias_initializer = keras.initializers.RandomUniform(minval=-np.sqrt(3 * sigmab), maxval=np.sqrt(3 * sigmab), seed=None)
        weight_initializer = keras.initializers.VarianceScaling(scale=sigmaw, mode='fan_in', distribution='uniform', seed=None)
        bias_initializer_last_layer = bias_initializer
    else:
        raise NotImplementedError
    # bias_initializer = keras.initializers.Zeros()
    # weight_initializer = keras.initializers.glorot_uniform()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    # config.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    activations_dict = {"relu":tf.nn.relu, "tanh":tf.nn.tanh}

    if network == "cnn":
        if intermediate_pooling_type=="avg":
            intermediate_pooling_layer = [keras.layers.AvgPool2D(pool_size=2, padding='same')]
        elif intermediate_pooling_type=="max":
            intermediate_pooling_layer = [keras.layers.MaxPool2D(pool_size=2, padding='same')]
        else:
            intermediate_pooling_layer = []

        if pooling=="avg":
            pooling_layer = [keras.layers.GlobalAveragePooling2D()]
        elif pooling=="max":
            pooling_layer = [keras.layers.GlobalMaxPooling2D()]
        else:
            pooling_layer = []
        model = keras.Sequential(
            sum([
                [keras.layers.Conv2D(input_shape=(image_height,image_width,number_channels) if index==0 else (None,), \
                    filters=num_filters, \
                    kernel_size=filter_size, \
                    padding=padding, \
                    strides=strides, \
                    activation=activations_dict[activation],
                data_format='channels_last',
                kernel_initializer=weight_initializer,
                bias_initializer=bias_initializer,)] +
                 (intermediate_pooling_layer if have_pooling else [])
                for index,(filter_size,padding,strides,have_pooling,activation) in enumerate(zip(filter_sizes,padding,strides,pooling_in_layer,activations))
            ],[])
            + pooling_layer
            + [ keras.layers.Flatten() ]
            + [
                # keras.layers.Dense(1,activation=tf.nn.sigmoid,)
                keras.layers.Dense(1,#activation=tf.nn.sigmoid,)
                kernel_initializer=weight_initializer,
                bias_initializer=bias_initializer_last_layer,)
                ])
                # ] + [keras.layers.Lambda(lambda x:x+shifted_init_shift)])

    elif network == "fc":
            model = keras.Sequential(
                ([
                keras.layers.Dense(layer_width, activation=activation, input_shape=(input_dim,) if index==0 else (None,),#)
                    kernel_initializer=weight_initializer,
                    bias_initializer=bias_initializer)
                    for index,(layer_width,activation) in enumerate(zip(layer_widths,activations))#range(number_layers)
                ] if number_layers > 0 else [])
                # + [keras.layers.Lambda(lambda x: x-1/np.sqrt(2*np.pi))]
                + [
                    keras.layers.Dense(1,input_shape=(input_dim,) if number_layers==0 else (None,),#activation=tf.nn.sigmoid,
                    kernel_initializer=weight_initializer,
                    bias_initializer=bias_initializer_last_layer,)
                ])
                # ])+[keras.layers.Lambda(lambda x:x+shifted_init_shift)])

    elif network == "resnet":
        #from keras_contrib.applications.resnet import ResNet
        #import sys
        #sys.path += keras_contrib.__path__ + [keras_contrib.__path__[0]+"/applications/"]
        from .resnet import ResNet
        import keras

        n_blocks = 3
        DEPTH = number_layers
        if DEPTH % 6 != 2:
            raise ValueError('DEPTH must be 6n + 2:', DEPTH)
        # block_depth = (DEPTH - 2) // 6
        # resnet_n_plain = resnet_n % 100
        block_depth = (DEPTH - 2) // (n_blocks * 2)
        model = ResNet(input_shape=(image_height,image_width,number_channels), classes=1,
            block='basic',
            repetitions=[block_depth]*n_blocks,
            transition_strides=[(1,1), (2,2), (2,2), (2,2), (2,2), (2,2), (2,2)][:n_blocks],
            initial_filters=64,
            initial_strides=(1,1),
            initial_kernel_size=(3,3),
            initial_pooling=None,
            #initial_pooling='max',
            #final_pooling=None,
            final_pooling=pooling if pooling is not None else "none",
            activation=None)
            # activation='sigmoid')
    else:
        model = keras.models.Sequential()
        if network == "vgg19":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras.applications.vgg19.VGG19(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "vgg16":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras.applications.vgg16.VGG16(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "resnet50":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            # model1 = keras.applications.resnet.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)
            model1 = keras_applications.resnet.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "resnet101":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras_applications.resnet.ResNet101(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "resnet152":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras_applications.resnet.ResNet152(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "resnetv2_50":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras_applications.resnet_v2.ResNet50V2(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "resnetv2_101":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras_applications.resnet_v2.ResNet101V2(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "resnetv2_152":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras_applications.resnet_v2.ResNet152V2(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "inception_resnet_v2":
            image_height, image_width, number_channels = max(image_height,75), max(image_width,75), max(number_channels,3)
            model1 = keras_applications.inception_resnet_v2.InceptionResNetV2(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "inception_v3":
            image_height, image_width, number_channels = max(image_height,75), max(image_width,75), max(number_channels,3)
            model1 = keras.applications.inception_v3.InceptionV3(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "resnext50":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras_applications.resnext.ResNeXt50(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "resnext101":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras_applications.resnext.ResNeXt101(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "densenet121":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras_applications.densenet.DenseNet121(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "densenet169":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras_applications.densenet.DenseNet169(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "densenet201":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras_applications.densenet.DenseNet201(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        elif network == "mobilenetv2":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(image_height,image_width,number_channels), alpha=1.0, include_top=False, weights=None, input_tensor=None, pooling=pooling, classes=num_classes)

        elif network == "nasnet":
            image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
            model1 = keras.applications.nasnet.NASNetLarge(input_shape=(image_height,image_width,number_channels), include_top=False, weights=None, input_tensor=None, pooling=pooling, classes=num_classes)

        elif network == "xception":
            image_height, image_width, number_channels = max(image_height,71), max(image_width,71), max(number_channels,3)
            model1 = keras.applications.xception.Xception(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=num_classes)

        model.add(model1)
        print(model1.output_shape)
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(1,
                    kernel_initializer=weight_initializer,
                    bias_initializer=bias_initializer_last_layer,))

    print("Number of parameters: ",model.count_params())
    print(model.summary())

    json_string = model.to_json()

    '''SAVE ARCHITECTURE'''
    save_arch(json_string,FLAGS)

if __name__ == '__main__':

    f = tf.compat.v1.app.flags

    from utils import define_default_flags

    define_default_flags(f)

    tf.compat.v1.app.run()
