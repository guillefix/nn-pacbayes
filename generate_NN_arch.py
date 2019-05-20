import numpy as np
import tensorflow as tf

# import tqdm
#import missinglink
#missinglink_callback = missinglink.KerasCallback()

arch_folder = "archs/"

def main(_):

    FLAGS = tf.app.flags.FLAGS.flag_values_dict()
    from utils import preprocess_flags
    FLAGS = preprocess_flags(FLAGS)
    globals().update(FLAGS)

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]=str((rank+1)%n_gpus)

    from tensorflow import keras
    # import keras
    import keras_applications
    keras_applications._KERAS_BACKEND = keras.backend
    keras_applications._KERAS_LAYERS = keras.layers
    keras_applications._KERAS_MODELS = keras.models
    keras_applications._KERAS_UTILS = keras.utils
    import warnings
    bias_initializer = keras.initializers.RandomNormal(stddev=1.0)
    # weight_initializer = keras.initializers.RandomNormal(stddev=1/np.sqrt(input_dim))
    # keras.layers.Flatten(input_shape=(28, 28)),

    # %%

    if dataset == "cifar":
            image_size=32
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
    else:
            raise NotImplementedError

    image_height = image_size
    image_width = image_size
    input_dim = image_height*image_width*number_channels
    set_session = keras.backend.set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = False  # to log device placement (on which device the operation ran)
    # config.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    if network == "cnn":
        if pooling=="avg":
            pooling_layer = [keras.layers.GlobalAveragePooling2D()]
        elif pooling=="max":
            pooling_layer = [keras.layers.GlobalMaxPooling2D()]
        else:
            pooling_layer = []
        model = keras.Sequential(
            sum([
                [keras.layers.Conv2D(input_shape=(image_height,image_width,number_channels), filters=num_filters, kernel_size=filter_size, padding=padding, strides=strides, activation=tf.nn.relu,
                # kernel_initializer=weight_initializer,
                bias_initializer=bias_initializer,)] +
                 ([keras.layers.MaxPool2D(pool_size=2, padding='same')] if have_pooling else [])
                # kernel_regularizer=keras.regularizers.l2(0.01*input_dim/(2*sigmaw**2)),
                # bias_regularizer=keras.regularizers.l2(1/(2*sigmab**2)))
                # kernel_regularizer=keras.regularizers.l2(0.05),
                # bias_regularizer=keras.regularizers.l2(0.1))
                for filter_size,padding,strides,have_pooling in zip(filter_sizes,padding,strides,pooling_in_layer)
            ],[])
            + pooling_layer
            + [ keras.layers.Flatten() ]
            + [
                # keras.layers.Dense(1,activation=tf.nn.sigmoid,)
                keras.layers.Dense(1),#activation=tf.nn.sigmoid,)
                # kernel_initializer=weight_initializer,
                # bias_initializer=bias_initializer,)
                # kernel_regularizer=keras.regularizers.l2(0.01*input_dim/(2*sigmaw**2)),
                # bias_regularizer=keras.regularizers.l2(1/(2*sigmab**2)))
                # kernel_regularizer=keras.regularizers.l2(0.05),
                # bias_regularizer=keras.regularizers.l2(0.1))
            ])

    elif network == "fc":
            model = keras.Sequential(
                [ keras.layers.Dense(input_dim, activation=tf.nn.relu,input_shape=(input_dim,))
                    # kernel_initializer=weight_initializer,
                    # bias_initializer=bias_initializer,)
            ]
            + [
                keras.layers.Dense(input_dim, activation=tf.nn.relu,)
                    # kernel_initializer=weight_initializer,
                    # bias_initializer=bias_initializer)
                    # kernel_regularizer=keras.regularizers.l2(0.01*input_dim/(2*sigmaw**2)),
                    # bias_regularizer=keras.regularizers.l2(1/(2*sigmab**2)))
                    # kernel_regularizer=keras.regularizers.l2(0.05),
                    # bias_regularizer=keras.regularizers.l2(0.1))
                    for i in range(number_layers-1)
                ]
                + [
                    keras.layers.Dense(1)#activation=tf.nn.sigmoid,)
                    # keras.layers.Dense(1,activation=tf.nn.sigmoid,)#activation=tf.nn.sigmoid,)
                    # kernel_initializer=weight_initializer,
                    # bias_initializer=bias_initializer,)
                    # kernel_regularizer=keras.regularizers.l2(0.01*input_dim/(2*sigmaw**2)),
                    # bias_regularizer=keras.regularizers.l2(1/(2*sigmab**2)))
                    # kernel_regularizer=keras.regularizers.l2(0.05),
                    # bias_regularizer=keras.regularizers.l2(0.1))
                ])
    elif network == "resnet":
        from keras_contrib.applications.resnet import ResNet
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
            final_pooling=pooling,
            activation=None)
            # activation='sigmoid')
    else:
        model = keras.models.Sequential()
        if network == "vgg19":
            model1 = keras.applications.vgg19.VGG19(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "vgg16":
            model1 = keras.applications.vgg16.VGG16(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "resnet50":
            model1 = keras.applications.ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "resnet101":
            model1 = keras_applications.resnet.ResNet101(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "resnet152":
            model1 = keras_applications.resnet.ResNet152(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "resnetv2_50":
            model1 = keras_applications.resnet_v2.ResNet50V2(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "resnetv2_101":
            model1 = keras_applications.resnet_v2.ResNet101V2(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "resnetv2_152":
            model1 = keras_applications.resnet_v2.ResNet152V2(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "resnext50":
            model1 = keras_applications.resnext.ResNeXt50(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "resnext101":
            model1 = keras_applications.resnext.ResNeXt101(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "densenet121":
            model1 = keras_applications.densenet.DenseNet121(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "densenet169":
            model1 = keras_applications.densenet.DenseNet169(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "densenet201":
            model1 = keras_applications.densenet.DenseNet201(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        elif network == "mobilenetv2":
            model1 = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(image_height,image_width,number_channels), alpha=1.0, include_top=False, weights=None, input_tensor=None, pooling=pooling, classes=2)

        elif network == "nasnet":
            model1 = keras.applications.nasnet.NASNetLarge(input_shape=(image_height,image_width,number_channels), include_top=False, weights=None, input_tensor=None, pooling=pooling, classes=2)

        elif network == "xception":
            model1 = keras.applications.xception.Xception(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)

        model.add(model1)
        model.add(keras.layers.Dense(1))

    json_string = model.to_json()

    '''SAVE ARCHITECTURE'''
    filename=arch_folder
    for flag in ["network","binarized","number_layers","pooling","intermediate_pooling"]:
        filename+=str(FLAGS[flag])+"_"
    filename += "model"
    with open(filename, "w") as f:
        f.write(json_string)

if __name__ == '__main__':

    f = tf.app.flags

    from utils import define_default_flags

    define_default_flags(f)

    tf.app.run()
