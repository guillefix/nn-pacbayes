import tensorflow as tf
import tensorflow.keras as keras
image_size=28
image_height = image_width = image_size
number_channels=1
model = keras.models.Sequential()
image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
pooling=None
model1 = keras.applications.vgg19.VGG19(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)
model.add(model1)
# sigmab = 0.0
# sigmaw = 1.0
# bias_initializer = keras.initializers.RandomNormal(stddev=sigmab)
# # weight_initializer = keras.initializers.RandomNormal(stddev=sigmaw/np.sqrt(input_dim))
# weight_initializer = keras.initializers.VarianceScaling(scale=sigmaw**2, mode='fan_in', distribution='normal', seed=None)
# bias_initializer_last_layer = bias_initializer
model.add(keras.layers.Dense(1))#,
            # kernel_initializer=weight_initializer,
            # bias_initializer=bias_initializer_last_layer,))

#%%

###CNN

number_layers = 4
num_filters = 200
filter_sizes = [[5,5],[2,2]]*10
padding = ["VALID", "SAME"]*10
padding = padding[:number_layers]
pooling_in_layer = [x=="1" for x in "0000"]
strides = [[1, 1]] * number_layers
intermediate_pooling_layer = []
pooling_layer = []

model = keras.Sequential( sum([
        [keras.layers.Conv2D(input_shape=(image_height,image_width,number_channels), filters=num_filters, kernel_size=filter_size, padding=padding, strides=strides, activation=tf.nn.relu,
        data_format='channels_last',
        kernel_initializer=weight_initializer,
        bias_initializer=bias_initializer,)] +
         (intermediate_pooling_layer if have_pooling else [])
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
        keras.layers.Dense(1,#activation=tf.nn.sigmoid,)
        kernel_initializer=weight_initializer,
        bias_initializer=bias_initializer_last_layer,)
        # kernel_regularizer=keras.regularizers.l2(0.01*input_dim/(2*sigmaw**2)),
        # bias_regularizer=keras.regularizers.l2(1/(2*sigmab**2)))
        # kernel_regularizer=keras.regularizers.l2(0.05),
        # bias_regularizer=keras.regularizers.l2(0.1))
    ])


####TRAIN

#%%
# import h5py
# filename="data/resnet50_KMNIST_1000_0.0_0.0_True_False_False_False_-1_True_False_False_data.h5"
# filename = "data/cnn_KMNIST_1000_0.0_0.0_True_False_False_False_-1_True_False_False_data.h5"
# h5f = h5py.File(filename,'r')
# train_images = h5f['train_images'][:]
ys = h5f['ys'][:]
import matplotlib.pyplot as plt
import numpy as np
train_images[0].shape
plt.imshow(train_images[0])
train_images[0].max()

train_images*=255.0

import torchvision
from torchvision import transforms
import torch
d = torchvision.datasets.KMNIST("./datasets",download=True,
        transform=transforms.Compose(
            [transforms.ToPILImage()]+
            # []+
            ([transforms.Resize((32,32))] if image_size is not None else [])+
            [transforms.ToTensor()]
        ),
    )
from math import ceil
mm = ceil(d.data.shape[0]*5/6)
(train_images,train_labels),(test_images,test_labels) = (d.data[:mm], d.targets[:mm]),(d.data[mm:],d.targets[mm:])
num_classes = 10
train_images.shape
train_images = np.expand_dims(train_images,-1)
train_images = np.tile(train_images,(1,1,1,3))
# train_images = np.transpose(train_images,(0,3,1,2)) # this is because the pytorch transform needs it in NCHW
# train_images = torch.Tensor(train_images)
# train_images.dtype
# train_images = train_images.astype(np.uint8)
train_images = np.stack([d.transform(image) for image in train_images])
train_images = np.transpose(train_images,(0,2,3,1)) # this is because the pytorch transform changes it to NCHW for some reason :P
train_labels
ys = [y.item()>=5 for y in train_labels]
ys
train_images = train_images[:1000]
ys = ys[:1000]
# train_images.shape
#
# plt.imshow(train_images[0])
#
# # transforms.ToPILImage()(train_images[0])
# # from PIL import Image
# train_images[0].shape
# transforms.ToTensor()(transforms.ToPILImage()(train_images[0])).shape
# np.tile(transforms.ToTensor()(transforms.ToPILImage()(train_images[0])),(1,1,1,3)).shape
# plt.imshow(np.tile(transforms.ToTensor()(transforms.ToPILImage()(train_images[0])),(1,1,1,3)))
# np.transpose(train_images[0], (3,1,2))
# np.transpose(transforms.ToTensor()(transforms.ToPILImage()(np.transpose(train_images[0], (2,0,1)))), (1,2,0)).shape
# plt.imshow(np.transpose(transforms.ToTensor()(transforms.ToPILImage()(np.transpose(torch.Tensor(np.tile(train_images[0],(1,1,3))), (2,0,1)))), (1,2,0)))
# plt.imshow(np.transpose(transforms.ToTensor()(transforms.ToPILImage()(np.transpose(torch.Tensor(train_images[0]), (2,0,1)))), (1,2,0)))
# plt.imshow(transforms.ToTensor()(transforms.ToPILImage()(train_images[0])))
#

#%%
from utils import binary_crossentropy_from_logits,EarlyStoppingByAccuracy, get_biases, get_weights, measure_sigmas, get_rescaled_weights
model.compile(optimizer='sgd',#keras.optimizers.SGD(lr=0.01,momentum=0.9,decay=1e-6),#'sgd',#tf.keras.optimizers.SGD(lr=0.01),
              #loss='binary_crossentropy',
              loss=binary_crossentropy_from_logits,
              # loss_weights=[50000],
              metrics=['accuracy'])


callbacks = [
        EarlyStoppingByAccuracy(monitor='val_acc', value=1.0, verbose=1, wait_epochs=32),
        #missinglink_callback,
        # EarlyStopping(monitor='val_loss', patience=2, verbose=0),
        # ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]
MAX_TRAIN_EPOCHS = 3000
sample_weights = None
#%%
model.fit(train_images, ys, verbose=1,validation_data=(train_images, ys), epochs=MAX_TRAIN_EPOCHS,callbacks=callbacks)
