import torchvision
from torchvision import transforms
import torch
from math import ceil
import tensorflow as tf
import numpy as np
import pickle
import tensorflow.keras as keras
image_size=28
image_height = image_width = image_size
number_channels=1
model = keras.models.Sequential()
image_height, image_width, number_channels = max(image_height,32), max(image_width,32), max(number_channels,3)
pooling=None
model1 = keras.applications.vgg19.VGG19(include_top=False, weights=None, input_tensor=None, input_shape=(image_height,image_width,number_channels), pooling=pooling, classes=2)
model.add(model1)
model.add(keras.layers.Dense(1))#,

#%%
# d = torchvision.datasets.KMNIST("./datasets",download=True,
#         transform=transforms.Compose(
#             [transforms.ToPILImage()]+
#             # []+
#             ([transforms.Resize((32,32))] if image_size is not None else [])+
#             [transforms.ToTensor()]
#         ),
#     )
# mm = ceil(d.data.shape[0]*5/6)
# (train_images,train_labels),(test_images,test_labels) = (d.data[:mm], d.targets[:mm]),(d.data[mm:],d.targets[mm:])
# train_images.shape
# train_images = np.expand_dims(train_images,-1)
# train_images = np.tile(train_images,(1,1,1,3))
# assert train_images.dtype == "uint8" #otherwise ToPILImage wants the input to be NCHW. wtff
# train_images = np.stack([d.transform(image) for image in train_images])
# train_images = np.transpose(train_images,(0,2,3,1)) # this is because the pytorch transform changes it to NCHW for some reason :P
#
# # train_images*=255.0
# ys = [y.item()>=5 for y in train_labels]
# train_images = train_images[:1000]
# ys = ys[:1000]
#
# pickle.dump(train_images, open("train_images_tmp.p","wb"))
# pickle.dump(ys, open("ys_tmp.p","wb"))
# import matplotlib.pyplot as plt
#
# plt.imshow(train_images[0])
train_images = pickle.load(open("train_images_tmp.p","rb"))
ys = pickle.load(open("ys_tmp.p","rb"))

#%%
def binary_crossentropy_from_logits(y_true,y_pred):
    return tf.keras.backend.binary_crossentropy(y_true, y_pred,from_logits=True)
model.compile(optimizer='sgd',
              loss=binary_crossentropy_from_logits,
              metrics=['accuracy'])
# model.fit(train_images, ys, verbose=2, validation_data=(train_images, ys), epochs=3000)
model.fit(train_images, ys, verbose=2, epochs=3000)
