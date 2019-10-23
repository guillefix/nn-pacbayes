import tensorflow as tf
import pickle
import tensorflow.keras as keras
model1 = keras.applications.ResNet50(include_top=False, weights=None, input_shape=(32,32,3))
model = keras.models.Sequential()
model.add(model1)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1))
train_images = pickle.load(open("train_images_tmp.p","rb"))
ys = pickle.load(open("ys_tmp.p","rb"))
import matplotlib.pyplot as plt
import numpy as np
plt.imshow(train_images[np.random.randint(len(train_images))])
plt.show()
import keras_applications
train_images = train_images.astype(np.float32)
# train_images = keras_applications.resnet.preprocess_input(train_images,backend=tf.keras.backend)
model.compile(optimizer='sgd',
              loss=lambda y_true, y_pred: tf.keras.backend.binary_crossentropy(y_true, y_pred,from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, ys, verbose=1, epochs=3000)
