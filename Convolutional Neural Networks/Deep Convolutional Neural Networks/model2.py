# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 22:06:07 2018

@author: wmy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
import pydot
from kt_utils import *

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255.0
X_test = X_test_orig/255.0

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

def creatModel(input_shape):
    # layer 1
    X_input = keras.layers.Input(shape = input_shape)
    X = keras.layers.ZeroPadding2D(padding=(1, 1))(X_input)
    X = keras.layers.Conv2D(8, kernel_size=(3,3), strides=(1,1))(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)
    # layer 2
    X = keras.layers.ZeroPadding2D(padding=(1, 1))(X)
    X = keras.layers.Conv2D(16, kernel_size=(3,3), strides=(1,1))(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)
    # layer 3
    X = keras.layers.ZeroPadding2D(padding=(1, 1))(X)
    X = keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1))(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)   
    # layer 5
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(20, activation='relu')(X)
    # layer 6
    X = keras.layers.Dense(7, activation='relu')(X)
    # layer 7
    Y = keras.layers.Dense(1, activation='sigmoid')(X)
    model = keras.models.Model(inputs = X_input, outputs = Y, name='HappyModel')
    return model

happyModel = creatModel((64, 64, 3))

happyModel.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), loss='binary_crossentropy', metrics=['accuracy'])

happyModel.fit(x=X_train, y=Y_train, batch_size=16, epochs=20)

preds = happyModel.evaluate(x=X_test, y=Y_test)

print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

img_path = 'images/my_image.jpg'

from keras.preprocessing import image
img = image.load_img(img_path, target_size=(64, 64))
plt.imshow(img)
plt.show()

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

from keras.applications.imagenet_utils import preprocess_input
x = preprocess_input(x)

print(happyModel.predict(x))

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
K.set_image_data_format('channels_last')
happyModel.summary()
plot_model(happyModel, to_file='HappyModel2.png')
SVG(model_to_dot(happyModel).create(prog='dot', format='svg'))
    