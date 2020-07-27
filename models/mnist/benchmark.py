import time, os

# Numpy
import numpy as np
#import scipy
#import sklearn
#import pandas as pd

# Tensorflow and keras layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense,\
                                    Activation, ZeroPadding2D,\
                                    BatchNormalization, Flatten, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D,\
                                    Dropout, GlobalMaxPooling2D,\
                                    GlobalAveragePooling2D

from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import activations
from tensorflow.keras import backend

# To generate GIFs



import glob
#import imageio
#import matplotlib.pyplot as plt
import numpy as np
import os
#import PIL
from tensorflow.keras import layers
import time
from datetime import datetime

#from tqdm.auto import tqdm

#import IPython
#from IPython import display
#import ipywidgets as widgets

#import ngraph_bridge

# For loading and making use of HE Transformer
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.tools import freeze_graph
import json

def create_mnist_npy():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X = np.concatenate([x_train, x_test]).reshape(-1, 28*28)
    Y = np.concatenate([y_train, y_test]).reshape(-1, 1)
    #print(X.shape, Y.shape)
    T = np.concatenate([X, Y], axis=1)
    np.save("mnist.npy", T)
    
def load_mnist():
    T = np.load('mnist.npy')
    X = T[:, :-1].reshape(-1, 28, 28)
    Y = T[:, -1]
    x_train, x_test = X[:-10000], X[-10000:]
    y_train, y_test = Y[:-10000], Y[-10000:]
    y_train = tf.compat.v1.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.compat.v1.keras.utils.to_categorical(y_test, num_classes=10)
    #print(y_train.shape, y_test.shape, y_train)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255.0
    x_test /= 255.0
    return (x_train, y_train, x_test, y_test)

if not os.path.exists('mnist.npy'):
    create_mnist_npy()
tic = time.time()
(x_train, y_train, x_test, y_test) = load_mnist()

base_model = keras.Sequential(
    [
        layers.Conv2D(filters=5, kernel_size=(5, 5), strides=(2, 2), padding="same", use_bias=True, input_shape=(28, 28, 1), name="conv2d_1", activation='relu'),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same", name="avgpooling_1"),
        layers.Conv2D(filters=50, kernel_size=(5, 5), strides=(2, 2), padding="same", use_bias=True, name="conv2d_2", activation='relu'),
        layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same", name="avgpooling_2"),
        layers.Flatten(),
        layers.Dense(100, use_bias=True, name="fc_1", activation='relu'),
        layers.Dense(10, use_bias=True, name='fc_2', activation='sigmoid')
    ]
)

base_model.compile(
    optimizer=SGD(learning_rate=0.008, momentum=0.9), 
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"])

base_model.fit(x_train, y_train, 
                     epochs=10, batch_size=64, 
                     validation_data=(x_test, y_test), 
                     verbose=1)

base_model.save('base_model.h6')
toc = time.time()

print("\n\n\nTOTAL TIME: %d\n\n\n" % (toc - tic))
