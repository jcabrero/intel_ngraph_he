import sys
sys.path.append('/root/mnt_dir/lib/') # Adding some necessary libraries from the repository.
# Other utilities
from common_utils import gen_csv_from_tuples, read_csv_list, pickle_object, unpickle_object, get_ram, get_elapsed_time
import time, os

# Numpy
import numpy as np

# Tensorflow and keras layers
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend


# Source code from IntelAI repository
def cryptonets_model(input):

    def square_activation(x):
        return x * x

    y = Conv2D(filters=5, kernel_size=(5, 5), strides=(2, 2), padding="same", use_bias=True, input_shape=(28, 28, 1), name="conv2d_1", )(input)
    y = Activation(square_activation)(y)
    y = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y)
    y = Conv2D(filters=50, kernel_size=(5, 5), strides=(2, 2), padding="same", use_bias=True, name="conv2d_2", )(y)
    y = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(y)
    y = Flatten()(y)
    y = Dense(100, use_bias=True, name="fc_1")(y)
    y = Activation(square_activation)(y)
    y = Dense(10, use_bias=True, name="fc_2")(y)

    return y


if __name__ == "__main__":
	x = Input(shape=(28, 28, 1,), name="input")
	y = cryptonets_model(x)
	model = Model(inputs=x, outputs=y)
	print(model.summary())