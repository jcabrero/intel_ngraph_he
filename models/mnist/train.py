import sys
sys.path.append('/root/mnt_dir/lib/') # Adding some necessary libraries from the repository.
# Other utilities
from common_utils import gen_csv_from_tuples, read_csv_list, pickle_object, unpickle_object, get_ram, get_elapsed_time
import time, os, argparse

# Numpy
import numpy as np

# Tensorflow and keras layers
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.tools import freeze_graph
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend

# Import the model
from model import cryptonets_model


def get_data(start_batch=0, batch_size=10000):
	"""Returns MNIST data in one-hot form"""
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	y_train = tf.compat.v1.keras.utils.to_categorical(y_train, num_classes=10)
	y_test = tf.compat.v1.keras.utils.to_categorical(y_test, num_classes=10)
	x_train = np.expand_dims(x_train, axis=-1)
	x_test = np.expand_dims(x_test, axis=-1)

	x_train = x_train.astype("float32")
	x_test = x_test.astype("float32")
	x_train /= 255.0
	x_test /= 255.0

	x_test = x_test[start_batch:start_batch + batch_size]
	y_test = y_test[start_batch:start_batch + batch_size]


	return x_train, y_train, x_test, y_test


def train(args):
	x_train, y_train, x_test, y_test = get_data()

	x = Input(shape=(28, 28, 1,), name="input")
	y = cryptonets_model(x)
	model = Model(inputs=x, outputs=y)
	print(model.summary())

	def loss(labels, logits):
		return keras.losses.categorical_crossentropy(labels, logits, from_logits=True)

	optimizer = SGD(learning_rate=0.008, momentum=0.9)
	model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
	
	model.fit(x_train, y_train, epochs=args.epochs,
		batch_size=args.batch_size,
		validation_data=(x_test, y_test),
		verbose=1)



def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--epochs", type=int, default=10, help="Number of training epochs")
	parser.add_argument(
		"--batch_size", type=int, default=128, help="Batch Size")
	args, unparsed = parser.parse_known_args()
	if unparsed:
		print("Unparsed flags: ", unparsed)
		exit(1)
	return args
def main():
	args = parse_args()
	train(args)


if __name__ == "__main__":
	main()