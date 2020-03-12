""" This is an implementation of LeNet5 for handwritten digit recogintion"""


# Numpy
import numpy as np

# Tensorflow and keras layers
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend

# One hot encoding 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# ONNX for Keras
import onnx
import onnxruntime
import keras2onnx

# NGRAPH ONNX
from ngraph_onnx.onnx_importer.importer import import_onnx_model
import ngraph as ng

# Other utilities
from lib.common_utils import gen_csv_from_tuples, read_csv_list, pickle_object, unpickle_object, get_ram, get_elapsed_time
import time, os

MODEL_DIR = "models/"
ONNX_DIR = MODEL_DIR + "onnx/"
H5_DIR = MODEL_DIR + "h5/"
PKL_DIR = MODEL_DIR + "pkl/"

#Dataset
from tensorflow.keras.datasets import mnist
class MyValidator(object):

	def __init__(self, model = None):
		# This list will store the different models recently trained
		self.generated_models= []
		# This list will store the files where the ONNX models are stored
		self.generated_models_onnx = []
		# This list will store the files where the H5 models are stored
		self.generated_models_h5 = []
		# This list stores the metrics of evaluation of the different models
		self.evaluations = []

		self.list_models()

	def list_models(self):
		self.generated_models_onnx = [ONNX_DIR + name for name in os.listdir(ONNX_DIR)]
		self.generated_models_h5 = [H5_DIR + name for name in  os.listdir(H5_DIR)]


	def __get_one_hot_encoding(self, y):
		# Obtains the one hot encoding for the y.
		return OneHotEncoder(sparse= False).fit_transform(y)
	
	def __get_label_encoding(self, y):
		# Not used but encodes the one hot encoding in a label
		return  LabelEncoder().fit_transform(y)

	def __get_kth_fold(self, x, y, tt_begin, tt_end):
		# Returns the division in k folds of x, y given start and end position.
		x_train = np.concatenate((x[:tt_begin], x[tt_end:]), axis=0)
		y_train = np.concatenate((y[:tt_begin], y[tt_end:]), axis=0)
		x_test = x[tt_begin:tt_end]
		y_test = y[tt_begin:tt_end]
		return (x_train, y_train), (x_test, y_test)

	def get_data(self):
		# Generates the dataset in x and y coordinates without any division.
		(x_train, y_train), (x_test, y_test) = mnist.load_data()
		x = np.concatenate((x_train, x_test))
		y = np.concatenate((y_train, y_test))
		x = x / 255
		y = self.__get_one_hot_encoding(y.reshape(-1,1))
		x = x.reshape((-1, x.shape[1], x.shape[2], 1)) # We add a final dimension to the input for the padding.
		return x, y

	def get_model(self, input_shape):
		# Generates the model for the given architecture.
		# We get two consecutive Conv + Max Pooling layers and afterwards we put Dense layers
		# Input size is going to be 32x32x1 (grayscale)
		X_input = Input(input_shape)
		X = ZeroPadding2D((4,4))(X_input)
		
		X = Conv2D(6, (5,5), strides=(1,1), name='conv0')(X)
		X = MaxPooling2D((2,2), name='max_pool0')(X)
		X = Conv2D(16, (5,5), strides=(1,1), name='conv1')(X)
		X = MaxPooling2D((2,2), name='max_pool1')(X)
		X = Flatten()(X)
		X = Dense(120, activation='relu', input_shape=(400,))(X)
		X = Dense(10, activation='softmax')(X)
		model = Model(inputs=X_input, outputs=X, name='mnist_model')
		return model

	def train(self, model, x_train, y_train):
		# Compiles and trains the model, uses binary crossentropy and ADAM optimizer. 
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['accuracy', 'AUC', 'TruePositives', 'FalsePositives' ,'TrueNegatives', 'FalseNegatives'])
		model.fit(x= x_train, y=y_train, epochs=1, batch_size=64)
		return model

	def predict(self, model, x, y):
		# If we need to predict something for a model.
		return model.predict(x)

	def train_1_model(self, x, y):
		(x_train, y_train), (x_test, y_test) = self.__get_kth_fold(x, y, 0, x.shape[0] * 0.1)
		model = self.get_model(x.shape[1:])
		trained_model = self.train(model, x_train, y_train)
		evaluation = trained_model.evaluate(x_test, y_test)
		self.save_model(model, evaluation)

	def k_fold_validation(self, x, y, k=10):
		# Executes the k fold cross validation for the model and stores the partial models.
		m = x.shape[0]
		fold_len = m // k# Might lose certain samples
		x, y = self.get_data()
		evaluations = []
		for i in range(k):
			# Calculate the fold that we use as test.
			tt_begin = i * fold_len
			tt_end = tt_begin + fold_len
			# Get the fold that we make use of.
			(x_train, y_train), (x_test, y_test) = self.__get_kth_fold(x, y, tt_begin, tt_end)

			# Generate the model
			model = self.get_model(x.shape[1:])
			# Train the model
			trained_model = self.train(model, x_train, y_train)
			# Evaluate the model
			evaluation = trained_model.evaluate(x_test, y_test)

			self.save_model(trained_model, evaluation)

			print("FINISHED ITERATION [%d]" %(i), evaluation)
		return evaluations

	def save_model(self, trained_model, evaluation):
		# Append evaluation to internal list
		self.evaluations.append(evaluation)
		
		# Elaborating  the filenames for the different archives and storing them.
		filename_base = 'mnist_model_%0.2f_%0.2f_%0.2f_%0.2f_%0.2f_%0.2f' % (evaluation[1], evaluation[2], evaluation[3], evaluation[4], evaluation[5], evaluation[6])
		filename_onnx = ONNX_DIR + filename_base + ".onnx"
		filename_h5 = H5_DIR + filename_base + ".h5"


		self.generated_models_h5.append(filename_h5)
		self.generated_models_onnx.append(filename_onnx)
	
		# Save the different models
		trained_model.save(filename_h5)
		self.export_to_onnx(trained_model, filename_onnx)




	def export_to_onnx(self, model, file):
		# Exports the model in Keras to ONNX format
		onnx_model = keras2onnx.convert_keras(model, model.name)
		print("Trying to transform to function")
		ng_model = mv.onnx_to_ngraph(onnx_model)
		onnx.save_model(onnx_model, file)

	# TODO: FINISH
	def evaluate_onnx(self, file):
		if opt == 1:
			onnx_protobuf = onnx.load(file)
			content = onnx_model.SerializeToString()
			sess = onnxruntime.InferenceSession(content)
			feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
			pred_onnx = sess.run(None, feed)

		else:
			#onnx_model = keras2onnx.convert_keras(model, model.name)
			keras2onnx.save_model(onnx_model, temp_model_file)
			sess = onnxruntime.InferenceSession(temp_model_file)
		
	def __transform(self, onnx_model):
		# Use some symbolic name not used for any other dimension
		#sym_batch_dim = "N"
		# or an actal value
		#actual_batch_dim = 4 
		onnx_model.graph.node[5].attribute[0].ints[0] = 1
		sym_batch_dim = "N"
		# The following code changes the first dimension of every input to be batch-dim
		# Modify as appropriate ... note that this requires all inputs to
		# have the same batch_dim 
		inputs = onnx_model.graph.input
		for x in onnx_model.graph.node:
			if len(x.attribute) > 0 and len(x.attribute[0].ints) > 0:
				print(x.attribute[0].ints)
				x.attribute[0].ints[0] = 1
		for inp in inputs:
			# Checks omitted.This assumes that all inputs are tensors and have a shape with first dim.
			# Add checks as needed.
			dim1 = inp.type.tensor_type.shape.dim[0]
			# update dim to be a symbolic value
			dim1.dim_param = sym_batch_dim
			# or update it to be an actual value:
			# dim1.dim_value = actual_batch_dim

	def onnx_to_ngraph(self, file):
		# Reads a ONNX graph and creates the Intel nGraph function for it to be used.
		onnx_protobuf = onnx.load(file)
		#self.__transform(onnx_protobuf)
		#print(onnx_protobuf)
		ngraph_function = import_onnx_model(onnx_protobuf)
		runtime = ng.runtime(backend_name='CPU')
		model = runtime.computation(ngraph_function)
		return model

def generate_models(mv):
	# Test function used to generate the data for the different cross validation sets.
	tic = time.time()
	x, y = mv.get_data()
	mv.k_fold_validation(x, y, k=10)
	return x, y

def test_ng_onnx(mv, x, y):
	print('[>>] Testing NG ONNX...')
	times = []
	t0 = time.time()
	times.append(("T0", t0 - t0)) # t0
	# Tests different parameters on the different created models.
	for model in mv.generated_models_onnx:
		# We create the ng_model
		model = 'models/alexnet.onnx'
		print('[>>] %s' % (model))
		times.append(("T1",time.time() - t0)) # t1
		ng_model = mv.onnx_to_ngraph(model)
		times.append(("T2",time.time() - t0)) # t2
		_y = ng_model(x)
		times.append(("T3",time.time() - t0)) # t3

	times.append(("T4",time.time() - t0)) # t4
	print(times)
	return times

# TODO: FINISH
def test_onnx(mv, x, y):
	times = []
	t0 = time.time()
	times.append(time.time()) # t0
	# Tests different parameters on the different created models.
	for model in mv.generated_models_onnx:
		# We create the ng_model
		times.append(time.time()) # t1
		ng_model = mv.export_to_onnx(ng_model)
		times.append(time.time()) # t2
		_y = ng_model(x)
		times.append(time.time()) # t3

	times.append(time.time()) # t4
	return times

# TODO: FINISH
def test_h5(mv, x, y):
	times = []
	times.append(("T0",time.time())) # t0
	# Tests different parameters on the different created models.
	for model in mv.generated_models_onnx:
		# We create the ng_model
		times.append(("T1",time.time())) # t1
		ng_model = mv.export_to_onnx(ng_model)
		times.append(("T2",time.time())) # t2
		for i in x:
			print(i.shape)
			_y = ng_model(i)
		times.append(("T3",time.time())) # t3

	times.append(("T4",time.time())) # t4
	return times

def test(mv, x, y):
	times_ng_onnx = test_ng_onnx(mv, x, y)
	#times_h5 = test_h5(mv, x, y)
	#times_k = test_keras(mv, x, y)
	#times_onnx = 
	gen_csv_from_tuples('times_ng_onnx.csv', ['time'], times_ng_onnx )
	#times3 = mv.evaluations

def main():
	mv = MyValidator()
	x, y = mv.get_data()
	import torch
	import torchvision

	x = torch.randn(10, 3, 224, 224)
	#x, y = generate_models(mv)
	test(mv, x, y)
	
if __name__ == "__main__":
	main()
