#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time, os

# Numpy
import numpy as np
import scipy
import sklearn
import pandas as pd

# Tensorflow and keras layers
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense,                                    Activation, ZeroPadding2D,                                    BatchNormalization, Flatten, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D,                                    Dropout, GlobalMaxPooling2D,                                    GlobalAveragePooling2D

from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Nadam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import activations
from tensorflow.keras import backend

# To generate GIFs


import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from datetime import datetime

from tqdm import tqdm

import IPython
from IPython import display
import ipywidgets as widgets

import ngraph_bridge
# For loading and making use of HE Transformer
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.tools import freeze_graph
import json



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

def create_dir(dirname):
    try:
        # Create target Directory
        os.mkdir(dirname)
        #print("dirname " , filename ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirname ,  " already exists")

create_dir('models')

def load_pb_file(filename):
    """"Returns the graph_def from a saved protobuf file"""
    if not os.path.isfile(filename):
        raise Exception("File, " + filename + " does not exist")

    with tf.io.gfile.GFile(filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    print("Model restored")
    return graph_def


# ## Generating the Homorphic Encryption Configuration
# We generate the parameters for the encryption. These parameteres usually set information about the Learning with Error (LWE) problem. The explanation of how to obtain the parameters is inline with the code. It is mainly extracted from the documentation of Intel nGraph HE Transformer Github Repository. After having executed this section code, a file 'config.json' has been generated with the configuration for the Neural Network.

# In[12]:


def get_N(total_coefficient_modulus_bit_width, security_level, sec_type='classical'):
    sec = {
        'classical':{
            128:{
                1024:  27,
                2048:  54,
                4096:  109,
                8192:  218,
                16384: 438,
                32768: 881
            },
            192:{
                1024:  19,
                2048:  37,
                4096:  75,
                8192:  152,
                16384: 305,
                32768: 611
            },
            256:{
                1024:  14,
                2048:  29,
                4096:  58,
                8192:  118,
                16384: 237,
                32768: 476
            }
        },
        'quantum':{
           128:{
                1024:  25,
                2048:  51,
                4096:  101,
                8192:  202,
                16384: 411,
                32768: 827
            },
            192:{
                1024:  17,
                2048:  35,
                4096:  70,
                8192:  141,
                16384: 284,
                32768: 571
            },
            256:{
                1024:  13,
                2048:  27,
                4096:  54,
                8192:  109,
                16384: 220,
                32768: 443
            }
        }
    }
    log2q_prev = 0
    for n, log2q in sec[sec_type][security_level].items():
        if log2q_prev < total_coefficient_modulus_bit_width < log2q:
            return n
        log2q_prev = log2q
    return n


def define_encryption_params(security_level):
    # Select the security level
    # Compute the multiplicative depth of the computational graph
    L = 8 # Assumed from configuration
    # Estimate the bit-precission s, required. According to Intel, the best tradeoff is ~24 bits
    s = 24
    # Choose the coeff_modulus = [s, s, s, ..., s]. A list of L coefficient moduli, each with s bits. Set the scale to s.
    coeff_modulus = [s] * L
    coeff_modulus[0] = 30
    coeff_modulus[-1] = 30
    scale = s
    # Compute the total coefficient modulus bit width, L * s in the above parameter selection.
    total_coefficient_modulus_bit_width = L * s
    # Set the poly_modulus_degree to the smallest power of two, with coefficient modulus smaller than the maximum allowed.
    # Based on the table of recommended parameters. 
    N = get_N(total_coefficient_modulus_bit_width, security_level)
    poly_modulus_degree = N
    # For best performance, we should choose the batch_size to max_batch_size
    max_batch_size = poly_modulus_degree / 2
    # We should only include the complex packing if there are polynomial activations.
    # That is, if we make use of our own activation functions
    complex_packing = False
    
    enc_params = {
        'scheme_name': 'HE_SEAL',          # Fixed, to use the HE backend
        'poly_modulus_degree': poly_modulus_degree,       # A power of 2 {1024, 2048, 4096, 8192, 16384}
        'security_level': security_level,             # The security we want to ensure {0, 128, 192, 256}
        'coeff_modulus': coeff_modulus, # A number inbetween 1 and 60
        'scale': 2 ** s,                  # The fixed bit precission of the encoding. (log2(scale) is the number of bits)
        'complex_packing': complex_packing,
    }
    print(enc_params)
    return enc_params
def gen_json_params_file(filename, security_level=128):
    enc_params_dict = define_encryption_params(security_level)
    enc_params_json = json.dumps(enc_params_dict)
    with open(filename, 'w+') as file:
        file.write(enc_params_json)
    print("Generated configuration in %s" %(filename))
    
gen_json_params_file('config.json', security_level=128)



# ## Trying the client-server example

# In[16]:


"""
python test.py --backend=HE_SEAL \
               --model_file=models/cryptonets.pb \
               --enable_client=true \
               --encryption_parameters=$HE_TRANSFORMER/configs/he_seal_ckks_config_N13_L8.json
"""
def get_config_for_ngraph_client(tensor_param_name):
    rewriter_options = rewriter_config_pb2.RewriterConfig()
    rewriter_options.meta_optimizer_iterations = rewriter_config_pb2.RewriterConfig.ONE
    rewriter_options.min_graph_nodes = -1
    server_config = rewriter_options.custom_optimizers.add()
    server_config.name = "ngraph-optimizer"
    server_config.parameter_map["ngraph_backend"].s = b'HE_SEAL'
    server_config.parameter_map["device_id"].s = b""
    server_config.parameter_map["encryption_parameters"].s = b'config.json'
    server_config.parameter_map["enable_client"].s = str(True).encode()

    server_config.parameter_map["enable_gc"].s = b"False"
    server_config.parameter_map["mask_gc_inputs"].s = b"False"
    server_config.parameter_map["mask_gc_outputs"].s = b"False"
    server_config.parameter_map["num_gc_threads"].s = b"1"

    # Only server
    #server_config.parameter_map[tensor_param_name].s = b"encrypt"
    # With client
    server_config.parameter_map[tensor_param_name].s = b"client_input"
    # Pack data
    #server_config.parameter_map[tensor_param_name].s += b",packed"

    config = tf.compat.v1.ConfigProto()
    config.MergeFrom(
            tf.compat.v1.ConfigProto(
                graph_options=tf.compat.v1.GraphOptions(
                    rewrite_options=rewriter_options)))
    return config


# In[17]:


print("STARTING CRITICAL SECTION OF CODE")
# %%px --target [1] --noblock

tf.compat.v1.reset_default_graph()
(x_train, y_train, x_test, y_test) = load_mnist()
tf.import_graph_def(load_pb_file('./models/mlp.pb'))

# Get input / output tensors
x_input = tf.compat.v1.get_default_graph().get_tensor_by_name('import/input:0')
y_output = tf.compat.v1.get_default_graph().get_tensor_by_name('import/output/BiasAdd:0')
batch_size = 1
x_test = x_test[0:batch_size]
# Create configuration to encrypt input
configx = get_config_for_ngraph_client(x_input.name)
print(configx)
print(x_test.shape)
with tf.compat.v1.Session(config=configx) as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    start_time = time.time()
    y_hat = y_output.eval(feed_dict={x_input: x_test})
    elasped_time = time.time() - start_time
    print("total time(s)", np.round(elasped_time, 3))


