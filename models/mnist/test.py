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

def server_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--enable_client",
        type=str2bool,
        default=False,
        help="Enable the client")
    parser.add_argument(
        "--enable_gc",
        type=str2bool,
        default=False,
        help="Enable garbled circuits")
    parser.add_argument(
        "--mask_gc_inputs",
        type=str2bool,
        default=False,
        help="Mask garbled circuits inputs",
    )
    parser.add_argument(
        "--mask_gc_outputs",
        type=str2bool,
        default=False,
        help="Mask garbled circuits outputs",
    )
    parser.add_argument(
        "--num_gc_threads",
        type=int,
        default=1,
        help="Number of threads to run garbled circuits with",
    )
    parser.add_argument(
        "--backend", type=str, default="HE_SEAL", help="Name of backend to use")
    parser.add_argument(
        "--encryption_parameters",
        type=str,
        default="",
        help=
        "Filename containing json description of encryption parameters, or json description itself",
    )
    parser.add_argument(
        "--encrypt_server_data",
        type=str2bool,
        default=False,
        help=
        "Encrypt server data (should not be used when enable_client is used)",
    )
    parser.add_argument(
        "--pack_data",
        type=str2bool,
        default=True,
        help="Use plaintext packing on data")
    parser.add_argument(
        "--start_batch", type=int, default=0, help="Test data start index")
    parser.add_argument(
        "--model_file",
        type=str,
        default="",
        help="Filename of saved protobuf model")
    parser.add_argument(
        "--input_node",
        type=str,
        default="import/input:0",
        help="Tensor name of data input",
    )
    parser.add_argument(
        "--output_node",
        type=str,
        default="import/output/BiasAdd:0",
        help="Tensor name of model output",
    )

    return parser