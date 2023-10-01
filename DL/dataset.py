"""
create a subset of 20 samples from CIFAR-10 dataset
---------------------------------------------------

Author: Muhammad Faizan

python dataset.py 
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import math
import argparse
import logging
from pathlib import Path
import os
import sys

# append root if doesn't exists in the system path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# create a logger to write logs in the file and stream on the terminal 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if os.path.exists('/logs'):
    pass
else:
    os.makedirs('/logs')
file_handler = logging.FileHandler('/logs/all_logs.log')
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# read command line arguments
def read_args():
    """
    getting args from the user
    --------------------------
    """
    parser = argparse.ArgumentParser(description= "command line arguments option")
    parser.add_argument("-s", "--subset", default= 20, type = int,  
                        help = "set size of the subset dataset, by default its 20")
    opt = parser.parse_args()
    return opt








if __name__ == "__main__":
    # get command line args from the user
    args = read_args()

    # get cifar10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    print(type(x_train))
