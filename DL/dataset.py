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
import random
import pandas as pd

# append root if doesn't exists in the system path
ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# create a logger to write logs in the file and stream on the terminal 
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if os.path.exists('logs'):
    pass
else:
    os.makedirs('logs')

file_handler = logging.FileHandler('logs/all_logs.log')
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.addHandler(file_handler)


# read command line arguments
def read_args():
    """
    getting args from the user
    --------------------------
    """
    parser = argparse.ArgumentParser(description= "command line arguments option")
    parser.add_argument("-s", "--subset", default= 20, type = int,  
                        help = "set size of the subset dataset, by default its 20")
    parser.add_argument('--visualize', action= 'store_true', help = 'visualize the dataset')
    opt = parser.parse_args()
    return opt


     
if __name__ == "__main__":
    # get command line args from the user
    args = read_args()

    # get cifar10 dataset from TensorFlow's Keras
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # plot an image with the label
    # cifar 10 label index dict
    class_map = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    if args.visualize:
        logger.info('Plotting a sample CIFAR10 dataset')
        visualize_cifar10(x_test, y_test, class_map=class_map)
    
    logger.info(f'Creating a subset of {args.subset} samples')
    x_small, y_small = create_subset(x_test, y_test, args.subset)
    # verify shapes
    logger.info(f'total images: {len(x_small)}, total labels: {len(y_small)}')