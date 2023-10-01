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

def visualize_cifar10(x_test: np.ndarray, y_test: np.ndarray):
    """
    Visualize the samples in the CIFAR10 dataset 
    as part of data exploration and data preprocessing
    --------------------------------------------------

    Parameters
    ----------
    x_test: test image dataset (np.ndarray)
    y_test: test labels dataset (np.ndarray)

    """
    # plot an image with the label
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
    # increae k initialization to get other samples set
    fig, axes = plt.subplots(5, 5, figsize = (10 , 8))
    k = 0
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].imshow(x_test[k])
            axes[i, j].set_title(class_map[int(y_test[k][0])])
            axes[i, j].axis('off')
            k += 1
    plt.show()

def number2perent(required_images: int, total_size: int):
    """
    get the percentage of the images you want to retreive
    -----------------------------------------------------
    Parameters
    ----------
    required_images: desired images (int)
    total_size: total size of the dataset(int)

    Return
    ------
    percentage: float
    """
    return (required_images/total_size) * 100

def create_subset(X: np.ndarray, y: np.ndarray, samples: int = 20):
    """
    create a subset of images and labels in the data frame format
    -------------------------------------------------------------

    X: total training images
    y: total training labels
    frac: percentage of data to be extracted if specifed
    samples: number of samples to be used if frac not specifed.
    """
    # normalize images to 0 and 1
    train_images = X/255.0

    # create a small subset as specified by the assignmet descriptiom
    df = pd.DataFrame(list(zip(train_images, y)), columns= ["image", "label"])
    df_small = df.sample(n = samples) # specify frac = ... if you are using percentage
    X_small = np.array([ i for i in list(df_small['image'])])
    y_small = np.array([ [i[0]] for i in list(df_small['label'])])
    return (X_small, y_small)
    



    
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
        visualize_cifar10(x_test, y_test)
    
    logger.info(f'Creating a subset of {args.subset} samples')
    x_small, y_small = create_subset(x_test, y_test, args.subset)