"""
Implementation of KNN algorithm on CIFAR-10 dataset (a subset)
KNN is implemented from scrath, and a SKlearn implementation is also 
added.
=====================================================================

Author: Muhammad Faizan
python knn.py -h for help
"""
import numpy as np
import math
import random
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import os
import sys
import math
import argparse
from collections import Counter
from dataset import visualize_cifar10, create_subset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from dataset import *
from utils import *

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
    parser.add_argument("--k", default=3, type = int, required= True,
                        help = "the value of K for algorithm")
    parser.add_argument('--default', action= 'store_true', help = 'use sklearns implementation')
    parser.add_argument('--split_size', type = float, default= 0.2, required= True,
                        help= "dataset split ratio")
    parser.add_argument('--img', type = int, default= 32, required= True,
                        help= 'image size')
    parser.add_argument('--data', type = str, default= "dataset/", required=True,
                        help = "dataset path")
    parser.add_argument('--batch', type = int, default=30, help = 'batch size')
    parser.add_argument('--report', action= 'store_true', help = 'print results report')
    parser.add_argument('--transform', action= "store_true", help = "dataset transforms options")
    opt = parser.parse_args()
    return opt

def KNN_scratch(data, query, K = 3, dist_fn = None, choice_fn= None):
    """
    Implementation of KNN algorithm from scrath
    -------------------------------------------

    Parameters
    ----------
    data: np.ndarray
    query: np.ndarray
    K: int
    """
    distances_indices = []
    
    # for each example
    for index, example in enumerate(data[0]):
        # calculate the distance betwwen the query and current image
        distance = dist_fn(example, query)
        distances_indices.append((distance, index))
    # sort the collection
    sorted_distances_indices = sorted(distances_indices)
    # pick the first k items
    k_nearest_neighbours_and_indices = sorted_distances_indices[:K]
    # get the corresponding ground truth for the selected items
    k_nearest_labels = [data[1][i][-1] for distance, i in k_nearest_neighbours_and_indices]

    return k_nearest_neighbours_and_indices, choice_fn(k_nearest_labels)

# Scikit-Learn implemenation of KNN algorithm
def KNN_sklearn(images, labels, k: int = 3, split: float = 0.2,
                distance: str = "Eculidean"):
    """
    Sklearn's KNN classifier on the classification dataset containing
    three classes such as cat, car, and dog.
    - Load and transform the  dataset 
       - rescale to (224, 224) image 
       - convert to torch Tensor
       - Normalize the pixel intensities with given mean and standard deviation
    - convert to numpy array
    - split the dataset into training and testing sections
    - initilize k value and distance metric (experimental to be varied for improvment)
    - training the KNN classifier on the dataset given the configuration
    - if results good enough, abort, otherwise change configurations 
    -----------------------------------------------------------------------------------
    
    
    Parameters
    ----------
    data: training images and labels
    k: n neighbors (hyperparameter)
    split: split size btw training and testing
    distance: distance metric to be used for picking top vots
    """
    # you may use different values of K, p =2 is euclidean distance
    if distance == "Euclidean":
        classifier = KNeighborsClassifier(n_neighbors= k, p = 2) # p = 1 (manhatten distance)
    else:
        classifier = KNeighborsClassifier(n_neighbors= k, p = 1)
    
    # prepare the dataset
    # channel last conversion, in pytorch natively it was channel first
    images = images.permute(0, 2, 3, 1)
    # convert to numpy array
    images_np = images.numpy()
    labels_np = labels.numpy()
    # now flatten the images to have (m, n)
    # where m is number of samples and n is features such as pixels intensities (32x32x3)
    images_processed = images_np.reshape(images_np.shape[0], -1)
    
    # now split the dataset into traini and test
    (trainX, testX, trainY, testY) = train_test_split(images_processed, labels_np, test_size = split,
                                                      random_state= 42)
    print('processed.')
    # classifier.fit(X, y)
    # prediction = classifier.predict(query)
    # return prediction



# TODO: code needs to be tested  with different values of K. (PENDING)
# DONE: custom data loading and processing option should be added.
if __name__ == "__main__":
    # configs
    args = read_args()
    transformations = image_transforms(img = args.img)

    # read the dataset
    logger.info('Loading the Custom dataset')
    data_loader = load_dataset(images= args.data, batch_size= args.batch, 
                               shuffle= False, transforms= transformations if args.transfroms else None)
    # split the images and the labels
    images, labels = next(iter(data_loader))
    class_to_idx = data_loader.dataset.class_to_idx
    logger.info(f"The datast classes information: {class_to_idx}")

    # now run the trainner on the dataset


    
    
    