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
    parser.add_argument('-k', "--k", default=3, type = int, 
                        help = "the value of K for algorithm")
    parser.add_argument('--default', action= 'store_true', help = 'use sklearns implementation')
    opt = parser.parse_args()
    return opt

def KNN(data, query, K = 3, dist_fn = None, choice_fn= None):
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

def KNN_sklearn(X: np.ndarray, y: np.ndarray, k: int = 3, query: np.ndarray = None):
    """
    Sklearn's KNN classifier
    -----------------------
    Parameters
    ---------
    X: training images
    y: labels
    k: n neighbors
    query: a query image

    """
    neigh = KNeighborsClassifier(n_neighbors= k)
    num_images, height, width, channels = X.shape
    X = X.reshape(num_images, -1)
    query = query.reshape(1, -1)
    neigh.fit(X, y)
    prediction = neigh.predict(query)
    return prediction

def eculidean_dist(img1: np.ndarray, img2: np.ndarray):
    """
    calculate the euclidean distance between the query and training image
    ---------------------------------------------------------------------

    Parameters
    ----------
    img1: np.ndarray (image 1)
    img2: np.ndarray (image 2)

    Return
    ------
    dist: float (distance btw images)
    """
    # preprocess the image into a vector
    img1_flatten = img1.flatten()
    img2_flatten = img2.flatten()

    # calculate the euclidean distance between two images
    dist = np.linalg.norm(img1_flatten - img2_flatten)
    return dist

def mode(labels):
    return Counter(labels).most_common(1)[0][0]

def normalize(X):
    return X/255.0

# TODO: code needs to be tested  with different values of K. 
# TODO: custom data loading and processing option should be added.
if __name__ == "__main__":
    args = read_args()

    # read the dataset
    logger.info('Loading the CIFAR10 dataset')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # normalize the 
    logger.info('Normalizing the dataset')
    x_train_norm = normalize(x_train)
    x_test_norm = normalize(x_test)
    
    # create a 20 samples dataset and set a query image from test set randomly
    logger.info(f"Creating a {args.subset} samples subset")
    x_train_subset, y_train_subset = create_subset(x_train_norm, y_train, samples=20)
    query_index = random.randint(0, len(x_test))
    query_image = x_test_norm[query_index]

    # run the KNN algorithm now
    logger.info(f'All done, now runing KNN algorithm on the dataset')
    if not args.default:
        k_neighbors, prediction = KNN(x_train_subset, query_image, args.k, eculidean_dist, mode)
    else:
        prediction = KNN_sklearn(x_train_subset, y_train_subset, args.k, query_image)
    logger.info(f'Predicted label for the query image: {int(prediction[0])}')
    logger.info(f'Actual label of the query image: {int(y_test[query_index][0])}')
