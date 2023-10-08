"""
c utilities functions
---------------------------------------------------

Author: Muhammad Faizan

python utils.py 
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
from collections import Counter

import torch, torchvision
from torchvision import datasets, transforms

# visualization function
def visualize_cifar10(x_test: np.ndarray, y_test: np.ndarray, class_map: dict = {}):
    """
    Visualize the samples in the CIFAR10 dataset 
    as part of data exploration and data preprocessing
    --------------------------------------------------

    Parameters
    ----------
    x_test: test image dataset (np.ndarray)
    y_test: test labels dataset (np.ndarray)

    """
    # plot an images with the labels
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

# convert to percent if needed.
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

# create a subset from the dataset.
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



def image_transforms(img: int = 224):
    """
    transfrom the image into a suitable format for preprocessing
    1. reshape the image
    2. convert it to torch tensor
    3. normalize the image with given mean and standard deviation
    ------------------------------------------------------------
    img: int (image size)


    """
    # add more augmentataion or transforms options as per the need.

    transformations = torchvision.transforms.Compose([
        transforms.Resize([img, img]), 
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                std =  [0.229, 0.224, 0.225])
    ])
    
    return transformations
    

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

def train_test_split(X, y, test_size=0.2, random_seed=None):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - X: The feature matrix.
    - y: The target labels.
    - test_size: The proportion of the dataset to include in the test split (default is 0.2).
    - random_seed: Seed for random number generation (optional).

    Returns:
    - X_train: The training feature matrix.
    - X_test: The testing feature matrix.
    - y_train: The training target labels.
    - y_test: The testing target labels.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    num_samples = len(X)
    num_test_samples = int(test_size * num_samples)

    # Shuffle the indices of the data
    shuffled_indices = np.random.permutation(num_samples)

    # Split the shuffled indices into training and testing sets
    test_indices = shuffled_indices[:num_test_samples]
    train_indices = shuffled_indices[num_test_samples:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def calculate_accuracy(y_true, y_pred):
    correct_predictions = np.sum(y_true.ravel() == y_pred)
    total_predictions = len(y_pred)
    accuracy = correct_predictions / total_predictions
    return accuracy

def calculate_metrics(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Populate the confusion matrix
    for i in range(len(y_true)):
        true_class = y_true[i]
        pred_class = y_pred[i]
        confusion_matrix[true_class][pred_class] += 1

    # Calculate precision, recall, and F1-score for each class
    metrics = []
    for class_id in range(num_classes):
        true_positive = confusion_matrix[class_id][class_id]
        false_positive = sum(confusion_matrix[i][class_id] for i in range(num_classes)) - true_positive
        false_negative = sum(confusion_matrix[class_id]) - true_positive

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        metrics.append([precision, recall, f1_score])

    return metrics

def process_data(images, labels):
    images = images.permute(0, 2, 3, 1)
    # convert to numpy array
    images_np = images.numpy()
    labels_np = labels.numpy()
    # now flatten the images to have (m, n)
    # where m is number of samples and n is features such as pixels intensities (32x32x3)
    images_processed = images_np.reshape(images_np.shape[0], -1)
    labels_processed = labels_np.reshape(labels_np.shape[0], -1)
    return (images_processed, labels_processed)
