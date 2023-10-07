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



def image_transforms(img: int = 224,
                     kind = 'train'):
    """
    transfrom the image into a suitable format for preprocessing
    ------------------------------------------------------------
    img: int (image size)
    kind: str (use training or testing phase)

    """
    # add more augmentataion or transforms options as per the need.
    if kind == 'train':
        transformations = torchvision.transforms.Compose([
            transforms.Resize([img, img]), 
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std =  [0.229, 0.224, 0.225])
        ])
    else:
        transformations = torchvision.transforms.Compose([
            transforms.Resize([img, img]),
            transforms.ToTesnor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std =  [0.229, 0.224, 0.225])
        ])
    return transformations
    


