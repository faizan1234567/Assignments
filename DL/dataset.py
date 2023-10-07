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
from utils import *

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets


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
    parser.add_argument('--img', type = int, default= 224, help = "input image size")
    parser.add_argument('--transform', action= 'store_true', help = "apply transforms to the image")
    parser.add_argument('--visualize', action= 'store_true', help = 'visualize the dataset')
    opt = parser.parse_args()
    return opt



def load_dataset(images: str = "dataset/", 
                 batch_size: int = 1, 
                 shuffle: bool = False, 
                 transform = None):
    """
    Load the image dataset from the specified directories
    -----------------------------------------------------
    images: str 
    batch_size: int
    shuffle: bool
    transform: torchvision.transforms
    """
    dataset = datasets.ImageFolder('path/to/data', transform=transform)
    loader = DataLoader(dataset, batch_size = batch_size, 
                        shuffle = shuffle, pin_memory= True, 
                        num_workers= 4)
    return loader



     
if __name__ == "__main__":
    # get command line args from the user
    args = read_args()

    # cifar 10 label index dict
    class_map = {
        0: 'non-cat',
        1: 'cat' }
    
    #TODO: data loading testing.
    transfromations = image_transforms(args.img, kind = 'train')
    print(transfromations)


    