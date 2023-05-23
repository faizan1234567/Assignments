'''implement frame to implement a generic neural network architecure to
calculate forward prop and then backprop

Author: Muhammad Faizan
Date: 23 May 2023


Some of the code of this exercise is borrowed from: https://github.com/faizan1234567/Deep-Learning-course/blob/master/Deep_learning_specialization_coursera/Neural%20Networks%20and%20Deep%20Learning/week%204/part%202/dnn_app_utils_v3.py'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0] # project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def sigmoid(Z):
    """"implement the sigmoid activation in numpy
    
    Parameters
    ----------
    Z: np.ndarray
    """
    A = (1/(1 + np.exp(-Z)))
    cache = Z
    return A, cache


def relu(Z):
    """implement the relu activation in numpy
    
    Parameters
    ----------
    Z: np.ndarray
    """
    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)
    cache = Z
    return A, cache

def diff_sigmoid(dA, cache):
    """differentiate sigmoid function with rest to Z
    
    It is obviously the derivate of activatoin times the sigmoid derivative wrt Z
    
    params
    ------
    dA: np.ndarray
    cache: np.ndarray
    """
    Z = cache
    A = sigmoid(Z)
    dz = dA * A * (1- A)
    assert (dz.shape == Z.shape)
    return dz

def diff_relu(dA, cache):
     """differentiate relu function with respect to Z
    
    set it is derivative to zero if its less than 0
    
    params
    ------
    dA: np.ndarray
    cache: np.ndarray
    """
     Z = cache
     dz = np.array(dA, copy= True)
     dz[Z <= 0] = 0
     assert (dz.shape == Z.shape)
     return dz

def initialize_deep_nn_parameters(layer_dims):
    """initialize parameters for deep neural net for l layers
    
    params
    ------
    layer_dims: list"""
    np.random.seed(1)
    parameters = {}
    total_layers = len(layer_dims)

    for i in range(1, total_layers):
        parameters["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1])/np.sqrt(layer_dims[i-1])
        parameters["b" + str(i)] = np.zeros((layer_dims[i], 1))

        assert (parameters["W" + str(i)].shape == (layer_dims[i], layer_dims[i - 1]))

    return parameters

def forward_linear(Aprev, W, b):
    """calculate forward prop linear part 
    
    params
    ------
    Aprev: np.ndarray
    W:np.ndarray
    b:np.ndarray"""
    cache = (Aprev, W, b)
    Z = np.dot(W, Aprev) + b
    assert (Z.shape == (W.shape[0], Aprev.shape[1]))
    return Z, cache

def forward_nonlinear(Aprev, W, b, activation = 'relu'):
    """implement linear --> nonlinear
    
    A_prev-->Z-->A
    
    params
    -----
    Aprev: np.ndarray
    W: np.ndarray
    b: np.ndarray
    activation: str"""
    if activation == 'relu':
        Z, linear_cache = forward_linear(Aprev, W, b)
        A, activation_cache = relu(Z)
    elif activation == 'sigmoid':
        Z, linaer_cache = forward_linear(Aprev, W, b)
        A, activation_cache = sigmoid(Z)
    cache = (linear_cache, activation_cache)
    assert (A.shape == (W.shape[0], Aprev.shape[1]))
    return A, cache

def forward_propagation(X, parameters):
    """implement the forward propagtion part for the model
    
    X: np.ndarray
    parameters: dict"""

    A = X
    L = len(parameters) //2
    caches = []
    for l in range(1, L):
        Aprev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = forward_nonlinear(Aprev, W, b, activation='relu')
        caches.append(cache)

    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    #for binary classification, change to softmax for multiclass depending on your usecase..
    AL, cache = forward_nonlinear(A, W, b, activation= 'sigmoid')
    caches.append(cache)
    assert(AL.shape == (1, X.shape[1]))
    return AL, cache

def calculate_cost(AL, Y):
    """calculate cost between AL and Y, it's logistic regression cost
    
    AL:  np.ndarray
    Y:   np.ndarray
    
    """
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.sqeeuze(cost)
    assert (cost.shape == ())
    return cost

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


if __name__ == "__main__":
    #testing activation functions
    data = load_data()
    print(data[-1])
    Z = np.array([[0.2, 0.3, 0.0], 
                  [-0.4, 0, 1.0 ],
                  [-10, 10, 0.5 ]], dtype = np.float32)
    print("relu: \n")
    print(relu(Z))
    print()
    print("Sigmoid: \n")
    print(sigmoid(Z))
