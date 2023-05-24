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
    A, _ = sigmoid(Z)
    dz = dA * A * (1 - A)
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
        Z, linear_cache = forward_linear(Aprev, W, b)
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
    return AL, caches


def calculate_cost(AL, Y):
    """calculate cost between AL and Y, it's logistic regression cost
    
    AL:  np.ndarray
    Y:   np.ndarray
    
    """
    m = AL.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)
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

def backward_linear(dZ, cache):
    """calculate the derivate of W, b, and dAprev wrt to cost
    
    dz is the derivate of linear activation wrt to cost 
    
    params:
    ------
    dZ: np.ndarray
    cache: tuple"""
    Aprev, W, b = cache
    m = Aprev.shape[1]
    dW = 1./m * (np.dot(dZ, Aprev.T))
    db = 1./m * (np.sum(dZ, axis = 1, keepdims = True))
    dAprev = np.dot(W.T, dZ)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    assert (dAprev.shape == Aprev.shape)
    return dW, db, dAprev


def backward_nonlinear(dA, cache, activation = 'relu'):
    """implementing backward pass using non linear activation such as relu or sigmoid
    
    dA: np.ndarray
    cache: tuple
    activation: str"""

    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dZ = diff_sigmoid(dA, activation_cache)
        dW, db, dAprev = backward_linear(dZ, linear_cache)
    elif activation == "relu":
        dZ = diff_relu(dA, activation_cache)
        dW, db, dAprev = backward_linear(dZ, linear_cache)
    
    return dAprev, dW, db

def backpropagation(AL, Y, caches):
    """implemente backpropagtion for the L layers NN model
    
    AL is the model prediction while Y is the ground truth labls
    cahces are the data structure stored during the forward prop
    
    params
    ------
    AL: np.ndarray
    Y: np.ndarray
    caches: tuple"""
    m = Y.shape[1]
    grads = {}
    L = len(caches) # number of layers
    Y = Y.reshape(AL.shape)
    #final layer activation gradient
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    # print(len(current_cache))
    grads["dA" + str(L -1)], grads["dW" + str(L)], grads["db" + str(L)] = backward_nonlinear(dAL, current_cache, activation='sigmoid')

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA = grads["dA" + str(l + 1)]
        temp_dAprev, temp_dW, temp_db = backward_nonlinear(dA, current_cache, activation= "relu")
        grads["dA" + str(l)] = temp_dAprev
        grads["dW" + str(l + 1)] = temp_dW
        grads["db" + str(l + 1)] = temp_db

    return grads


def update_params(parameters, grads, learning_rate):
    """update parameters for gradient descent update
    
    parameters: dict
    grads: dict
    learning_rate: float"""

    L = len(parameters) // 2
    for l in range(1, L):
        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]
    
    return parameters

def predict(X, Y, parameters, threshold = 0.5, print_acc = True):
    """measure accuracy on the given data using the trained parameters
    
    params
    ------
    X: np.ndarray
    Y: np.ndarray
    parameters: dict
    threshold: float"""
    m = X.shape[1]
    L = len(parameters) //2
    predictions = np.zeros((1, m))

    #forward_prop
    probablities, caches = forward_propagation(X, parameters=parameters)

    for i in range(0, probablities.shape[1]):
        if probablities[0, i] > threshold:
            predictions[0, i] = 1
        else:
            predictions[0, i] = 0
    if print_acc:
        print("Accuracy: " + str(np.sum(predictions == Y)/m))
    return predictions

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
