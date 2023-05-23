'''
Training on image dataset
This assignment will implement end to end deep learning in numpy. Numpy is linear algebra and scientific computing libaray in python. In this assignment, the following thing will be covered.

- Import packages
- Model architecture design
- Random initialization of parameters
- Forward propagation
- Cost calculation
- Backward propagation
- Parameters update
- Evaluation of the model on the dataset
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random
from PIL import Image
from scipy import ndimage
from utils.deep_nn import *
import argparse

#matplotlib variables for plotting...
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
random.seed(1)

def read_args():
    """command line args
    
    read some important variables such hyperparmeters and model setting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default = "/datasets", help = 'dataset dir')
    parser.add_argument('--lr', default = 0.001, type = float, help = 'learning rate value')
    parser.add_argument('--iterations', default = 5000, type = int, help = 'training iterations')
    parser.add_argument('--img', default= "/test_images/1.png", type = str, help= 'a test image')
    parser.add_argument('--label', default= None, type = int, help= "img label if given")
    return parser.parse_args()

def preprocess_dataset():
    """preprocess the dataset for training
    
    split the dataset into training and test sets
    normalize the dataset
    flatten for NN input
    print its attributes"""
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]

    print ("Number of training examples: " + str(m_train))
    print ("Number of testing examples: " + str(m_test))
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
    print ("train_x_orig shape: " + str(train_x_orig.shape))
    print ("train_y shape: " + str(train_y.shape))
    print ("test_x_orig shape: " + str(test_x_orig.shape))
    print ("test_y shape: " + str(test_y.shape))

    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.

    print ("train_x's shape: " + str(train_x.shape))
    print ("test_x's shape: " + str(test_x.shape))

    return train_x, train_y, test_x, test_y, classes


def L_layer_model(X, Y, 
                layer_dims, 
                learning_rate = 0.007, 
                num_iterations = 4000, 
                print_cost = False):
    '''
    implement L layer general model architecture for training 

    Parameters
    ----------
    X: np.ndarray (input features)
    Y: np.ndarray (labels)
    layers_dims: list (model architecture)
    learning_rate: float (step size for learning)
    num_iterations: int (total iterations to run gradient descent)
    print_cost: bool (whether to print cost or not)
    '''
    np.random.seed(1)
    costs = []

    #initialize parameters
    parameters = initialize_deep_nn_parameters(layer_dims)

    # run training
    for iter in range(0, num_iterations):

        AL, caches = forward_propagation(X, parameters)

        cost = calculate_cost(AL, Y)

        grads = backpropagation(AL, Y, caches)

        parameters = update_params(parameters, grads, learning_rate)

        if print_cost and iter % 100 == 0 or iter == num_iterations - 1:
            print("Cost after iteration {}: {}".format(iter, np.squeeze(cost)))
        if iter % 100 == 0 or iter == num_iterations:
            costs.append(cost)
    return parameters, costs

def test_on_image(image_path, img_label, parameters, num_px = 64):
    """test model weights on an image
    
    params
    ------
    image_path: str
    image_label: int [0] for non cat and [1] for cat
    """
    index_classes = {0: 'Not cat',
                     1: 'Cat'}
    image = np.array(Image.open(image_path).resize((num_px, num_px)))
    img = image/255
    img = img.reshape((1, num_px * num_px * 3)).T
    prediction = predict(img, img_label, parameters)
    plt.imshow(image)
    plt.title(f"Model prediction: {index_classes[int(prediction)]}")
    plt.xticks([]), plt.yticks([])
    plt.show()

def plot_costs(costs, learning_rate=0.0075):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()


def main():
    """everything goes here.."""
    args = read_args()
    layer_dims = [12288, 20, 7, 5, 1]
    iterations = args.iterations
    lr = args.lr
    if args.img and args.label:
        test_image = args.img
        test_label = args.label
    X_train, Y_train, X_test, Y_test, classes = preprocess_dataset()
    parameters, costs = L_layer_model(X_train, Y_train, layer_dims= layer_dims,
                                      learning_rate= lr, num_iterations= iterations, print_cost=True)
    plot_costs(costs, lr)
    print("Training accuracy: \n")
    pred_train = predict(X_train, Y_train, parameters)
    print()
    print('Test accuracy: \n')
    pred_test = predict(X_test, Y_test, parameters)
    print()

    print('Test on an image..')
    test_on_image(test_image, test_label, parameters, 64)
    print('done')

if __name__ == "__main__":
    main()




