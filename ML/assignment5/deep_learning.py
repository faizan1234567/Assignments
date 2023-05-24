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
from tabulate import tabulate
import seaborn as sns


def read_args():
    """command line args
    
    read some important variables such hyperparmeters and model setting
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default = None, help = 'dataset dir')
    parser.add_argument('--lr', default = 0.001, type = float, help = 'learning rate value')
    parser.add_argument('--iterations', default = 5000, type = int, help = 'training iterations')
    parser.add_argument('--img', default= "/test_images/1.png", type = str, help= 'a test image')
    parser.add_argument('--label', default= None, type = int, help= "img label if given")
    parser.add_argument('--default_data', action= "store_true", help= 'use default data for testing...')
    return parser.parse_args()

def preprocess_dataset(print_info = True):
    """preprocess the dataset for training
    
    split the dataset into training and test sets
    normalize the dataset
    flatten for NN input
    print its attributes"""
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    m_train = train_x_orig.shape[0]
    num_px = train_x_orig.shape[1]
    m_test = test_x_orig.shape[0]


    train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_x = train_x_flatten/255.
    test_x = test_x_flatten/255.
    if print_info:
        print ("Number of training examples: " + str(m_train))
        print ("Number of testing examples: " + str(m_test))
        print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print ("train_x_orig shape: " + str(train_x_orig.shape))
        print ("train_y shape: " + str(train_y.shape))
        print ("test_x_orig shape: " + str(test_x_orig.shape))
        print ("test_y shape: " + str(test_y.shape))
        print ("train_x's shape: " + str(train_x.shape))
        print ("test_x's shape: " + str(test_x.shape))

    return train_x, train_y, test_x, test_y, classes


def print_cost_table(cost, iteration, table, headers):
    """print cost in a nice table
    
    params
    ------
    cost: float
    iterations: int"""

    row = [f"{iteration:04d}", f"{cost:.4f}"]
    table.append(row)

    print(tabulate(table, headers, tablefmt="fancy_grid"))

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
    table = []
    headers = ["Iterations", "Cost"]
    #initialize parameters
    parameters = initialize_deep_nn_parameters(layer_dims)

    # run training
    for iter in range(0, num_iterations):

        AL, caches = forward_propagation(X, parameters)

        cost = calculate_cost(AL, Y)

        grads = backpropagation(AL, Y, caches)

        parameters = update_params(parameters, grads, learning_rate)

        if print_cost and iter % 100 == 0 or iter == num_iterations - 1:
            print_cost_table(cost, iter, table, headers)
            
        if iter % 100 == 0 or iter == num_iterations:
            costs.append(cost)
    return parameters, costs

def test_on_image(parameters, num_px = 64, image_dir = None, 
                  default_data = None, print_acc = False):
    """test model weights on an image
    
    params
    ------
    image_path: str
    image_label: int [0] for non cat and [1] for cat
    """
    IMAGES = []
    INPUTS = []
    INPUT_LABELS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]
    index_classes = {0: 'Not cat',
                     1: 'Cat'}
    if default_data:
        train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
        train_x, _, test_x, _, _ = preprocess_dataset(print_info= False)
        indices = [random.randint(0, test_x_orig.shape[0] - 1) for _ in range(16)]
        k = 0 # for iterating over the images in teh folder.
        fig, axes = plt.subplots(4, 4, figsize = (15, 10))
        for i in range(len(indices)//4):
            for j in range(len(indices)//4): 
                axes[i, j].imshow(test_x_orig[indices[k]])
                x = test_x[:, indices[k]].reshape(-1, 1)
                y = test_y[:, indices[k]]
                prediction = predict(x, y, parameters, print_acc= print_acc)
                axes[i, j].set_title(f"Model prediction: {index_classes[int(prediction)]}")
                axes[i, j].set_axis_off()
                k+=1
        plt.show()

    elif image_dir is not None:
        for img in os.listdir(image_dir):
            img_path = os.path.join(image_dir, img)
            image = np.array(Image.open(img_path).resize((num_px, num_px)))
            img = image/255
            img = img.reshape((1, num_px * num_px * 3)).T
            IMAGES.append(image)
            INPUTS.append(img)
        fig, axes = plt.subplots(4, 4, figsize = (15, 10))
        k = 0
        for i in range(len(IMAGES)//4):
            for j in range(len(IMAGES)//4): 
                axes[i, j].imshow(IMAGES[k])
                prediction = predict(INPUTS[k], INPUT_LABELS[k], parameters)
                axes[i, j].set_title(f"Model prediction: {index_classes[int(prediction)]}")
                axes[i, j].set_axis_off()
                k+=1
        plt.show()
    

def plot_costs(costs, learning_rate=0.0075):
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    # sns.set(style="ticks", palette="colorblind")

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.lineplot(data=np.squeeze(costs), color='navy', linewidth=2.5)
    plt.title(f'Cost vs. Iterations with learning_rate {learning_rate}', fontsize=16)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(['Cost'], loc='upper right', fontsize=10)
    sns.despine()
    plt.show()


def main():
    """everything goes here.."""
    print('--'* 40)
    print('Configuring parameters... \n')
    args = read_args()
    layer_dims = [12288, 20, 7, 5, 1]
    iterations = args.iterations
    lr = args.lr
    if args.default_data:
        data_flag = args.default_data
    else:
        data_flag = False

    if args.data:
        image_dir = args.data
    else:
        image_dir = None
    if args.img and args.label:
        test_image = args.img
        test_label = args.label
    print('--'* 40)
    print('Now Loading the dataset... \n')
    X_train, Y_train, X_test, Y_test, classes = preprocess_dataset()
    print('--'* 40)
    print('Starting training...\n')
    parameters, costs = L_layer_model(X_train, Y_train, layer_dims= layer_dims,
                                      learning_rate= lr, num_iterations= iterations, print_cost=True)
    plot_costs(costs, lr)
    print('--'*40)
    print("Training accuracy: \n")
    pred_train = predict(X_train, Y_train, parameters)
    print()
    print('--'*40)
    print('Test accuracy: \n')
    pred_test = predict(X_test, Y_test, parameters)
    print()
    print('--'*40)
    print('Test on images..')
    test_on_image(parameters, 64, image_dir= image_dir, default_data= data_flag)
    print('All done!!!!')
    print('--'* 40)

if __name__ == "__main__":
    main()




