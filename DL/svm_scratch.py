"""
implement knn from the scrath
-----------------------------

Author: Muhammad Faizan
python knn_scrath.py -h
"""

import numpy as np
from collections import Counter
from utils import *
from dataset import *
from pathlib import Path
import sys
import os
import logging
from tabulate import tabulate

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


def read_args():
    """
    getting args from the user
    --------------------------
    """
    parser = argparse.ArgumentParser(description= "command line arguments option")
    parser.add_argument('--split_size', type = float, default= 0.2, required= True,
                        help= "dataset split ratio")
    parser.add_argument('--img', type = int, default= 32, required= True,
                        help= 'image size')
    parser.add_argument('--data', type = str, default= "dataset/", required=True,
                        help = "dataset path")
    parser.add_argument('--batch', type = int, default=30, help = 'batch size')
    parser.add_argument('--report', action= 'store_true', help = 'print results report')
    opt = parser.parse_args()
    return opt

class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

        
# TODO: slight updates to the code ..   
if __name__ == "__main__":
    args = read_args()
    transformations = image_transforms(img = args.img)

    # read the dataset
    logger.info('Loading the Custom dataset')
    data_loader = load_dataset(images = args.data, batch_size = args.batch, 
                               shuffle= False, transforms= transformations)
    # split the images and the labels
    images, labels = next(iter(data_loader))
    class_to_idx = data_loader.dataset.class_to_idx
    logger.info(f"The datast classes information: {class_to_idx}")
    X, y = process_data(images, labels)
    (Xtrain, Xtest, Ytrain, Ytest) = train_test_split(X=X, y=y, test_size= args.split_size, random_seed= 42)
    logger.info("Now fit the classfier")
    classifier = SVM(learning_rate= 0.001)
    classifier.fit(Xtrain, Ytrain)
    ytest_pred = classifier.predict(Xtest)
    print(ytest_pred)
