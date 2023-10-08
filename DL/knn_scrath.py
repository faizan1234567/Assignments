
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


# read command line arguments
def read_args():
    """
    getting args from the user
    --------------------------
    """
    parser = argparse.ArgumentParser(description= "command line arguments option")
    parser.add_argument("--k", default=3, type = int, required= True,
                        help = "the value of K for algorithm")
    parser.add_argument('--split_size', type = float, default= 0.2, required= True,
                        help= "dataset split ratio")
    parser.add_argument('--img', type = int, default= 32, required= True,
                        help= 'image size')
    parser.add_argument('--data', type = str, default= "dataset/", required=True,
                        help = "dataset path")
    parser.add_argument('--batch', type = int, default=30, help = 'batch size')
    parser.add_argument('--report', action= 'store_true', help = 'print results report')
    parser.add_argument('--manhatten', action= 'store_true', help= "use manhatten distance")
    opt = parser.parse_args()
    return opt


class KNNModel:
    def __init__(self, k, distance):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, img1, img2):
        return np.linalg.norm(img1 - img2)
    
    def manhatten_distance(self, img1, img2):
        return np.linalg.norm(img1 - img2, 1)
    
    def predict(self, X_test):
        y_pred = [self.train(X_test[i, :]) for i in range(X_test.shape[0])]
        return np.array(y_pred)

    def train(self, query, distance = "Eculidean"):
        if distance == "Euclidean":
            distances = [self.euclidean_distance(query, self.X_train[i, :]) for i in range(self.X_train.shape[0])]
        else:
            distances = [self.manhatten_distance(query, self.X_train[i, :]) for i in range(self.X_train.shape[0])]
        k_indices = sorted(range(len(distances)), key=lambda k: distances[k])[:self.k]
        k_nearest_labels = [int(self.y_train[i][0]) for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

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
    classifier = KNNModel(k = args.k, distance= 'Euclidean' if not args.manhatten else 'manhatten')
    classifier.fit(Xtrain, Ytrain)
    ytest_pred = classifier.predict(Xtest)
    metrics1, metrics2, metrics3 = calculate_metrics(Ytest.ravel(), ytest_pred, num_classes=3)
    acc = calculate_accuracy(Ytest, ytest_pred)
    logger.info(f'Accuracy: {acc}')
    if args.report:
        data = [["Metric", "precision", "recall", "f1-score", "accuracy"],
                ["car", metrics1[0], metrics1[1], metrics1[2], acc],
                ["cat", metrics2[0], metrics2[1], metrics2[2], acc],
                ["dog", metrics3[0], metrics3[1], metrics3[2], acc]]
            
        table = tabulate(data, headers="firstrow", tablefmt="grid")

        print(table)
