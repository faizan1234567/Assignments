import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
import random
import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type= str, default= "", help = "path to image")
    opt = parser.parse_args()
    return opt


def read_n_resize(image):
    '''read and resize the image
    
    params
    ------
    image: str
    '''
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    return img

def main():
    args = read_args()
    IMAGES = []
    TITLES = []
    img = read_n_resize(args.img)
    img_orginal = img.copy()
    IMAGES.append(img_orginal)
    TITLES.append('original')
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    t_lower = 30 # Lower Threshold
    t_upper = 70  # Upper threshold
    minLineLength = 1
    maxLineGap = 10
    edges = cv2.Canny(gray_img, t_lower, t_upper, 3)
    IMAGES.append(edges)
    TITLES.append('edges')
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength, maxLineGap)
    for line in lines:
        x0, y0, x1, y1 = line[0]
        cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), 2)
    IMAGES.append(img)
    cv2.imwrite('images/result.png', img)
    TITLES.append('result')

    fig, axes = plt.subplots(1, 3, figsize = (10, 8))
    for i in range(len(IMAGES)):
        axes[i].imshow(IMAGES[i], cmap = 'gray')
        axes[i].set_title(TITLES[i])
        axes[i].set_axis_off()
    plt.show()


if __name__ == "__main__":
    main()


