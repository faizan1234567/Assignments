import cv2
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default="images/", type = str, help= "path to the images")
    parser.add_argument('--save', default="/runs", type=str, help="path to save results")
    opt = parser.parse_args()
    return opt

def read_n_resize(path):
    '''read and resize an image
    
    params
    ------
    path: str
    '''
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    return img

def main():
    args = read_args()
    images_paths = os.listdir(args.images)
    print(f'There are {len(images_paths)} images')
    IMAGES = []
    for path in images_paths:
        image_path = os.path.join(args.images, path)
        img = read_n_resize(image_path)
        IMAGES.append(img)
  
    stitchy=  cv2.Stitcher.create()
    (dummy,output) = stitchy.stitch(IMAGES)
  
    if dummy != cv2.STITCHER_OK:
        print("stitching ain't successful")
    else: 
        print('Your Panorama is ready!!!')
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{args.save}/panorma_image.png', output)
        cv2.imshow('final result',output)
        cv2.waitKey(0)

if __name__ == '__main__':
    main()