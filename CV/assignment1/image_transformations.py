"""Apply a variety of image transformations to tranform an input image, and show transformed
image.

Assignment: 01
Author: Muhammad Faizan
Subject: Computer vision

Transformations
---------------
Scaling: apply scalling to an image to scale up or down an image
Euclidean: apply rotation + translation
Translation: apply translation
Affine: apply affine transformation
perspective: apply perspective transfrom"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def read_args():
    parser = argparse.ArgumentParser(description="read an image from directory")
    parser.add_argument("--image", default= "", type = str, help="path to an image to apply transformations")
    parser.add_argument("--scale", action="store_true", help="scale up or down an image")
    parser.add_argument("--scale_factor", default= 1.5, type=float, help="image scaling factor")
    parser.add_argument("--rotate", action="store_true", help="rotate image")
    parser.add_argument("--angle", default=45, type= int, help="image rotation angle")
    parser.add_argument("--translate", action="store_true", help="translate image")
    parser.add_argument("--euclidean", action="store_true", help="apply rotation + translation")
    parser.add_argument("--affine", action="store_true", help="apply affine transformation")
    parser.add_argument("--perspective", action="store_true", help="apply perspective transform")
    parser.add_argument("--visualize", action="store_true", help="show output")
    opt = parser.parse_args()
    return opt

class Transformations:
    """Apply set of transformations as assinged in assignment 1
    """
    def __init__(self, image) -> None:
        self.image = cv2.resize(cv2.imread(image, cv2.IMREAD_UNCHANGED), (640, 640))
        # print(f'IMAGE SHAPE: {self.image.shape}')


    def scale(self, scale_factor):
        """scale an input image using scaling factor, the scaling factor less than 1 means
        image dimension will reduce, whereas greater than means it will scale up. However, it will 
        stay as it if scaling factor is 1
        
        Parameter
        ---------
        scale_factor: float
                      image scaling factor 
        """
        scale_factor = float(scale_factor)
        heigth, width = self.image.shape[:2]
        dims = (int(heigth * scale_factor), int(width * scale_factor))
        image = self.image.copy()
        resized_img = cv2.resize(image, dims, interpolation=cv2.INTER_AREA)
        return resized_img
    
    def translate(self, tx, ty):
        """Translate the image using M transformation matrix
        
        Parameters
        ----------
        tx: int
            translation in x axis
        ty: int
            translation in y axis
        
           Transformation matrix that translate image into x and y units using tx and ty
           [[1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]]
        """
        height, width = self.image.shape[:2]
        M = np.array([[1, 0, tx],
                      [0, 1, ty]], dtype = np.float32)
        img = self.image.copy()
        dst = cv2.warpAffine(img, M, (height, width))
        return dst
    
    def rotate(self, angle, translated_img = None):
        """apply 2D rotation to an input image
        
        Parameters
        ----------
        angle: float
               rotation angle value
        """
        if translated_img is None:
            height, width = self.image.shape[:2]
            img = self.image.copy()
        else:
            height, width = translated_img.shape[:2]
            img = translated_img.copy()

        M = cv2.getRotationMatrix2D(((width - 1)/2.0,(height  -1)/2.0), angle ,1)
        dst = cv2.warpAffine(img, M, (width, height))
        return dst

    def euclidean(self, angle, tx, ty):
        """apply both translation and rotation simultaneously

        Parameters
        ----------
        angle: float
               rotation angle value
        tx: int
            translation in x-axis
        ty: int
            translation in y-axis
        """
        translated = self.translate(tx, ty)
        rotated_translated= self.rotate(angle, translated_img=translated)
        return rotated_translated
    
    def affine(self,pt1 = None, pt2 = None, pt3 = None):
        """apply affine transformation to the image
        
        Parameters:
        ----------
        pt1: tuple
             first point
        pt2: tuple
              second point
        pt3: tuple
              thrid point
        """
        heigth, width = self.image[:2]
        img = self.image.copy()
        #get three points
        pts1 = np.float32([[50, 50],
                           [200, 50],
                           [50, 200]])
 
        pts2 = np.float32([[10, 100],
                           [200, 50],
                           [100, 250]])
        # M = np.float32([[1, 0, 100], [0, 1, 50]])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(img, M, (int(width), int(heigth)))
        return dst

    def perspective(self):
        """apply perspective transform, you will need 4 points on input image and 4 points on 
        corresponding output image, this transfrom perserve straight lines"""
        rows, cols, ch = self.image.shape
        img = self.image.copy()
        pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
        pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        dst = cv2.warpPerspective(img, M, (300,300))
        return dst

def main():
    args = read_args()
    if args.image:
        transform = Transformations(args.image)
        if args.scale:
            img = transform.scale(args.scale_factor)
            print(f'scaled image shape: {img.shape}')
            label = "Scaling"
        elif args.rotate:
            img = transform.rotate(args.angle)
            label = "Rotation"
        elif args.translate:
            # translate 30 units in each direction
            img = transform.translate(30, 30)
            label = "Translation"
        elif args.euclidean:
            #rotate 45 degrees and translate 30 units in each direction
            img = transform.euclidean(45, 30, 30)
            label = "Translation and rotation"
        elif args.affine:
            img = transform.affine()
            label = "Affine"
        elif args.perspective:
            img = transform.perspective()
            label = "Perspective"
        else:
            print("No transformation option selected!!")
            img = None

        if args.visualize and img is not None:
            matplot = False
            if matplot:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img)
                plt.xticks([])
                plt.yticks([])
                plt.title(f"{label}")
                plt.show()
            else:
                cv2.imshow(f"{label}", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    print('Done!!!')

if __name__ == "__main__":
    main()
        















