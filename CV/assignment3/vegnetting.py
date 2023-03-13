import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default="", type=str, help="path to the image")
    opt = parser.parse_args()
    return opt

def read_n_resize(image, size, RGB = False):
    """read and resize the iamge
    
    Parameters
    ----------
    image: str
      path of the iamge
      
    size: tuple
      resize dimensions 
      
    Return
    ------
    output_image: np.ndarray
      resized_image
      """
    image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, size, interpolation= cv2.INTER_AREA)
    if RGB:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def apply_vegnetting(image, resize_dim = (640, 640)):
    """apply vegnetting effect on an image.

    The Vignette filter is generally used to focus viewer
    attention on certain parts of the image without hiding
    other parts completely. Generally focused part has higher
    brightness and saturation and brightness and saturation
    decrease as we go radially out from center to periphery. 

    Parameters
    ----------
    image: str
      path to the image 
    
    resize_dim: tuple
      image resize dimension (spatial dimension)


    Return
    ------

    output_image: np.ndarray
      image with vegnetting effect
     """
    image = read_n_resize(image, resize_dim)
    #get the height and width of the image
    rows, cols = image.shape[:2]
    #generate vegnetting masks using gaussina kernels, you can change sigma value
    X_gaussian = cv2.getGaussianKernel(ksize = int(rows), sigma = 200)
    Y_gaussian = cv2.getGaussianKernel(ksize = int(cols), sigma = 200)

    XY_gaussian = Y_gaussian * np.transpose(X_gaussian)
    #now normalize the mask
    mask = (255 * XY_gaussian ) / (np.linalg.norm(XY_gaussian))
    #create a copy of original iamge
    image_copy = image.copy()

    #apply the mask to each color channel of the image (B, G, R)
    for i in range(image.shape[-1]):
        image_copy[:, :, i] = image_copy[:, :, i] * mask

    return image_copy


def main():
    """everything goes here..."""
    args = read_args()
    if args.image:
        orig_image = read_n_resize(args.image, (640, 640), True)
        output_image = apply_vegnetting(args.image)
        fig, axxr = plt.subplots(1, 2)
        titles = ["original", "Vegnetting effect"]
        images = [orig_image, output_image]
        for i in range(len(images)):
            axxr[i].imshow(images[i])
            axxr[i].set_title(titles[i])
            axxr[i].set_axis_off()
        plt.show()
    print('done!!')


if __name__ == "__main__":
    main()

    