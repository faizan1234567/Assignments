# Assignments
Computer vision and ML course assignments.

## Computer vision
Code for each of the computer vision assignment 1 task. 
![Cat Picture](https://github.com/faizan1234567/Assignments/blob/main/CV/assignment1/images/cat1.jpg)
 ### Assignment 01: Image Transformations

   In this assignment, Following transformation needs to be applied. 
    - scaling
    - Rotation
    - Translation
    - Euclidean
    - Affine
    - Perspective

## Install & Usage
To get the code

```
git clone https://github.com/faizan1234567/Assignments
cd Assignments
```
To run assignmet1 code, use the following args

```
python image_transformations.py -h
```
For rotating an image for 45 degrees, use this

```
python image_transformations.py --image images/cat1.jpg --rotate --angle 45 --visualize
```
if you want to translate the image:
```
python image_transformations.py --image images/cat1.jpg --translate --visualize
```
And, for the image translation and rotation simultenously use:

```
python image_transformations.py --image images/cat1.jpg --euclidean --visualize
```

```
  -h, --help            show this help message and exit
  --image IMAGE         path to an image to apply transformations
  --scale               scale up or down an image
  --scale_factor SCALE_FACTOR
                        image scaling factor
  --rotate              rotate image
  --angle ANGLE         image rotation angle
  --translate           translate image
  --euclidean           apply rotation + translation
  --affine              apply affine transformation
  --perspective         apply perspective transform
  --visualize           show output
```
