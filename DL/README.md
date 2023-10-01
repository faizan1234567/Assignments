# KNN Based Classification on a Small Subset of CIFAR-10 Dataset
## Introduction
Humans poses a great capability in visual recognition tasks even though they may vary in size, pose, scale, and illumination. In the past many studies have been published to add human intelligence in computers. Recently, ```deep neural networks``` have shown great performance in visual perception tasks and have reported to surpass humans. Yet there is lack of thorough and fair comparison between human and artificial recognition systems.  ```CIFAR-10 dataset```, a well know dataset of natural images. This dataset allows fair comparison with state-of-the-art deep neural networks. ```Convolutional neural networks``` show great performance on the benchmark dataset; however, they are still far from human recognition capabilities. Moreover, a detailed investigation using multiple levels of difficulty reveals that easy images for humans may not be easy for deep neural networks. [1] 

![alt text](images/cifar10.png)

Figure 1:  CIFAR10 dataset image [1]

The ```CIFAR-10``` dataset (Canadian Institute for Advanced Research, 10 classes) is a subset of the Tiny Images dataset and consists of ```60000 32x32``` color images. The images are labelled with one of ```10``` mutually exclusive classes: airplane, automobile (but not truck or pickup truck), bird, cat, deer, dog, frog, horse, ship, and truck (but not pickup truck). There are ```6000``` images per class with ```5000``` training and ```1000``` testing images per class. [2]

## The K-NN Algorithm
The KNN algorithm assumes similar things exist in proximity. In simple words, similar things are near to each other. 

![alt text](images/KNN.webp)

Notice in the above image similar data points are close to each other and they form a pattern. In addition, there is a boundary between each group which separate one group from another. [3]



## References
[1]	T. Ho-Phuoc, “CIFAR10 to Compare Visual Recognition Performance between Deep Neural Networks and Humans,” 2018, [Online]. Available: http://arxiv.org/abs/1811.07270

[2]	“CIFAR10 dataset description.” [Online]. Available: https://paperswithcode.com/dataset/cifar-10

[3] O. Harrison, “machine learning with K Nearest Neighbor algorithm.” 2018. [Online]. Available: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

