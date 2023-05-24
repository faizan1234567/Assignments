# Backpropagtion implementation using Numpy from scrath
Training a deep learning model involves series of step. The following step are necessary to to train a 
deep neural network sucessfully. 
- Model architecutre search
- Parameters initialization
- Forward Propagation
- Backward Propagation
- Parameters update
- Hyperparameters search
- Evaluation and improvment 

### Architecture
The implementation supports any kind of architecture, you can define different model architectures to check the 
performance. 
![alt text](https://github.com/faizan1234567/Assignments/blob/main/ML/assignment5/images/nn.png)

### Installation

```bash
git clone https://github.com/faizan1234567/Assignments.git
cd Assignments/ML/assignment5

python3 -m venv backprop
source backprop/bin/activate #linux

./backprop/Scripts/activate #windows

python3 -m pip install --upgrade pip
pip install -r requirments.txt
```

### Usage

```python
python deep_learning.py -h

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           dataset dir
  --lr LR               learning rate value
  --iterations ITERATIONS
                        training iterations
  --img IMG             a test image
  --label LABEL         img label if given
  --default_data        use default data for testing...

python deep_learning.py --iterations 1500 --lr 0.0075 --default_data
```
This code will run the training and print cost vs iteration table, like the one below
![alt text](https://github.com/faizan1234567/Assignments/tree/main/ML/assignment5/images/training_progress.png)

And some it will plot the cost as function of iterations
![alt text](https://github.com/faizan1234567/Assignments/tree/main/ML/assignment5/images/Figure_1.png)

### Model prediction
Model has been tested on a batch of imagse from the test set. Which pretty descent considering that we are not using
regularization.
![alt text](https://github.com/faizan1234567/Assignments/tree/main/ML/assignment5/images/Figure_2.png)

