# Classification-AutoEncoder

The aim of this project is to train autoencoder, and use its trained weights as initialization to improve classification accuracy with cifar10 dataset. This is a kind of [transfer learning](http://cs231n.github.io/transfer-learning/) where we can have pretrained models using the unsupervised learning approach of auto-encoders. 

## How to run

Download and prepare cifar10 dataset. Just run the following script: 
```Shell
  cd data/
  ./gen_cifar10.sh
  ```
  This script will download cifar10 dataset, then split it to folders as needed for Keras `flow_from_directory()` function. The Autoencoder and Classification scripts here can work with any dataset, just remember to set input image size to match your data. 
  
### Train Auto-encoder network 
To train the autoencoder, just run `python Train_Autoencoder.py` <br />

The autoencoder network here is trained to be initialization for the classification network. If you want to get better output images, consider removing the fully connected layer. 

![Alt text](models/ae_cifar10/ae_trained_ep50_1.jpg)

### Train Classification network

To start training the classifier, run `python Train_Classifier.py`

### Classification model evaluation

Finally, to evaluate the trained classification model, run `python Evaluate_Classifier_cifar10.py` <br />

The evaluation script here is done for cifar10 dataset, but it can be easily modified to work with any dataset you have. <br /> 


## Requirements

1. Python 3
2. Keras with tensorflow version 1.9, It is expected to work with newer version of tensorflow 1.xx
