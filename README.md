# Classification-AutoEncoder

The aim of this project is to train autoencoder, and use the trained weights as initialization to improve classification accuracy with cifar10 dataset. 

### How to run
1. To download and prepare cifar10 dataset, just run the following script from inside `data/` folder 

```Shell
  cd data/
  ./gen_cifar10.sh
  ```
  
2. To start training the autoencoder, run `python Autoencoder.py`

3. To start training the classifier, run `python Train_Classifier.py`

4. Finally, to evaluate the trained classification model, run `python Classify_Evaluate.py`


### Requirements

1. Python 3.6
2. Keras with tensorflow version 1.9, It is expected to work with newer version of tensorflow 1.xx
