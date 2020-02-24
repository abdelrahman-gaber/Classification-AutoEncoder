import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback

from Networks import *

class SaveOutputImages(Callback):
    def __init__(self, validation_generator, save_dir):
        self.x_test = validation_generator.next()
        self.out_dir = save_dir
        self.count = 0

    def on_epoch_end(self, batch, logs={}):
        self.count+=1
        decoded_imgs = self.model.predict(self.x_test)

        n = 10
        plt.figure(figsize=(n*5, 2*5))
        #plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i+1)
            plt.imshow(self.x_test[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n+1)
            plt.imshow(decoded_imgs[i])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(self.out_dir + "/" + "ae_ep%d.jpg"%(self.count),  bbox_inches='tight', dpi=300)
        plt.close('all')

# generate same image patch for input and output
def fixed_generator(generator):
    for batch in generator:
        yield (batch, batch)

parser = argparse.ArgumentParser(description='Train Autoencoder network')
parser.add_argument('--train_data', help='full path to training images, expects a folder with sub-folder for each class', \
         default='data/cifar10_keras/train/')
parser.add_argument('--val_data', help='full path to validation images, expects a folder with sub-folder for each class', \
         default='data/cifar10_keras/val/')
parser.add_argument('--save_model_path', help='full path to the model to save (.h5)', default='models/ae_cifar10.h5')
parser.add_argument('--num_train_samples', help='number of images in training set', type=int, default=27500)
parser.add_argument('--num_val_samples', help='number of images in validation set', type=int, default=1000)
parser.add_argument('--num_epochs', help='number of epochs for training', type=int, default=50)
args = parser.parse_args()

img_width, img_height = 32, 32  # image size for cifar10
batch_size = 32
train_data_dir = args.train_data
validation_data_dir = args.val_data
nb_epoch = args.num_epochs 
nb_train_samples = args.num_train_samples
nb_validation_samples = args.num_val_samples

# to save the output images while training
out_dir = os.path.splitext(args.save_model_path)[0]
if not os.path.exists(out_dir):
        os.makedirs(out_dir)

train_datagen_cnn = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        brightness_range =[0.9, 1.1],
        horizontal_flip=True, 
        vertical_flip=True)

test_datagen_cnn = ImageDataGenerator(rescale=1./255)

train_generator_cnn = train_datagen_cnn.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None, shuffle=True)

validation_generator_cnn = test_datagen_cnn.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None, shuffle=True)

input_img = Input(shape=(img_width, img_height, 3))
encoded = EncoderC3(input_img)
decoded = DecoderC3(encoded)
autoencoder_cnn = Model(input_img, decoded)

#autoencoder_cnn = AE_CNN_Simple()
print(autoencoder_cnn.summary())

optimizer = Adam(lr=0.0001)
autoencoder_cnn.compile(optimizer=optimizer, loss='binary_crossentropy')

# Save autoencoder input and output each epoch
save_out_imgs = SaveOutputImages(validation_generator_cnn, out_dir)
# tensorboard
tensorboard = TensorBoard(log_dir=out_dir, histogram_freq=0, 
                          write_graph=False, write_images=True)

autoencoder_cnn.fit_generator(
    fixed_generator(train_generator_cnn),
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=nb_epoch,
    validation_data=fixed_generator(validation_generator_cnn),
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[tensorboard, save_out_imgs] )

autoencoder_cnn.save(args.save_model_path)

# Test random n images and visualize the results
x_test = validation_generator_cnn.next()
decoded_imgs = autoencoder_cnn.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig(out_dir + "/ae_trained_ep%d.jpg"%(nb_epoch), dpi=100)
plt.show()
