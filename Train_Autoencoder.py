import argparse
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import Callback, LearningRateScheduler

from Networks import *
from Utils import *

# This script is tested with keras tensorflow 1.9

# generate same image patch for input and output
def fixed_generator(generator):
    for batch in generator:
        yield (batch, batch)

parser = argparse.ArgumentParser(description='Train Autoencoder network')
parser.add_argument('--train_data', help='full path to training images, expects a folder with sub-folder for each class', \
         default='data/cifar10_keras/train/')
parser.add_argument('--val_data', help='full path to validation images, expects a folder with sub-folder for each class', \
         default='data/cifar10_keras/test/')
parser.add_argument('--save_model_path', help='full path to save the trained model', default='models/ae_cifar10')
parser.add_argument('--img_size', help='input image size', type=int, default=32)
parser.add_argument('--num_train_samples', help='number of images in training set', type=int, default=50000)
parser.add_argument('--num_val_samples', help='number of images in validation set', type=int, default=10000)
parser.add_argument('--batch_size', help='batch size for training', type=int, default=32)
parser.add_argument('--num_epochs', help='number of epochs for training', type=int, default=50)
args = parser.parse_args()

img_width = img_height = args.img_size 
batch_size = args.batch_size
train_data_dir = args.train_data
validation_data_dir = args.val_data
epochs = args.num_epochs 
nb_train_samples = args.num_train_samples
nb_validation_samples = args.num_val_samples
out_dir = args.save_model_path

save_model_name = out_dir.split('/')[-1] + ".h5"
save_model_path = os.path.join(out_dir, save_model_name)
out_dir_imgs = os.path.join(out_dir, "imgs")
if not os.path.exists(out_dir_imgs):
        os.makedirs(out_dir_imgs) 

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True, 
    vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode=None, shuffle=True)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode=None, shuffle=True)


input_img = Input(shape=(img_width, img_height, 3))
encoded = Net_Encoder(input_img)
decoded = Net_Decoder(encoded) 
autoencoder_cnn = Model(input_img, decoded)
print(autoencoder_cnn.summary())

optimizer = Adam(lr=0.00015)
autoencoder_cnn.compile(optimizer=optimizer, loss='binary_crossentropy') 

# Save autoencoder input and output each epoch
save_out_imgs = SaveOutputImages(validation_generator, out_dir_imgs)

history = autoencoder_cnn.fit_generator(
    fixed_generator(train_generator),
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=fixed_generator(validation_generator),
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[save_out_imgs, LearningRateScheduler(lr_schedule_ae)] )

autoencoder_cnn.save(save_model_path)


PlotTrainValLoss(history, out_dir, epochs)

# Test random images and visualize the results
x_test = validation_generator.next()
decoded_imgs = autoencoder_cnn.predict(x_test)
VisualizeAE(x_test, decoded_imgs , out_dir, epochs)
