import argparse
import os

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import regularizers

from Networks import *
from Utils import *

# This script is tested with keras tensorflow 1.9
parser = argparse.ArgumentParser(description='Train models pretrained with AutoEncoders for classification')
parser.add_argument('--train_data', help='full path to training images, expects a folder with sub-folder for each class',
                        default='data/cifar10_keras/train/')
parser.add_argument('--val_data', help='full path to validation images, expects a folder with sub-folder for each class',
                        default='data/cifar10_keras/test/')
parser.add_argument('--save_model_path', help='full path to save the trained model', default='models/cls_cifar10')
parser.add_argument('--base_model_path', help='full path to input autoencoder model (.h5)', default='models/ae_cifar10/ae_cifar10.h5')
parser.add_argument('--num_train_samples', help='number of images in training set', type=int, default=50000) 
parser.add_argument('--num_val_samples', help='number of images in validation set', type=int, default=10000)
parser.add_argument('--img_size', help='input image size', type=int, default=32)
parser.add_argument('--number_of_classes', help='number of classes in the dataset', type=int, default=10)
parser.add_argument('--num_epochs', help='number of epochs for training', type=int, default=100)
parser.add_argument('--batch_size', help='batch size for training', type=int, default=32)
parser.add_argument('--freeze_feature_extractor', action='store_true') # default is false
parser.add_argument('--train_from_scratch', action='store_true') # default is false
parser.add_argument('--extra_dense_layer', action='store_true') # default is false
args = parser.parse_args()

train_data_dir = args.train_data
validation_data_dir = args.val_data
nb_train_samples = args.num_train_samples
nb_validation_samples = args.num_val_samples
out_dir = args.save_model_path
base_model_path = args.base_model_path
freeze = args.freeze_feature_extractor
train_from_scratch = args.train_from_scratch
epochs = args.num_epochs
img_width = img_height = args.img_size
num_classes = args.number_of_classes
batch_size = args.batch_size
add_extra_dense = args.extra_dense_layer

save_model_name = out_dir.split('/')[-1] + ".h5"
save_model_path = os.path.join(out_dir, save_model_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if train_from_scratch:
    print("Training the classification model from scratch ...")
    input_img = Input(shape=(img_width, img_height, 3))
    net = Net_Encoder(input_img)
    if add_extra_dense:
        net = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='cls_dense')(net)
    predictions = Dense(num_classes, activation='softmax')(net)
    model = Model(inputs=input_img, outputs=predictions)

else:  
    orig_model = load_model(base_model_path) # create the base pre-trained model
    base_model = Model(inputs=orig_model.input, outputs=orig_model.get_layer('latent_feats').output)
    print('Pretrained model loaded ....')
    net = base_model.output
    if add_extra_dense:
        x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='cls_dense')(x)
    predictions = Dense(num_classes, activation='softmax')(net)
    model = Model(inputs=base_model.input, outputs=predictions)
    if freeze:
        # train only the top layers (which were randomly initialized), i.e. freeze all convolutional layers
        print("all feature extractor freezed ... ")
        for layer in base_model.layers:
            layer.trainable = False

print(model.summary())

optimizer = SGD(lr=0.001, momentum=0.9, decay=1e-6) 
#optimizer = RMSprop(lr=0.0008, decay=1e-6)
model.compile(optimizer=optimizer,
			loss='categorical_crossentropy',
			metrics=['accuracy'])

# this is the augmentation we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1)
train_datagen.mean = GetCifar10Mean()
train_datagen.std = GetCifar10STD() 

test_datagen = ImageDataGenerator(rescale=1. / 255,
    featurewise_center=True,
    featurewise_std_normalization=True) 
test_datagen.mean = GetCifar10Mean()
test_datagen.std = GetCifar10STD()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    shuffle=True,
    workers=8,
    verbose=1,
    callbacks=[LearningRateScheduler(lr_schedule)]) 

model.save(save_model_path)

PlotTrainValAccuracy(history, out_dir, epochs)
PlotTrainValLoss(history, out_dir, epochs)