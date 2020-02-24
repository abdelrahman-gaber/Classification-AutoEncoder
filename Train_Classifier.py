import argparse
import os

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from Networks import *

# This code is tested with tensorflow 1.9

parser = argparse.ArgumentParser(description='Train models pretrained with AutoEncoders for classification')
parser.add_argument('--train_data', help='full path to training images, expects a folder with sub-folder for each class', \
                        default='data/cifar10_keras/train/')
parser.add_argument('--val_data', help='full path to validation images, expects a folder with sub-folder for each class', \
                        default='data/cifar10_keras/val/')
parser.add_argument('--save_model_path', help='full path to the model to save (.h5)', default='models/classification_cifar10.h5')
parser.add_argument('--base_model_path', help='full path to input autoencoder model (.h5)', default='models/ae_cifar10.h5')
parser.add_argument('--num_train_samples', help='number of images in training set', type=int, default=27500)
parser.add_argument('--num_val_samples', help='number of images in validation set', type=int, default=1000)
parser.add_argument('--num_epochs', help='number of epochs for training', type=int, default=20)
parser.add_argument('--freeze_feature_extractor', action='store_true') # default is false
parser.add_argument('--train_from_scratch', action='store_false') # store_true
args = parser.parse_args()

train_data_dir = args.train_data
validation_data_dir = args.val_data
nb_train_samples = args.num_train_samples
nb_validation_samples = args.num_val_samples
save_model_path = args.save_model_path
base_model_path = args.base_model_path
freeze = args.freeze_feature_extractor
train_from_scratch = args.train_from_scratch
epochs = args.num_epochs

# dimensions of cifar10 images
img_width, img_height = 32, 32
num_classes = 10 # cifar10
batch_size = 32

out_dir = os.path.splitext(save_model_path)[0]
if not os.path.exists(out_dir):
        os.makedirs(out_dir)

if train_from_scratch:
  print("Training the model from scratch")
  input_img = Input(shape=(img_width, img_height, 3))
  #x = EncoderC3(input_img)
  x = Encoder_1(input_img)
  # add a global spatial average pooling layer
  #x = GlobalAveragePooling2D()(x)
  x = Dense(1024, activation='relu')(x)
  predictions = Dense(num_classes, activation='softmax')(x)
  model = Model(inputs=input_img, outputs=predictions)

else:  # load the model pretrained with AutoEncoders
  # create the base pre-trained model
  orig_model = load_model(base_model_path)
  base_model = Model(inputs=orig_model.input, outputs=orig_model.get_layer('latent_feats').output)
  print('Pretrained model loaded ....')
  x = base_model.output
  #add a fully-connected layer
  x = Dense(1024, activation='relu')(x)
  # and a logistic layer -- we have num_classes classes
  predictions = Dense(num_classes, activation='softmax')(x)
  # this is the model we will train
  model = Model(inputs=base_model.input, outputs=predictions)
  if freeze:
    # train only the top layers (which were randomly initialized), i.e. freeze all convolutional layers
    print("all feature extractor freezed ... ")
    for layer in base_model.layers:
        layer.trainable = False

print(model.summary())

# compile the model (should be done after setting layers to non-trainable)
optimizer = SGD(lr=0.0005, momentum=0.9) # 0.0005
model.compile(optimizer=optimizer,
			loss='categorical_crossentropy',
			metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=10,
    brightness_range =[0.7, 1.1])

# no augmentation for validation set:
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# tensorboard
tensorboard = TensorBoard(log_dir=out_dir, histogram_freq=0, 
                          write_graph=False, write_images=True) 

#checkpoints = ModelCheckpoint(out_dir + "/weights-ep-{epoch:02d}-valacc-{val_acc:.4f}.h5",
#                              monitor='val_acc', verbose=0, save_best_only=True,
#                              save_weights_only=False, mode='auto', period=1)

# train the model on the new data for few epochs
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    shuffle=True,
    workers=8,
    callbacks=[tensorboard]) #,checkpoints])

model.save(save_model_path)
