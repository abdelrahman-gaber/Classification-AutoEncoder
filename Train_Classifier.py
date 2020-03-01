import argparse
import os
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import regularizers

from Networks import *

# This code is tested with keras tensorflow 1.9

def lr_schedule(epoch):
    lrate = 0.0005
    if epoch > 35:
        lrate = 0.0001
    elif epoch > 44:
        lrate = 0.00005        
    return lrate

parser = argparse.ArgumentParser(description='Train models pretrained with AutoEncoders for classification')
parser.add_argument('--train_data', help='full path to training images, expects a folder with sub-folder for each class', \
                        default='data/cifar10_keras/train/')
parser.add_argument('--val_data', help='full path to validation images, expects a folder with sub-folder for each class', \
                        default='data/cifar10_keras/test/')
parser.add_argument('--save_model_path', help='full path to save the trained model', default='models/cls_cifar10')
parser.add_argument('--base_model_path', help='full path to input autoencoder model (.h5)', default='models/ae_cifar10/ae_cifar10.h5')
parser.add_argument('--num_train_samples', help='number of images in training set', type=int, default=27500) #27500
parser.add_argument('--num_val_samples', help='number of images in validation set', type=int, default=1000) #1000
parser.add_argument('--num_epochs', help='number of epochs for training', type=int, default=50)
parser.add_argument('--freeze_feature_extractor', action='store_true') # default is false
parser.add_argument('--train_from_scratch', action='store_true') # default is false
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

# dimensions of cifar10 images
img_width, img_height = 32, 32
num_classes = 10 # cifar10
batch_size = 32

save_model_name = out_dir.split('/')[-1] + ".h5"
save_model_path = os.path.join(out_dir, save_model_name)
if not os.path.exists(out_dir):
        os.makedirs(out_dir)

if train_from_scratch:
  print("Training the model from scratch ...")
  input_img = Input(shape=(img_width, img_height, 3))
  x = Net1_bn(input_img)
  x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='cls_dense')(x)
  predictions = Dense(num_classes, activation='softmax')(x)
  model = Model(inputs=input_img, outputs=predictions)

else:  # load the model pretrained with AutoEncoders
  orig_model = load_model(base_model_path) # create the base pre-trained model
  base_model = Model(inputs=orig_model.input, outputs=orig_model.get_layer('latent_feats').output)
  print('Pretrained model loaded ....')
  x = base_model.output
  #add a fully-connected layer
  x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='cls_dense')(x)
  #x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='cls_dense_2')(x)
  predictions = Dense(num_classes, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
  if freeze:
    # train only the top layers (which were randomly initialized), i.e. freeze all convolutional layers
    print("all feature extractor freezed ... ")
    for layer in base_model.layers:
        layer.trainable = False

print(model.summary())

# compile the model (should be done after setting layers to non-trainable)
optimizer = SGD(lr=0.0005, momentum=0.9) # 0.001
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
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range =[0.8, 1.1])

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
#tensorboard = TensorBoard(log_dir=out_dir, histogram_freq=0, 
#                          write_graph=False, write_images=True) 

#checkpoints = ModelCheckpoint(out_dir + "/weights-ep-{epoch:02d}-valacc-{val_acc:.4f}.h5",
#                              monitor='val_acc', verbose=0, save_best_only=True,
#                              save_weights_only=False, mode='auto', period=1)

# train the model on the new data for few epochs
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    shuffle=True,
    workers=8,
    verbose=1,
    callbacks=[LearningRateScheduler(lr_schedule)]) #, tensorboard, checkpoints])

model.save(save_model_path)


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(out_dir + "/" + "train_val_accuracy_ep%d.jpg"%(epochs),  bbox_inches='tight', dpi=300)
#plt.show()
plt.close()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(out_dir + "/" + "train_val_loss_ep%d.jpg"%(epochs),  bbox_inches='tight', dpi=300)
#plt.show()
