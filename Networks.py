import argparse
import os
import numpy as np

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def Net1_bn(input_img, weight_decay = 0.0001):

    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_1')(input_img)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_1_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_1')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_2_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_2')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_3')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_3_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_3')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_4')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_4_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_4')(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='latent_feats')(x)

    return x

def Net1_Decoder(encoder, weight_decay = 0.0001):

    x = encoder
    x = Reshape((2,2,256))(x)

    x = UpSampling2D((2, 2), name='upsample_4')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_4')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_4_2')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2), name='upsample_3')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_3')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_3_2')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2), name='upsample_2')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_2_2')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2), name='upsample_1')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_1_2')(x)
    x = BatchNormalization()(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='upconv_final')(x) #sigmoid

    return x
