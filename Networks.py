import argparse
import os
import numpy as np

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


def Net_Encoder(input_img, weight_decay = 0.0001, add_dense=True):

    net = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_1')(input_img)
    net = BatchNormalization()(net)
    net = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_1_2')(net)
    net = BatchNormalization()(net)
    net = MaxPooling2D((2, 2), strides=(2, 2), name='pool_1')(net)
    net = Dropout(0.1)(net)

    net = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_2')(net)
    net = BatchNormalization()(net)
    net = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_2_2')(net)
    net = BatchNormalization()(net)
    net = MaxPooling2D((2, 2), strides=(2, 2), name='pool_2')(net)
    net = Dropout(0.1)(net)

    net = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_3')(net)
    net = BatchNormalization()(net)
    net = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_3_2')(net)
    net = BatchNormalization()(net)
    net = MaxPooling2D((2, 2), strides=(2, 2), name='pool_3')(net)
    net = Dropout(0.1)(net)

    net = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_4')(net)
    net = BatchNormalization()(net)
    net = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='conv_4_2')(net)
    net = BatchNormalization()(net)
    net = MaxPooling2D((2, 2), strides=(2, 2), name='pool_4')(net)
    net = Dropout(0.1)(net)

    if add_dense:
        net = Flatten()(net)
        net = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(weight_decay), name='latent_feats')(net)

    return net

def Net_Decoder(encoder, weight_decay = 0.0001, dense_added=True):

    net = encoder
    if dense_added:
        net = Reshape((2,2,256))(net)

    net = UpSampling2D((2, 2), name='upsample_4')(net)
    net = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_4')(net)
    net = BatchNormalization()(net)
    net = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_4_2')(net)
    net = BatchNormalization()(net)
    net = UpSampling2D((2, 2), name='upsample_3')(net)
    net = Dropout(0.1)(net)

    net = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_3')(net)
    net = BatchNormalization()(net)
    net = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_3_2')(net)
    net = BatchNormalization()(net)
    net = UpSampling2D((2, 2), name='upsample_2')(net)
    net = Dropout(0.1)(net)

    net = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_2')(net)
    net = BatchNormalization()(net)
    net = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_2_2')(net)
    net = BatchNormalization()(net)
    net = UpSampling2D((2, 2), name='upsample_1')(net)
    net = Dropout(0.1)(net)

    net = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_1')(net)
    net = BatchNormalization()(net)
    net = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay), name='upconv_1_2')(net)
    net = BatchNormalization()(net)
    net = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='upconv_final')(net) #sigmoid

    return net
