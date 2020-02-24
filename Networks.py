import argparse
import os
import numpy as np

from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Flatten, Reshape
from tensorflow.keras.models import Model

# models 
def AE_CNN_Simple():
    img_width, img_height = 32, 32
    input_img = Input(shape=(img_width, img_height, 3))
    
    # Encoding network
    x = Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)(x)
    encoded = Conv2D(32, (2, 2), activation='relu', padding="same", strides=2)(x)

    # Decoding network
    x = Conv2D(32, (2, 2), activation='relu', padding="same")(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    return Model(input_img, decoded)

def Encoder_1(input_img, no_dense=False):
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1')(input_img)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_4')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_5')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_3')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_6')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool_4')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_7')(x)

    if not no_dense:
        x = Flatten()(x)
        x = Dense(1024, activation='relu', name='latent_feats')(x)

    return x

def Decoder_1(encoder):
    x = Conv2D(128, (3, 3), activation='relu', padding="same", name='upconv_7')(x)
    x = UpSampling2D((2, 2), name='upsample_4')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding="same", name='upconv_6')(x)
    x = UpSampling2D((2, 2), name='upsample_3')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding="same", name='upconv_5')(x)
    x = UpSampling2D((2, 2), name='upsample_2')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding="same", name='upconv_4')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding="same", name='upconv_3')(x)
    x = UpSampling2D((2, 2), name='upsample_1')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding="same", name='upconv_2')(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding="same", name='upconv_final')(x)

    return x

def EncoderC3(input_img, no_dense=False):
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1')(input_img)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1_2')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', strides=2, name='conv_1_s')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2_2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, name='conv_2_s')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3_2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', strides=2, name='conv_3_s')(x)

    x = Conv2D(192, (3, 3), activation='relu', padding='same', name='conv_4')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', name='conv_4_2')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', strides=2, name='conv_4_last')(x)

    if not no_dense:
        x = Flatten()(x)
        x = Dense(1024, activation='relu', name='latent_feats')(x)

    return x

def DecoderC3(encoder, no_dense=False):
    if not no_dense:
        x = Dense(2*2*192, activation='relu')(encoder)
        x = Reshape((2,2,192))(x)
    else:
        x = encoder

    x = Conv2D(192, (3, 3), activation='relu', padding='same', name='upconv_4')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same', name='upconv_4_2')(x)
    x = UpSampling2D((2, 2), name='upsample_4')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='upconv_3')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='upconv_3_2')(x)
    x = UpSampling2D((2, 2), name='upsample_3')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='upconv_2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='upconv_2_2')(x)
    x = UpSampling2D((2, 2), name='upsample_2')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='upconv_1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='upconv_1_2')(x)
    x = UpSampling2D((2, 2), name='upsample_1')(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='upconv_final')(x) #sigmoid

    return x
