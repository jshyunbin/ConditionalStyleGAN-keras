from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
from PIL import Image
import PIL
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
import io
import numpy as np
import utils

class ACGAN():
    def __init__(self):
        super().__init__()
        self.flags = flags
        self.img_rows = self.flags.size
        self.img_cols = self.flags.size
        self.img_channels = 3
        self.img_size = (self.img_rows, self.img_cols, self.img_channels)

        self.latent_dim = 100
        self.label_dim = 10

        log_path = '../logs/imgnet/cgan'
        self.writer = tf.summary.FileWriter(log_path)

        dis_opt = Adam(0.0002, 0.5)
        gan_opt = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'categorical_crossentropy']

        if self.flags.load_model:
            # fill this
            self.generator = self.build_discriminator()
        else: 
            self.generator = self.build_generator()
            self.discriminator = self.build_discriminator()
    
    def build_generator(self):
        
        model = Sequential()

        model.add(Dense(768, activation="relu", input_dim=self.latent_dim + self.label_dim))
        model.add(Conv2DTranspose(768, 5, strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(384, 5, strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(256, 5, strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(192, 5, strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(self.img_channels, 5, strides=(2, 2), padding='same'))
        model.add(Activation('tanh'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.label_dim,))
        
        model_input = concatenate([noise, label], axis=1)
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D())


