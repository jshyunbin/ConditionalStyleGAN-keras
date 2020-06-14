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
from src.layers import AdaInstanceNormalization

class CSGAN():
    def __init__(self, flags):
        self.flags = flags
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 5
        self.latent_dim = 100
        log_path = '../logs/csgan'
        self.writer = tf.summary.FileWriter(log_path)

        dis_opt = Adam(0.0002, 0.5)
        gan_opt = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'binary_crossentropy']

        if load_model is True:
            print('Loading CSGAN model...')
            print('Using epoch %d model' % self.flags.load_model)
            json_file_gen = open('../saved_model/generator.json', 'r')
            json_file_dis = open('../saved_model/discriminator.json', 'r')
            generator_json = json_file_gen.read()
            self.generator = model_from_json(generator_json)
            self.generator.load_weights('../saved_model/generator_%dweights.hdf5' % self.flags.load_model)
            discriminator_json = json_file_dis.read()
            self.discriminator = model_from_json(discriminator_json)
            self.discriminator.load_weights('../saved_model/discriminator_%dweights.hdf5' % self.flags.load_model)
        else:
            self.discriminator = self.build_discriminator()
            self.generator = self.build_generator()

        self.discriminator.compile(loss=losses,
            optimizer=dis_opt,
            metrics=['accuracy'])

    def g_block(inp, style, fil, u = True):
        
        b = Dense(fil)(style)
        b = Reshape([1, 1, fil])(b)
        g = Dense(fil)(style)
        g = Reshape([1, 1, fil])(g)
        
        if u:
            out = UpSampling2D(interpolation = 'bilinear')(inp)
            out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
        else:
            out = Activation('linear')(inp)
        
        out = AdaInstanceNormalization()([out, b, g])
        out = LeakyReLU(0.01)(out)
        
        b = Dense(fil)(style)
        b = Reshape([1, 1, fil])(b)
        g = Dense(fil)(style)
        g = Reshape([1, 1, fil])(g)

        out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
        out = AdaInstanceNormalization()([out, b, g])
        out = LeakyReLU(0.01)(out)
        
        return out

    def build_generator(self):

        inp_s = Input(shape=[self.num_classes])
        sty = Dense(512, kernel_initializer='he_normal')(inp_s)
        sty = LeakyReLU(0.1)(sty)
        sty = Dense(512, kernel_initializer='he_normal')(sty)
        sty = LeakyReLU(0.1)(sty)

        inp = Input(shape=[self.latent_dim])
        x = Dense(4 * 4 * 512, kernel_initializer='he_normal')(inp)
        x = Reshape([4, 4, 512])(x)
        x = g_block(x, sty, 512)
        x = g_block(x, sty, 128)
        x = g_block(x, sty, 64)
        x = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(x)

        return x
    
    def build_discriminator(self):
        
