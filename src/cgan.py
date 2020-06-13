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

class CGAN():
    def __init__(self, flags):
        super().__init__()
        self.flags = flags
        self.img_rows = self.flags.size
        self.img_cols = self.flags.size
        self.img_channels = 3
        self.img_size = (self.img_rows, self.img_cols, self.img_channels)

        self.latent_dim = 100

        log_path = '../logs/imgnet/cgan'
        self.writer = tf.summary.FileWriter(log_path)

        dis_opt = Adam(0.0002, 0.5)
        gan_opt = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'categorical_crossentropy']