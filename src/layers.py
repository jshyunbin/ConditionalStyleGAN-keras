from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate, Layer, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras import backend as K
import numpy as np

# code copied from manicman1999/StyleGAN-Keras directory of GitHub.
class AdaInstanceNormalization(Layer):
    def __init__(self, 
             axis=-1,
             momentum=0.99,
             epsilon=1e-3,
             center=True,
             scale=True,
             **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
    
    
    def build(self, input_shape):
    
        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')
    
        super(AdaInstanceNormalization, self).build(input_shape) 
    
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))
        
        beta = inputs[1]
        gamma = inputs[2]

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta
    
    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
    
        return input_shape[0]

def d_block(inp, fil, p = True, init=False):
    if init:
        inp.add(Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', input_shape=np.prod((64, 64, 3))))
    else:
        inp.add(Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal'))
    inp.add(LeakyReLU(0.01))

    if p:
        inp.add(AveragePooling2D())
    inp.add(Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal'))
    inp.add(LeakyReLU(0.01))
    
    return inp

