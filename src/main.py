from __future__ import print_function, division

from PIL import Image
from absl import app
from absl import flags
import PIL
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import io
import numpy as np
from src.acgan import ACGAN

FLAGS = flags.FLAGS

flags.DEFINE_integer('load_model', 49800, 'Epoch num. of the model you wish to open.')
flags.DEFINE_boolean('validate', False, 'Generate images with the latest generator model with given classes.')
flags.DEFINE_boolean('glasses', False, 'Generate only glasses images when validate')
flags.DEFINE_boolean('male', False, 'Generate only male images when validate')


def main(argv):
    if FLAGS.validate:
        acgan = ACGAN(FLAGS, True)
        acgan.validate(glasses=FLAGS.glasses, male=FLAGS.male)
    else:
        if FLAGS.load_model == 0:
            acgan = ACGAN(FLAGS)
            acgan.train(epochs=50000, batch_size=128, sample_interval=200)
        else:
            acgan = ACGAN(FLAGS, True)
            acgan.train(epochs=50000, batch_size=128, sample_interval=200, start_point=FLAGS.load_model + 1)


if __name__ == '__main__':
    app.run(main)