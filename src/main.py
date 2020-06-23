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
from acgan import ACGAN
from csgan import CSGAN
from cgan import CGAN

FLAGS = flags.FLAGS

flags.DEFINE_integer('load_model', -1, 'Epoch num. of the model you wish to open.')
flags.DEFINE_boolean('validate', False, 'Generate images with the latest generator model with given classes.')
flags.DEFINE_boolean('glasses', False, 'Generate only glasses images when validate')
flags.DEFINE_boolean('male', False, 'Generate only male images when validate')
flags.DEFINE_enum('model', 'CSGAN', ['CSGAN', 'CGAN', 'ACGAN'], 'Choose the model you want to train')
flags.DEFINE_integer('gpu', 0, 'The gpu number you use for training')
flags.DEFINE_string('name', None, 'the directory name for saving tensorboard data')
flags.DEFINE_integer('batch_size', 64, 'Batch size when training')


def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % FLAGS.gpu
    if FLAGS.model == 'CSGAN':
        model = CSGAN(FLAGS)
    elif FLAGS.model == 'ACGAN':
        model = ACGAN(FLAGS)
    elif FLAGS.model == 'CGAN':
        model = CGAN(FLAGS)

    if FLAGS.validate:
        model.validate(glasses=FLAGS.glasses, male=FLAGS.male)
    else:
        model.train(epochs=1000000, batch_size=FLAGS.batch_size, sample_interval=200, start_point=FLAGS.load_model+1)


if __name__ == '__main__':
    app.run(main)