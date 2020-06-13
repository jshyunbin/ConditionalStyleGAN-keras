from absl import app
from absl import flags
from src.cgan import CGAN
from src.acgan import ACGAN

FLAGS = flags.FLAGS

flags.DEFINE_integer('load_model', 49800, 'Epoch num. of the model you wish to open.')
flags.DEFINE_boolean('validate', False, 'Generate images with the latest generator model with given classes.')
flags.DEFINE_integer('size', 64, 'Size of the image the model will generate.')


def main(argv):
    if FLAGS.validate:
        acgan = ACGAN(True)
        acgan.validate(glasses=FLAGS.glasses, male=FLAGS.male)
    else:
        if FLAGS.load_model == 0:
            acgan = ACGAN()
            acgan.train(epochs=50000, batch_size=128, sample_interval=200)
        else:
            acgan = ACGAN(True)
            acgan.train(epochs=50000, batch_size=128, sample_interval=200, start_point=FLAGS.load_model + 1)


if __name__ == '__main__':
    app.run(main)