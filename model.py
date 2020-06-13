from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
from PIL import Image
from absl import app
from absl import flags
import PIL
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
import io
import numpy as np

class ACGAN():
    def __init__(self, load_model=False):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 5
        self.latent_dim = 100
        log_path = './logs'
        self.writer = tf.summary.FileWriter(log_path)

        

        dis_opt = Adam(0.0002, 0.5)
        gan_opt = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'binary_crossentropy']

        # Build and compile the discriminator
        if load_model is True:
            print('Loading ACGAN model...')
            print('Using epoch %d model' % FLAGS.load_model)
            json_file_gen = open('saved_model/generator.json', 'r')
            json_file_dis = open('saved_model/discriminator.json', 'r')
            generator_json = json_file_gen.read()
            self.generator = model_from_json(generator_json)
            self.generator.load_weights('saved_model/generator_%dweights.hdf5' % FLAGS.load_model)
            discriminator_json = json_file_dis.read()
            self.discriminator = model_from_json(discriminator_json)
            self.discriminator.load_weights('saved_model/discriminator_%dweights.hdf5' % FLAGS.load_model)
        else:
            self.discriminator = self.build_discriminator()
            self.generator = self.build_generator()
        
        self.discriminator.compile(loss=losses,
            optimizer=dis_opt,
            metrics=['accuracy'])


        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses, optimizer=gan_opt)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(512 * 4 * 4, activation="relu", input_dim=self.latent_dim+self.num_classes))
        model.add(Reshape((4, 4, 512)))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(256, 4, strides=(2, 2), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(128, 4, strides=(2, 2), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(64, 4, strides=(2, 2), padding='same'))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2DTranspose(self.channels, 4, strides=(2, 2), padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.num_classes,))
        
        model_input = concatenate([noise, label], axis=1)
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="sigmoid")(features)

        return Model(img, [validity, label])

    def load_data(self):
        print("Loading celebA dataset ...")

        dataset_file = 'dataset/celeba.hdf5'

        if os.path.isfile(dataset_file):
            print('Loading dataset from saved file')
            f = h5py.File(dataset_file, 'r')
            trainX, trainy = f['image'].value, f['label'].value
            f.close()
            self.write_image('Dataset', trainX[:10] * 0.5 + 0.5)
            return (trainX, trainy)


        train_img_file = sorted([os.path.join('dataset/train', fname) for fname in os.listdir('dataset/train')])
        # loading labels
        label_file = open('dataset/list_attr_celeba_full.txt', 'r').readlines()

        label = [[0. if i == '-1' else 1. for i in x.split()[1:]] for x in label_file[2:]]

        label_filtered = [[label[i][36], label[i][15], label[i][20], label[i][22], label[i][31]] for i in range(len(label))]

        trainy = [x for x in label_filtered[:len(train_img_file)]]

        trainX = []


        for path in train_img_file:
            img = Image.open(path)
            img = np.asarray(img)
            
            h, w = img.shape[:2]
            h_start = int(round((h - 108) / 2.))
            w_start = int(round((w - 108) / 2.))
            # resize image
            image = Image.fromarray(img[h_start:h_start + 108, w_start:w_start + 108])
            img_crop = np.array(image.resize((64, 64), PIL.Image.BICUBIC))

            trainX.append(img_crop)

        # make numpy arrays
        trainX = np.array(trainX)
        trainX = (trainX - 127.5) / 127.5
        trainy = np.array(trainy)
        
        hf = h5py.File(dataset_file, 'w')
        hf.create_dataset('image', data=trainX)
        hf.create_dataset('label', data=trainy)
        hf.close()

        print('Saved dataset for faster load')

        return (trainX, trainy)


    def train(self, epochs, batch_size=32, sample_interval=50, start_point=0):

        # Load the dataset
        X_train, y_train = self.load_data()

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(start_point, epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.uniform(0, 1, (batch_size, self.num_classes))
            sampled_labels = np.around(sampled_labels)

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels. 0-9 
            img_labels = y_train[idx]

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print("%d [LC loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [LS loss: %f]" % (epoch, (d_loss[0] + g_loss[0])/2, 100*d_loss[3], 100*d_loss[4], (d_loss[0] - g_loss[0])/2))
            self.write_log( ['LC loss', 'LS loss', 'accuracy'], [(d_loss[0] + g_loss[0])/2, (d_loss[0] - g_loss[0])/2, 100*d_loss[3]], epoch)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_model(epoch)
                self.sample_images(epoch)

    def validate(self, glasses=False, male=False):
        noise = np.random.normal(0, 1, (10, self.latent_dim))
        
        if glasses or male:
            for j in range(10):
                noise = np.random.normal(0, 1, (10, self.latent_dim))
                fig, axs = plt.subplots(2, 10)
                label = np.array([[0, 0, 0, 0, 0] for _ in range(10)])
                img_default = 0.5 * self.generator.predict([noise, label]) + 0.5
                for i in range(10):
                    axs[0, i].imshow(img_default[i])
                    axs[0, i].axis('off')
                if glasses:
                    label = np.array([[0, 1, 0, 0, 0] for _ in range(10)])
                    img_condition = 0.5 * self.generator.predict([noise, label]) + 0.5
                elif male:
                    label = np.array([[0, 0, 1, 0, 0] for _ in range(10)])
                    img_condition = 0.5 * self.generator.predict([noise, label]) + 0.5
                for i in range(10):
                    axs[1, i].imshow(img_condition[i])
                    axs[1, i].axis('off')
                fig.savefig('images_condition/validate{}{}.png'.format('_glasses' if glasses else '_male', j))
            return
            
        fig, axs = plt.subplots(4, 8)
        
        for i in range(2**5):
            label_str = "{:05b}".format(i)
            print(label_str)
            label = np.array([[int(label_str[j]) for j in range(len(label_str))] for _ in range(10)])
            imgs = 0.5 * self.generator.predict([noise, label]) + 0.5
            self.write_image('Image: {}'.format(label_str), imgs)
            axs[i//(2**3), i%(2**3)].imshow(imgs[0])
            axs[i//(2**3), i%(2**3)].axis('off')
        fig.savefig('images_condition/validate{}{}.png'.format('_glasses' if glasses else '', '_male' if male else ''))
        plt.close()

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.random.uniform(0, 1, (r*c, self.num_classes))
        sampled_labels = np.around(sampled_labels)
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        self.write_image('Generated Image', gen_imgs[:10], step=epoch)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self, step):

        def save(model, model_name, step):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_%dweights.hdf5" % (model_name, step)
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator", step)
        save(self.discriminator, "discriminator", step)

    def write_log(self, names, logs, step):
        for name, value in zip(names, logs):
            summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
            self.writer.add_summary(summary, step)

    def write_image(self, name, images, step=None):
        for i in range(images.shape[0]):
            output = io.BytesIO()
            image = Image.fromarray((images[i] * 255).astype('uint8'))
            image.save(output, 'PNG')
            image_string = output.getvalue()
            output.close()
            summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=tf.Summary.Image(height=64, width=64, colorspace=3, encoded_image_string=image_string))])
            self.writer.add_summary(summary, step)

FLAGS = flags.FLAGS

flags.DEFINE_integer('load_model', 49800, 'Epoch num. of the model you wish to open.')
flags.DEFINE_boolean('validate', False, 'Generate images with the latest generator model with given classes.')
flags.DEFINE_boolean('glasses', False, 'Generate only glasses images when validate')
flags.DEFINE_boolean('male', False, 'Generate only male images when validate')


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