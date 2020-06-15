from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam
from functools import partial
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
from layers import AdaInstanceNormalization
import keras.backend as K
import layers

def gradient_penalty_loss(y_true, y_pred, averaged_samples, weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    
    # weight * ||grad||^2
    # Penalize the gradient norm
    return K.mean(gradient_penalty * weight)

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

        dis_opt = Adam(0.0001, beta_1=0.5, beta_2=0.99, decay=0.00001)
        gan_opt = Adam(0.0001, beta_1=0.5, beta_2=0.99, decay=0.00001)
        losses = ['mse', 'mse']

        if self.flags.load_model != -1:
            print('Loading CSGAN model...')
            print('Using epoch %d model' % self.flags.load_model)
            json_file_gen = open('../saved_model/csgan/generator.json', 'r')
            json_file_dis = open('../saved_model/csgan/discriminator.json', 'r')
            generator_json = json_file_gen.read()
            self.generator = model_from_json(generator_json)
            self.generator.load_weights('../saved_model/csgan/generator_%dweights.hdf5' % self.flags.load_model)
            discriminator_json = json_file_dis.read()
            self.discriminator = model_from_json(discriminator_json)
            self.discriminator.load_weights('../saved_model/csgan/discriminator_%dweights.hdf5' % self.flags.load_model)
        else:
            self.discriminator = self.build_discriminator()
            self.generator = self.build_generator()

        self.DM = None
        self.GM = None
        self.build_disModel()
        self.build_genModel()

        # self.discriminator.compile(loss=losses,
        #     optimizer=dis_opt,
        #     metrics=['accuracy'])

        # # The generator takes noise and the target label as input
        # # and generates the corresponding digit of that label
        # noise = Input(shape=(self.latent_dim,))
        # label = Input(shape=(self.num_classes,))
        # img = self.generator([noise, label])

        # # For the combined model we will only train the generator
        # for layer in self.discriminator.layers:
        #     layer.trainable = False

        # # The discriminator takes generated image as input and determines validity
        # # and the label of that image
        # valid, target_label = self.discriminator(img)

        # # The combined model  (stacked generator and discriminator)
        # # Trains the generator to fool the discriminator
        # self.combined = Model([noise, label], [valid, target_label])
        # self.combined.compile(loss=losses, optimizer=gan_opt)

    def g_block(self, inp, style, fil, u = True):
        
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
        x = self.g_block(x, sty, 512)
        x = self.g_block(x, sty, 256)
        x = self.g_block(x, sty, 128)
        x = self.g_block(x, sty, 64)
        x = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(x)

        return Model([inp, inp_s], x)
    
    def build_discriminator(self):

        inp = Input([self.img_rows, self.img_rows, self.channels])

        model = Sequential()
        model = layers.d_block(model, 32, init=True)
        model = layers.d_block(model, 64)
        model = layers.d_block(model, 128)
        model = layers.d_block(model, 256)

        model.summary()

        x = model(inp)
        out = Flatten()(x)
        val = Dense(1, activation='sigmoid')(out)
        label = Dense(self.num_classes, activation='sigmoid')(out)
        
        return Model(inp, [val, label])

    def build_disModel(self):
        self.discriminator.trainable = True
        for layer in self.discriminator.layers:
            layer.trainable = True
        
        self.generator.trainable = False
        for layer in self.generator.layers: 
            layer.trainable = False
        
        #real pipeline
        ri = Input(shape=[self.img_rows, self.img_cols, self.channels])
        dr, drl = self.discriminator(ri)
        #fake pipeline
        gi = Input(shape=[self.latent_dim])
        gi2 = Input(shape=[self.num_classes])
        gf = self.generator([gi, gi2])
        df = self.discriminator(df)

        da, dal = self.discriminator(ri)

        self.DM = Model(inputs=[ri, gi, gi2], outputs=[dr, drl, df, da, dal])

        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=ri, weight=5)

        self.DM.compile(optimizer=Adam(0.0001, beta_1=0, beta_2=0.99, decay=0.00001), loss=['binary_cross_entropy', 'binary_cross_entropy', 'binary_cross_entropy', partial_gp_loss, ])

    def build_genModel(self):
        self.discriminator.trainable = False
        for layer in self.discriminator.layers:
            layer.trainable = False
        
        self.generator.trainable = True
        for layer in self.generator.layers: 
            layer.trainable = True
        
        gi = Input(shape=[self.latent_dim])
        gi2 = Input(shape=[self.num_classes])

        gf = self.generator([gi, gi2])
        df, dfl = self.discriminator(gf)

        self.AM = Model(inputs=[gi, gi2], outputs=[df, dfl])
        self.AM.compile(optimizer=Adam(0.0001, beta_1=0, beta_2=0.99, decay=0.00001), loss=['mse', 'mse'])


    def train(self, epochs, batch_size=32, sample_interval=50, start_point=0):

        # Load the dataset
        X_train, y_train = utils.load_data(self.writer)

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

            # Image labels. 0-9 
            img_labels = y_train[idx]

            d_loss = self.DM.train_on_batch([imgs, noise, sampled_labels], [valid, fake, valid])


            # Generate a half batch of new images
            # gen_imgs = self.generator.predict([noise, sampled_labels])


            # # Train the discriminator
            # d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            # d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.AM.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Train the generator
            # g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
            utils.write_log( self.writer, ['D loss', 'G loss', 'accuracy'], [d_loss[0], g_loss[0], 100*d_loss[3]], epoch)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                utils.save_model('csgan/celeba', self.generator, self.discriminator, epoch)
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
                fig.savefig('../images_condition/validate{}{}.png'.format('_glasses' if glasses else '_male', j))
            return
            
        fig, axs = plt.subplots(4, 8)
        
        for i in range(2**5):
            label_str = "{:05b}".format(i)
            print(label_str)
            label = np.array([[int(label_str[j]) for j in range(len(label_str))] for _ in range(10)])
            imgs = 0.5 * self.generator.predict([noise, label]) + 0.5
            utils.write_image(self.writer, 'Image: {}'.format(label_str), imgs)
            axs[i//(2**3), i%(2**3)].imshow(imgs[0])
            axs[i//(2**3), i%(2**3)].axis('off')
        fig.savefig('../images_condition/validate{}{}.png'.format('_glasses' if glasses else '', '_male' if male else ''))
        plt.close()

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.random.uniform(0, 1, (r*c, self.num_classes))
        sampled_labels = np.around(sampled_labels)
        gen_imgs = self.generator.predict([noise, sampled_labels])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        utils.write_image(self.writer, 'Generated Image', gen_imgs[:10], step=epoch)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("../images/csgan/%d.png" % epoch)
        plt.close()
