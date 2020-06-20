from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Cropping2D, add
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

def gradient_penalty_loss(y_true, y_pred, sample_weight, averaged_samples, weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradient_penalty = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
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
        self.latent_dim = 1

        if self.flags.name is None:
            log_path = '../logs/csgan'
        else: 
            log_path = '../logs/' + self.flags.name
        self.writer = tf.summary.FileWriter(log_path)

        if self.flags.name is None:
            model_path = '../saved_model/csgan'
        else: 
            model_path = '../saved_model/' + self.flags.name

        self.images_path = '../images/%s' % ('csgan' if self.flags.name is None else self.flags.name)

        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        if not os.path.isdir(self.images_path):
            os.mkdir(self.images_path)

        if self.flags.load_model != -1:
            print('Loading CSGAN model...')
            print('Using epoch %d model' % self.flags.load_model)
            json_file_gen = open(model_path + '/generator.json', 'r')
            json_file_dis = open(model_path + '/discriminator.json', 'r')
            generator_json = json_file_gen.read()
            self.generator = model_from_json(generator_json, custom_objects={'AdaInstanceNormalization':AdaInstanceNormalization})
            self.generator.load_weights(model_path + '/generator_%dweights.hdf5'% self.flags.load_model)
            discriminator_json = json_file_dis.read()
            self.discriminator = model_from_json(discriminator_json)
            self.discriminator.load_weights(model_path + '/discriminator_%dweights.hdf5' % self.flags.load_model)
        else:
            self.discriminator = self.build_discriminator()
            self.generator = self.build_generator()

        self.DM = None
        self.GM = None
        self.build_disModel()
        self.build_genModel()

        print(self.DM.metrics_names)

    def g_block(self, inp, style, noise, fil, u = True):

        b = Dense(fil)(style)
        b = Reshape([1, 1, fil])(b)
        g = Dense(fil)(style)
        g = Reshape([1, 1, fil])(g)

        n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
        
        if u:
            out = UpSampling2D(interpolation = 'bilinear')(inp)
            out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
        else:
            out = Activation('linear')(inp)
        
        out = AdaInstanceNormalization()([out, b, g])
        out = add([out, n])
        out = LeakyReLU(0.01)(out)
        
        b = Dense(fil)(style)
        b = Reshape([1, 1, fil])(b)
        g = Dense(fil)(style)
        g = Reshape([1, 1, fil])(g)

        n = Conv2D(filters = fil, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
        
        out = Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
        out = AdaInstanceNormalization()([out, b, g])
        out = add([out, n])
        out = LeakyReLU(0.01)(out)
        
        return out

    def build_generator(self):
        # class label input (latent for StyleGAN)
        inp_s = Input(shape=[self.num_classes])
        sty = Dense(512, kernel_initializer='he_normal')(inp_s)
        sty = LeakyReLU(0.1)(sty)
        sty = Dense(512, kernel_initializer='he_normal')(sty)
        sty = LeakyReLU(0.1)(sty)

        #Get the noise image and crop for each size
        inp_n = Input(shape = [self.img_rows, self.img_cols, 1])
        noi = [Activation('linear')(inp_n)]
        curr_size = self.img_rows
        while curr_size > 4:
            curr_size = int(curr_size / 2)
            noi.append(Cropping2D(int(curr_size/2))(noi[-1]))
        

        #latent vector input (img generator for StyleGAN)
        inp = Input(shape=[1])
        x = Dense(4 * 4 * 512, kernel_initializer='he_normal')(inp)
        x = Reshape([4, 4, 512])(x)
        x = self.g_block(x, sty, noi[3], 512)
        x = self.g_block(x, sty, noi[2], 256)
        x = self.g_block(x, sty, noi[1], 128)
        x = self.g_block(x, sty, noi[0], 64)
        x = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(x)

        return Model([inp, inp_n, inp_s], x)
    
    def build_discriminator(self):

        inp = Input([self.img_rows, self.img_rows, self.channels])

        model = Sequential()

        model = layers.d_block(model, 32, init=True)
        model = layers.d_block(model, 64)
        model = layers.d_block(model, 128)
        model = layers.d_block(model, 256)
        model.add(Flatten())

        model.summary()

        out = model(inp)

        x = Dense(128)(out)
        x = Activation('relu')(x)
        x = Dropout(0.6)(x)
        val = Dense(1)(x)

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
        gi = Input(shape=[1])
        gi1 = Input(shape=[self.img_rows, self.img_cols, 1])
        gi2 = Input(shape=[self.num_classes])
        gf = self.generator([gi, gi1, gi2])
        df, df1 = self.discriminator(gf)

        #gradient penalty pipeline
        da, dal = self.discriminator(ri)

        self.DM = Model(inputs=[ri, gi, gi1, gi2], outputs=[dr, drl, df, df1, da, dal])

        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=ri, weight=5)

        self.DM.compile(optimizer=Adam(0.0003, beta_1=0, beta_2=0.99, decay=0.00001), loss=['binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy', partial_gp_loss, partial_gp_loss], metrics=['accuracy'])

    def build_genModel(self):
        self.discriminator.trainable = False
        for layer in self.discriminator.layers:
            layer.trainable = False
        
        self.generator.trainable = True
        for layer in self.generator.layers:
            layer.trainable = True
        
        gi = Input(shape=[self.latent_dim])
        gi1 = Input(shape=[self.img_rows, self.img_cols, 1])
        gi2 = Input(shape=[self.num_classes])

        gf = self.generator([gi, gi1, gi2])
        df, dfl = self.discriminator(gf)

        self.AM = Model(inputs=[gi, gi1, gi2], outputs=[df, dfl])
        self.AM.compile(optimizer=Adam(0.0003, beta_1=0, beta_2=0.99, decay=0.00001), loss=['mse', 'mse'])


    def train(self, epochs, batch_size=32, sample_interval=50, start_point=0):

        # Load the dataset
        X_train, y_train = utils.load_data(self.writer)

        # Adversarial ground truths
        ones = np.ones((batch_size, 1))
        zeros = np.zeros((batch_size, 1))
        nones = -ones
        
        enoiseImage = np.random.uniform(0.0, 1.0, size = [batch_size, self.img_rows, self.img_cols, 1])

        for epoch in range(start_point, epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]


            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.uniform(0, 1, (batch_size, self.num_classes))
            sampled_labels = np.around(sampled_labels)

            # Image labels. 0-9 
            img_labels = y_train[idx]
            # Train the discriminator
            d_loss = self.DM.train_on_batch([imgs, ones, enoiseImage, sampled_labels], [ones, img_labels, nones, sampled_labels, ones, img_labels])
            # Train the generator
            g_loss = self.AM.train_on_batch([ones, enoiseImage, sampled_labels], [zeros, sampled_labels])

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[6], 100*d_loss[7], g_loss[0]))
            utils.write_log( self.writer, ['D loss', 'G loss', 'accuracy', 'class accuracy'], [d_loss[0], g_loss[0], 100*d_loss[6], 100*d_loss[7]], epoch)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                utils.save_model('%s/' % ('csgan' if self.flags.name is None else self.flags.name), self.generator, self.discriminator, epoch)
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
        ones = np.ones((r*c, 1))
        enoiseImage = np.random.uniform(0.0, 1.0, size = [r*c, self.img_rows, self.img_cols, 1])
        sampled_labels = np.random.uniform(0, 1, (r*c, self.num_classes))
        sampled_labels = np.around(sampled_labels)
        gen_imgs = self.generator.predict([ones, enoiseImage, sampled_labels])
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
        fig.savefig(self.images_path + "/%d.png" % (epoch))
        plt.close()
