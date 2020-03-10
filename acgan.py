# example of fitting an auxiliary classifier gan (ac-gan) on fashion mnsit
import os
import numpy as np
import scipy.misc
import PIL
import tensorflow as tf
from numpy import zeros
from numpy import ones
from numpy import expand_dims
from numpy.random import randn
from numpy.random import randint
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Activation
from keras.layers import Concatenate
from keras.initializers import RandomNormal
from matplotlib import pyplot
from PIL import Image

# define the standalone discriminator model
def define_discriminator(in_shape=(64,64,3), n_classes=5):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# image input
	in_image = Input(shape=in_shape)
	# downsample to 32x32
	fe = Conv2D(4, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# normal
	fe = Conv2D(8, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)

	# downsample to 16x16
	fe = Conv2D(16, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# normal
	fe = Conv2D(32, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)

	# downsample to 8x8
	fe = Conv2D(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# normal
	fe = Conv2D(128, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)

	# downsample to 4x4
	fe = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# normal
	fe = Conv2D(512, (3,3), padding='same', kernel_initializer=init)(fe)
	fe = BatchNormalization()(fe)
	fe = LeakyReLU(alpha=0.2)(fe)
	fe = Dropout(0.5)(fe)
	# flatten feature maps
	fe = Flatten()(fe)
	# real/fake output
	out1 = Dense(1, activation='sigmoid')(fe)
	# class label output
	out2 = Dense(n_classes, activation='softmax')(fe)
	# define model
	model = Model(in_image, [out1, out2])
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)

	print(model.summary())
	return model

# define the standalone generator model
def define_generator(latent_dim, n_classes=5):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# label input
	in_label = Input(shape=(n_classes,))
	# embedding for categorical input
	li = Embedding(n_classes, 50)(in_label)
	# linear multiplication
	n_nodes = 8 * 8
	li = Dense(n_nodes, kernel_initializer=init)(li)
	# reshape to additional channel
	li = Reshape((8, 8, 5))(li)
	# image generator input
	in_lat = Input(shape=(latent_dim,))
	# foundation for 7x7 image
	n_nodes = 384 * 8 * 8
	gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
	gen = Activation('relu')(gen)
	gen = Reshape((8, 8, 384))(gen)
	# merge image gen and label input
	merge = Concatenate()([gen, li])
	# upsample to 16x16
	gen = Conv2DTranspose(192, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(merge)
	gen = BatchNormalization()(gen)
	gen = Activation('relu')(gen)
	# upsample to 32x32
	gen = Conv2DTranspose(96, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	gen = BatchNormalization()(gen)
	gen = Activation('relu')(gen)
	# upsample to 64x64
	gen = Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(gen)
	out_layer = Activation('tanh')(gen)
	# define model
	model = Model([in_lat, in_label], out_layer)

	return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect the outputs of the generator to the inputs of the discriminator
	gan_output = d_model(g_model.output)
	# define gan model as taking noise and label and outputting real/fake and label outputs
	model = Model(g_model.input, gan_output)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)
	return model

# load images
def load_real_samples():
	# load dataset
	print("Loading celebA dataset ...")
	print(os.getcwd())

	train_img_file = sorted([os.path.join('dataset/train', fname) for fname in os.listdir('dataset/train')])

	label_file = open('dataset/list_attr_celeba_full.txt', 'r').readlines()

	label = [[0. if i == '-1' else 1.0 for i in x.split()[1:]] for x in label_file[2:]]

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

	trainX = np.array(trainX)
	trainX = (trainX - 127.5) / 127.5
	trainy = np.array(trainy)

	return [trainX, trainy]

# select real samples
def generate_real_samples(dataset, n_samples):
	# split into images and labels
	images, labels = dataset
	# choose random instances
	ix = randint(0, images.shape[0], n_samples)
	# select images and labels
	X, labels = images[ix], labels[ix]
	# generate class labels
	y = ones((n_samples, 1))
	return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=5):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = np.array([randint(0, 1, n_classes) for i in range(n_samples)])
	labels.astype(np.float32)
	return [z_input, labels]

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	z_input, labels_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	images = generator.predict([z_input, labels_input])
	# create class labels
	y = zeros((n_samples, 1))
	return [images, labels_input], y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, writer, n_samples=100):
	# prepare fake examples
	[X, _], _ = generate_fake_samples(g_model, latent_dim, n_samples)
	# scale from [-1,1] to [0,1]
	# print(X[0:4])
	# with writer.as_default():
	# 	images = np.reshape(X[0:4], (-1, 64, 64, 3))
	# 	tf.summary.image("Generated Image", images, max_outputs=4, step=step)
	# 	writer.flush()
	X = (X + 1) / 2.0
	# plot images
	for i in range(100):
		# define subplot
		pyplot.subplot(10, 10, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(X[i, :, :, :])
	# save plot to file
	filename1 = 'generated_plot_%04d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%04d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

def write_log(writer, names, logs, batch_no):
	for name, value in zip(names, logs):
		with writer.as_default():
			tf.summary.scalar(name, value, step=batch_no)
			writer.flush()

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, writer, n_epochs=100, n_batch=64):
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset[0].shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_steps):
		# get randomly selected 'real' samples
		[X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
		# update discriminator model weights
		_,d_r1,d_r2 = d_model.train_on_batch(X_real, [y_real, labels_real])
		# generate 'fake' examples
		[X_fake, labels_fake], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
		# update discriminator model weights
		_,d_f,d_f2 = d_model.train_on_batch(X_fake, [y_fake, labels_fake])
		# prepare points in latent space as input for the generator
		[z_input, z_labels] = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = ones((n_batch, 1))
		# update the generator via the discriminator's error
		_,g_1,g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
		write_log(writer, ['LS', 'LC'], [g_1, g_2], i)
		# summarize loss on this batch
		if (i+1) % bat_per_epo == 0:
			print('>Epoch: %d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f,d_f2, g_1,g_2))

		# evaluate the model performance every 'epoch'
		if (i+1) % (bat_per_epo) == 0:
			summarize_performance((i+1) / bat_per_epo, g_model, latent_dim, writer)

# size of the latent space
latent_dim = 100

log_path = './logs'
writer = tf.summary.create_file_writer(log_path)
# create the discriminator
discriminator = define_discriminator()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, discriminator)
# load image data
dataset = load_real_samples()

with writer.as_default():
	images = np.reshape(dataset[0][0:25], (-1, 64, 64, 3))
	tf.summary.image("Training Data", images, max_outputs=25, step=0)
	writer.flush()

print("Dataset Load finished")

# train model
train(generator, discriminator, gan_model, dataset, latent_dim, writer)