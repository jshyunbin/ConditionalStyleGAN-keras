from keras.models import Sequential, Model, model_from_json
from PIL import Image
import tensorflow as tf
import numpy as np
import io
import os
import h5py
import keras


def save_model(name, generator, discriminator, step):

        def save(model, model_name, step):
            model_path = "../saved_model/%s.json" % model_name
            weights_path = "../saved_model/%s_%dweights.hdf5" % (model_name, step)
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(generator, name + "_generator", step)
        save(discriminator, name + "_discriminator", step)

def write_log(writer, names, logs, step):
        for name, value in zip(names, logs):
            summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
            writer.add_summary(summary, step)

def write_image(writer, name, images, step=None):
    for i in range(images.shape[0]):
        output = io.BytesIO()
        image = Image.fromarray((images[i] * 255).astype('uint8'))
        image.save(output, 'PNG')
        image_string = output.getvalue()
        output.close()
        summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=tf.Summary.Image(height=64, width=64, colorspace=3, encoded_image_string=image_string))])
        writer.add_summary(summary, step)

def load_data(writer):
        print("Loading celebA dataset ...")

        dataset_file = '../dataset/celeba.hdf5'

        if os.path.isfile(dataset_file):
            print('Loading dataset from saved file')
            f = h5py.File(dataset_file, 'r')
            trainX, trainy = f['image'].value, f['label'].value
            f.close()
            write_image(writer, 'Dataset', trainX[:10] * 0.5 + 0.5)
            return (trainX, trainy)


        train_img_file = sorted([os.path.join('../dataset/train', fname) for fname in os.listdir('../dataset/train')])
        # loading labels
        label_file = open('../dataset/list_attr_celeba_full.txt', 'r').readlines()

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
            img_crop = np.array(image.resize((64, 64), Image.BICUBIC))

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