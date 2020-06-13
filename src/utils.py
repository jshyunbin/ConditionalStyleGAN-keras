from keras.models import Sequential, Model, model_from_json
from PIL import Image
import tensorflow as tf
import io
import keras


def save_model(name, generator, discriminator, step):

        def save(model, model_name, step):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_%dweights.hdf5" % (model_name, step)
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