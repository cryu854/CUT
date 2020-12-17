import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image


def create_dir(dir):
    """ Create the directory.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f'Directory {dir} createrd')
    else:
        print(f'Directory {dir} already exists')  

    return dir


def imsave(image, file_name):
    """ Save the image.
    """
    image = (image + 1) * 127.5
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)
    image = Image.fromarray(np.array(image).astype(np.uint8).squeeze())
    image.save(file_name)


def load_image(image_file, image_size=None):
    """ Load the image file.
    """
    image = tf.io.read_file(image_file)
    image = tf.image.decode_png(image)
    image = tf.image.flip_left_right(image)
    image = (tf.cast(image, tf.float32) / 127.5) - 1.0

    if image_size is not None:
        image = tf.image.resize(image, size=(image_size[0], image_size[1]))
    if tf.shape(image)[-1] == 1:
        image = tf.tile(image, [1,1,3])

    return  image
