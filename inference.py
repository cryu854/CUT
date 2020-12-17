""" USAGE
python ./inference.py --weights ./output/checkpoints --input ./datasets/horse2zebra/testA
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from modules.cut_model import CUT_model
from utils import create_dir, load_image


def ArgParse():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CUT inference usage.')
    # Inference
    parser.add_argument('--mode', help="Model's mode be one of: 'cut', 'fastcut'", type=str, default='cut', choices=['cut', 'fastcut'])
    parser.add_argument('--weights', help='Pre-trained checkpoints/weights directory', type=str, default='./output/checkpoints')
    parser.add_argument('--input', help='Input folder', type=str, default='./source')
    parser.add_argument('--output', help='Output folder', type=str, default='./translated')
    parser.add_argument('--output_channel', help="Output image's channel", type=int, default=3)

    args = parser.parse_args()

    # Check arguments
    assert os.path.exists(args.input), 'Input folder does not exist.'
    assert os.path.exists(args.weights), 'Pre-trained checkpoints/weights does not exist.'
    assert args.output_channel > 0, 'Number of channels must greater than zero.'
    
    return args


def main(args):
    # Load input images
    input_images = tf.data.Dataset.list_files([args.input+'/*.jpg', args.input+'/*.png'])
    input_images = (
        input_images.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(1)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
        
    # Get image shape
    image_shape = next(iter(input_images))
    height, width, channel = image_shape.shape[1:]

    # Create model
    cut = CUT_model(source_shape = [height,width,channel],
                    target_shape = [height,width,args.output_channel],
                    cut_mode = args.mode)

    # Load weights
    latest_ckpt = tf.train.latest_checkpoint(args.weights)
    cut.load_weights(latest_ckpt).expect_partial()
    cut.save_weights('./weights/weights')
    print(f"Restored weights from {latest_ckpt}.")

    # Translate images
    out_dir = create_dir(args.output)
    for i, source in enumerate(input_images):
        prediction = cut.netG(source)[0].numpy()
        prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
        source = (source[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        _, ax = plt.subplots(1, 2, figsize=(15, 10))
        ax[0].imshow(source)
        ax[1].imshow(prediction)
        ax[0].set_title("Input")
        ax[1].set_title("Translated")
        ax[0].axis("off")
        ax[1].axis("off")

        plt.savefig(f'{out_dir}/infer={i + 1}.png')
        plt.close()


if __name__ == '__main__':
    main(ArgParse())
