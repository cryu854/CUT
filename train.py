""" USAGE
python ./train.py --train_src_dir ./datasets/horse2zebra/trainA --train_tar_dir ./datasets/horse2zebra/trainB --test_src_dir ./datasets/horse2zebra/testA --test_tar_dir ./datasets/horse2zebra/testB
"""

import os
import argparse
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from modules.cut_model import CUT_model
from utils import create_dir, load_image


def ArgParse():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CUT training usage.')
    # Training
    parser.add_argument('--mode', help="Model's mode be one of: 'cut', 'fastcut'", type=str, default='cut', choices=['cut', 'fastcut'])
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=400)
    parser.add_argument('--batch_size', help='Training batch size', type=int, default=1)
    parser.add_argument('--beta_1', help='First Momentum term of adam', type=float, default=0.5)
    parser.add_argument('--beta_2', help='Second Momentum term of adam', type=float, default=0.999)
    parser.add_argument('--lr', help='Initial learning rate for adam', type=float, default=0.0002)
    parser.add_argument('--lr_decay_rate', help='lr_decay_rate', type=float, default=0.9)
    parser.add_argument('--lr_decay_step', help='lr_decay_step', type=int, default=100000)
    # Define data
    parser.add_argument('--out_dir', help='Outputs folder', type=str, default='./output')
    parser.add_argument('--train_src_dir', help='Train-source dataset folder', type=str, default='./datasets/horse2zebra/trainA')
    parser.add_argument('--train_tar_dir', help='Train-target dataset folder', type=str, default='./datasets/horse2zebra/trainB')
    parser.add_argument('--test_src_dir', help='Test-source dataset folder', type=str, default='./datasets/horse2zebra/testA')
    parser.add_argument('--test_tar_dir', help='Test-target dataset folder', type=str, default='./datasets/horse2zebra/testB')
    # Misc
    parser.add_argument('--ckpt', help='Resume training from checkpoint', type=str)
    parser.add_argument('--save_n_epoch', help='Every n epochs to save checkpoints', type=int, default=5)
    parser.add_argument('--impl', help="(Faster)Custom op use:'cuda'; (Slower)Tensorflow op use:'ref'", type=str, default='ref', choices=['ref', 'cuda'])

    args = parser.parse_args()

    # Check arguments
    assert args.lr > 0
    assert args.epochs > 0
    assert args.batch_size > 0
    assert args.save_n_epoch > 0
    assert os.path.exists(args.train_src_dir), 'Error: Train source dataset does not exist.'
    assert os.path.exists(args.train_tar_dir), 'Error: Train target dataset does not exist.'
    assert os.path.exists(args.test_src_dir), 'Error: Test source dataset does not exist.'
    assert os.path.exists(args.test_tar_dir), 'Error: Test target dataset does not exist.'

    return args


def main(args):
    # Create datasets
    train_dataset, test_dataset = create_dataset(args.train_src_dir,
                                                 args.train_tar_dir,
                                                 args.test_src_dir,
                                                 args.test_tar_dir,
                                                 args.batch_size)

    # Get image shape
    source_image, target_image = next(iter(train_dataset))
    source_shape = source_image.shape[1:]
    target_shape = target_image.shape[1:]

    # Create model
    cut = CUT_model(source_shape, target_shape, cut_mode=args.mode, impl=args.impl)

    # Define learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.lr,
                                                                 decay_steps=args.lr_decay_step,
                                                                 decay_rate=args.lr_decay_rate,
                                                                 staircase=True)

    # Compile model
    cut.compile(G_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2),
                F_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2),
                D_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=args.beta_1, beta_2=args.beta_2),)

    # Restored from previous checkpoints, or initialize checkpoints from scratch
    if args.ckpt is not None:
        latest_ckpt = tf.train.latest_checkpoint(args.ckpt)
        cut.load_weights(latest_ckpt)
        initial_epoch = int(latest_ckpt[-3:])
        print(f"Restored from {latest_ckpt}.")
    else:
        initial_epoch = 0
        print("Initializing from scratch...")

    # Create folders to store the output information
    result_dir = f'{args.out_dir}/images'
    checkpoint_dir = f'{args.out_dir}/checkpoints'
    log_dir = f'{args.out_dir}/logs/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    
    # Create validating callback to generate output image every epoch
    plotter_callback = GANMonitor(cut.netG, test_dataset, result_dir)

    # Create checkpoint callback to save model's checkpoints every n epoch (default 5)
    # "period" to save every n epochs, "save_freq" to save every n batches
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir+'/{epoch:03d}', period=args.save_n_epoch, verbose=1)

    # Create tensorboard callback to log losses every epoch
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

    # Train cut model
    cut.fit(train_dataset,
            epochs=args.epochs,
            initial_epoch=initial_epoch,
            callbacks=[plotter_callback, checkpoint_callback, tensorboard_callback],
            verbose=1)


def create_dataset(train_src_folder, 
                   train_tar_folder, 
                   test_src_folder,
                   test_tar_folder, 
                   batch_size):
    """ Create tf.data.Dataset.
    """
    # Create train dataset
    train_src_dataset = tf.data.Dataset.list_files([train_src_folder+'/*.jpg', train_src_folder+'/*.png'], shuffle=True)
    train_src_dataset = (
        train_src_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    train_tar_dataset = tf.data.Dataset.list_files([train_tar_folder+'/*.jpg', train_tar_folder+'/*.png'], shuffle=True)
    train_tar_dataset = (
        train_tar_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    
    train_dataset = tf.data.Dataset.zip((train_src_dataset, train_tar_dataset))

    # Create test dataset
    test_src_dataset = tf.data.Dataset.list_files([test_src_folder+'/*.jpg', test_src_folder+'/*.png'])
    test_src_dataset = (
        test_src_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    test_tar_dataset = tf.data.Dataset.list_files([test_tar_folder+'/*.jpg', test_tar_folder+'/*.png'])
    test_tar_dataset = (
        test_tar_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    test_dataset = tf.data.Dataset.zip((test_src_dataset, test_tar_dataset))

    return train_dataset, test_dataset


class GANMonitor(tf.keras.callbacks.Callback):
    """ A callback to generate and save images after each epoch
    """
    def __init__(self, generator, test_dataset, out_dir, num_img=2):
        self.num_img = num_img
        self.generator = generator
        self.test_dataset = test_dataset
        self.out_dir = create_dir(out_dir)

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(self.num_img, 4, figsize=(20, 10))
        [ax[0, i].set_title(title) for i, title in enumerate(['Source', "Translated", "Target", "Identity"])]
        for i, (source, target) in enumerate(self.test_dataset.take(self.num_img)):
            translated = self.generator(source)[0].numpy()
            translated = (translated * 127.5 + 127.5).astype(np.uint8)
            source = (source[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            idt = self.generator(target)[0].numpy()
            idt = (idt * 127.5 + 127.5).astype(np.uint8)
            target = (target[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            [ax[i, j].imshow(img) for j, img in enumerate([source, translated, target, idt])]
            [ax[i, j].axis("off") for j in range(4)]

        plt.savefig(f'{self.out_dir}/epoch={epoch + 1}.png')
        plt.close()


if __name__ == '__main__':
    main(ArgParse())
