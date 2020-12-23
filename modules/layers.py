""" Implement the following layers that used in CUT/FastCUT model.
Padding2D
InstanceNorm
L2Normalize
AntialiasSampling
ConvBlock
ResBlock
"""

import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Layer, Conv2D, Activation, BatchNormalization, Lambda
from modules.ops.upfirdn_2d import upsample_2d, downsample_2d


class Padding2D(Layer):
    """ 2D padding layer.
    Args:
        padding(tuple): Amount of padding for the
        spatial dimensions.
    Returns:
        A padded tensor with the same type as the input tensor.
    """
    def __init__(self, padding=(1, 1), pad_type='constant', **kwargs):
        assert pad_type in ['constant', 'reflect', 'symmetric']
        super(Padding2D, self).__init__(**kwargs)
        self.padding = (padding, padding) if type(padding) is int else tuple(padding)
        self.pad_type = pad_type

    def call(self, inputs, training=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]

        return tf.pad(inputs, padding_tensor, mode=self.pad_type)


class InstanceNorm(tf.keras.layers.Layer):
    """ Instance Normalization layer (https://arxiv.org/abs/1607.08022).
    """
    
    def __init__(self, epsilon=1e-5, affine=True, **kwargs):
        super(InstanceNorm, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.affine = affine
 
    def build(self, input_shape):
        if self.affine:
            self.gamma = self.add_weight(name='gamma',
                                        shape=(input_shape[-1],),
                                        initializer=tf.random_normal_initializer(0, 0.02),
                                        trainable=True)
            self.beta = self.add_weight(name='beta',
                                        shape=(input_shape[-1],),
                                        initializer=tf.zeros_initializer(),
                                        trainable=True)
        else:
            self.gamma = tf.ones_like(input_shape[-1], dtype=tf.float32)
            self.beta = tf.zeros_like(input_shape[-1], dtype=tf.float32)

    def call(self, inputs, training=None):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean), tf.math.sqrt(tf.add(var, self.epsilon)))

        return self.gamma * x + self.beta


class L2Normalize(Layer):
    """ L2 Normalization layer.
    """
    def __init__(self, epsilon=1e-10, **kwargs):
        super(L2Normalize, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, inputs, training=None):
        norm_factor = tf.math.sqrt(tf.reduce_sum(inputs**2, axis=1, keepdims=True))
        
        return inputs / (norm_factor + self.epsilon)


class AntialiasSampling(tf.keras.layers.Layer):
    """ Down/Up sampling layer with blur-kernel.
    """
    def __init__(self,
                 kernel_size,
                 mode,
                 impl, 
                 **kwargs):
        super(AntialiasSampling, self).__init__(**kwargs)
        if(kernel_size == 1):
            self.k = np.array([1., ])
        elif(kernel_size == 2):
            self.k = np.array([1., 1.])
        elif(kernel_size == 3):
            self.k = np.array([1., 2., 1.])
        elif(kernel_size == 4):
            self.k = np.array([1., 3., 3., 1.])
        elif(kernel_size == 5):
            self.k = np.array([1., 4., 6., 4., 1.])
        elif(kernel_size == 6):
            self.k = np.array([1., 5., 10., 10., 5., 1.])
        elif(kernel_size == 7):
            self.k = np.array([1., 6., 15., 20., 15., 6., 1.])
        self.mode = mode
        self.impl = impl

    def call(self, inputs, training=None):
        if self.mode == 'up':
            x = upsample_2d(inputs, k=self.k, data_format='NHWC', impl=self.impl)
        elif self.mode == 'down':
            x = downsample_2d(inputs, k=self.k, data_format='NHWC', impl=self.impl)
        else:
            raise ValueError(f'Unsupported sampling mode: {self.mode}')

        return x


class ConvBlock(Layer):
    """ ConBlock layer that consists of Conv2D + Normalization + Activation.
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1,1),
                 padding='valid',
                 use_bias=True,
                 norm_layer=None,
                 activation='linear',
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        initializer = tf.random_normal_initializer(0., 0.02)
        self.conv2d = Conv2D(filters,
                             kernel_size,
                             strides,
                             padding,
                             use_bias=use_bias,
                             kernel_initializer=initializer)
        self.activation = Activation(activation)
        if norm_layer == 'batch':
            self.normalization = BatchNormalization()
        elif norm_layer == 'instance':
            self.normalization = InstanceNorm(affine=False)
        else:
            self.normalization = Lambda(lambda x: tf.identity(x))

    def call(self, inputs, training=None):
        x = self.conv2d(inputs)
        x = self.normalization(x)
        x = self.activation(x)

        return x


class ResBlock(Layer):
    """ ResBlock is a ConvBlock with skip connections.
    Original Resnet paper (https://arxiv.org/pdf/1512.03385.pdf).
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 use_bias,
                 norm_layer,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)

        self.reflect_pad1 = Padding2D(1, pad_type='reflect')
        self.conv_block1 = ConvBlock(filters,
                                     kernel_size,
                                     padding='valid',
                                     use_bias=use_bias,
                                     norm_layer=norm_layer,
                                     activation='relu')

        self.reflect_pad2 = Padding2D(1, pad_type='reflect')
        self.conv_block2 = ConvBlock(filters,
                                     kernel_size,
                                     padding='valid',
                                     use_bias=use_bias,
                                     norm_layer=norm_layer)

    def call(self, inputs, training=None):
        x = self.reflect_pad1(inputs)
        x = self.conv_block1(x)

        x = self.reflect_pad2(x)
        x = self.conv_block2(x)

        return inputs + x
