# -*- coding: utf-8 -*-

"""
keras_resnet.models._1d
~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular one-dimensional residual models.
"""

from keras import layers as ll
from keras.layers import Input, Dense, Activation
from keras.models import Sequential, Model
from keras.layers.core import Flatten, RepeatVector, Reshape, Dropout, Masking
from keras.layers.convolutional import Conv1D, ZeroPadding1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers.merge import dot, multiply, concatenate, add
from keras.layers.normalization import BatchNormalization



def ResNet18_1d(input_tensor, final_layer_output=220, append='n'):
    x = Conv1D(
        64, 7, strides=2, padding='same', name='conv1' + append)(input_tensor)
    x = BatchNormalization(name='bn_conv1' + append)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides=2)(x)
    x = conv_block(x, 3, [64, 64, 256],
                   stage=2, block='a' + append, strides=1)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b' + append)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c' + append)
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a' + append)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + append)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c' + append)
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + append)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c' + append)
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a' + append)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b' + append)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c' + append)
    x = AveragePooling1D(final_layer_output, name='avg_pool' + append)(x)
    x = Flatten()(x)
    return x
