# -*- coding: utf-8 -*-

"""
keras_resnet.classifiers
~~~~~~~~~~~~~~~~~~~~~~~~

This module implements popular residual one-dimensional classifiers.
"""

import keras.backend
import keras.layers
import keras.models
import keras.regularizers

import keras_resnet.models

class ResNet18_1d(keras.models.Model):
    """
    A :class:`ResNet18 <ResNet18>` object.

    :param inputs: input tensor (e.g. an instance of `keras.layers.Input`)

    Usage:

        >>> import keras_resnet.classifiers

        >>> shape, classes = (224, 224, 3), 1000

        >>> x = keras.layers.Input(shape)

        >>> model = keras_resnet.classifiers.ResNet18(x)

        >>> model.compile("adam", "categorical_crossentropy", ["accuracy"])
    """
    def __init__(self, inputs, classes):
        outputs = keras_resnet.models.ResNet18_1d(inputs)

        outputs = keras.layers.Flatten()(outputs.output)

        outputs = keras.layers.Dense(classes, activation="softmax")(outputs)

        super(ResNet18_1d, self).__init__(inputs, outputs)
