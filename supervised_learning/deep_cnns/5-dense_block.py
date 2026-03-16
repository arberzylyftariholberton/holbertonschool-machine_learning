#!/usr/bin/env python3
"""Module that defines the dense_block function."""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block for a DenseNet network.

    The block uses the DenseNet-B bottleneck architecture.
    Each layer is made of batch normalization, ReLU,
    a 1x1 convolution, batch normalization, ReLU, and
    a 3x3 convolution. The output of each layer is
    concatenated with the previous outputs.

    Returns:
        The concatenated output of the dense block and
        the updated number of filters.
    """

    initializer = K.initializers.he_normal(seed=0)

    for _ in range(layers):
        Y = K.layers.BatchNormalization(axis=3)(X)
        Y = K.layers.Activation('relu')(Y)
        Y = K.layers.Conv2D(4 * growth_rate, (1, 1),
                            padding='same',
                            kernel_initializer=initializer)(Y)

        Y = K.layers.BatchNormalization(axis=3)(Y)
        Y = K.layers.Activation('relu')(Y)
        Y = K.layers.Conv2D(growth_rate, (3, 3),
                            padding='same',
                            kernel_initializer=initializer)(Y)

        X = K.layers.Concatenate(axis=3)([X, Y])
        nb_filters += growth_rate

    return X, nb_filters
