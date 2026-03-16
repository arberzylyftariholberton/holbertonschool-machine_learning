#!/usr/bin/env python3
"""Module that defines the transition_layer function."""

from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer for a DenseNet network.

    The layer uses batch normalization, ReLU, a 1x1
    convolution, and average pooling. It applies the
    compression factor to reduce the number of filters.

    Returns:
        The output of the transition layer and the
        updated number of filters.
    """

    initializer = K.initializers.he_normal(seed=0)
    nb_filters = int(nb_filters * compression)

    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(nb_filters, (1, 1),
                        padding='same',
                        kernel_initializer=initializer)(X)

    X = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(X)

    return X, nb_filters
