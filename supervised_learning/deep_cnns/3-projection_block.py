#!/usr/bin/env python3
"""Module that defines the projection_block function."""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Builds a projection block for a residual network.

    The block applies three convolutions in the main path and
    a convolution in the shortcut path to match dimensions.
    Each convolution is followed by batch normalization, and
    ReLU activation is applied after the first two convolutions
    and after adding the shortcut connection.

    Returns:
        The activated output of the projection block.
    """

    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    X = K.layers.Conv2D(F11, (1, 1), strides=(s, s),
                        padding='valid',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(F12, (1, 1),
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    X_short = K.layers.Conv2D(F12, (1, 1), strides=(s, s),
                              padding='valid',
                              kernel_initializer=initializer)(A_prev)
    X_short = K.layers.BatchNormalization(axis=3)(X_short)

    X = K.layers.Add()([X, X_short])
    X = K.layers.Activation('relu')(X)

    return X
