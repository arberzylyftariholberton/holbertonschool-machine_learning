#!/usr/bin/env python3
"""Module that defines the identity_block function."""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """Builds an identity block for a residual network.

    The block follows the architecture described in
    "Deep Residual Learning for Image Recognition" (2015).
    It applies three convolutions, each followed by batch
    normalization, with ReLU activation after the first two
    convolutions and after adding the shortcut connection.

    Returns:
        The activated output of the identity block.
    """

    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    X = K.layers.Conv2D(F11, (1, 1),
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

    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)

    return X
