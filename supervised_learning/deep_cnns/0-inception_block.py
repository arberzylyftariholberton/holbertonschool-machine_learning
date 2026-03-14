#!/usr/bin/env python3
"""Module that defines the inception_block function."""

from tensorflow import keras as K


def inception_block(A_prev, filters):
    """Builds an Inception block for a convolutional neural network.

    A_prev is the output from the previous layer.
    filters is a tuple/list containing:
        F1  - filters for 1x1 convolution
        F3R - filters for 1x1 before 3x3 convolution
        F3  - filters for 3x3 convolution
        F5R - filters for 1x1 before 5x5 convolution
        F5  - filters for 5x5 convolution
        FPP - filters for 1x1 after max pooling

    Returns the concatenated output of the block.
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = K.layers.Conv2D(F1, (1, 1),
                            padding='same',
                            activation='relu')(A_prev)

    conv3r = K.layers.Conv2D(F3R, (1, 1),
                             padding='same',
                             activation='relu')(A_prev)
    conv3 = K.layers.Conv2D(F3, (3, 3),
                            padding='same',
                            activation='relu')(conv3r)

    conv5r = K.layers.Conv2D(F5R, (1, 1),
                             padding='same',
                             activation='relu')(A_prev)
    conv5 = K.layers.Conv2D(F5, (5, 5),
                            padding='same',
                            activation='relu')(conv5r)

    pool = K.layers.MaxPooling2D((3, 3),
                                 strides=(1, 1),
                                 padding='same')(A_prev)
    pool_conv = K.layers.Conv2D(FPP, (1, 1),
                                padding='same',
                                activation='relu')(pool)

    return K.layers.Concatenate()([conv1, conv3, conv5, pool_conv])
