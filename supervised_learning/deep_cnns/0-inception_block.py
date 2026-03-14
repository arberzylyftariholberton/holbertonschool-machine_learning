#!/usr/bin/env python3
"""Module that defines the inception_block function for CNNs."""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Concatenate


def inception_block(A_prev, filters):
    """Builds an Inception block as described in GoogLeNet (2014).

    Parameters:
    A_prev -- output from the previous layer
    filters -- tuple/list containing:
        F1: filters for 1x1 conv
        F3R: filters for 1x1 conv before 3x3
        F3: filters for 3x3 conv
        F5R: filters for 1x1 conv before 5x5
        F5: filters for 5x5 conv
        FPP: filters for 1x1 conv after max pooling

    Returns:
    Concatenated output of the inception block
    """

    F1, F3R, F3, F5R, F5, FPP = filters

    conv1 = Conv2D(F1, (1, 1), padding='same', activation='relu')(A_prev)

    conv3r = Conv2D(F3R, (1, 1), padding='same', activation='relu')(A_prev)
    conv3 = Conv2D(F3, (3, 3), padding='same', activation='relu')(conv3r)

    conv5r = Conv2D(F5R, (1, 1), padding='same', activation='relu')(A_prev)
    conv5 = Conv2D(F5, (5, 5), padding='same', activation='relu')(conv5r)

    pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(A_prev)
    pool_conv = Conv2D(FPP, (1, 1), padding='same',
                       activation='relu')(pool)

    return Concatenate()([conv1, conv3, conv5, pool_conv])
