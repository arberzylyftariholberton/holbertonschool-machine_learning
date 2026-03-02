#!/usr/bin/env python3
"""Randomly adjusts the contrast of an image"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image: 3D tf.Tensor representing the input image
        lower: Lower bound for the random contrast factor
        upper: Upper bound for the random contrast factor

    Returns:
        Contrast-adjusted image as a tf.Tensor
    """
    return tf.image.random_contrast(image, lower, upper)
