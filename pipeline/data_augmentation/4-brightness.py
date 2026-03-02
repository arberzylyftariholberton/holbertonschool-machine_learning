#!/usr/bin/env python3
"""Randomly changes the brightness of an image"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image: 3D tf.Tensor containing the image to change
        max_delta: Maximum delta for brightness adjustment

    Returns:
        Brightness-adjusted image as a tf.Tensor
    """
    return tf.image.random_brightness(image, max_delta)
