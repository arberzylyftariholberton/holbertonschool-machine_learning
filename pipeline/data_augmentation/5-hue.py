#!/usr/bin/env python3
"""Changes the hue of an image"""

import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image: 3D tf.Tensor containing the image to change
        delta: Amount the hue should change

    Returns:
        Hue-adjusted image as a tf.Tensor
    """
    return tf.image.adjust_hue(image, delta)
