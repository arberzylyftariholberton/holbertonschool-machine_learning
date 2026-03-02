#!/usr/bin/env python3
"""Flips an image horizontally"""

import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    Args:
        image: 3D tf.Tensor containing the image to flip

    Returns:
        Flipped image as a tf.Tensor
    """
    return tf.image.flip_left_right(image)
