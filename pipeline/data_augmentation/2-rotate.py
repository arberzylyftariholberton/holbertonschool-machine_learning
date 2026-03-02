#!/usr/bin/env python3
"""Rotates an image by 90 degrees counter-clockwise"""

import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Args:
        image: 3D tf.Tensor containing the image to rotate

    Returns:
        Rotated image as a tf.Tensor
    """
    return tf.image.rot90(image)
