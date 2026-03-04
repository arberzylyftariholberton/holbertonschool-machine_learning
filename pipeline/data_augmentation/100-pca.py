#!/usr/bin/env python3
"""PCA Color Augmentation as described in the AlexNet paper"""
import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation on an image.

    Args:
        image: 3D tf.Tensor of shape (H, W, 3)
        alphas: tuple/array of length 3 - per-channel scaling

    Returns:
        Augmented image as tf.uint8 Tensor
    """
    img = tf.cast(image, tf.float32)
    pixels = tf.reshape(img, [-1, 3])

    mean = tf.reduce_mean(pixels, axis=0, keepdims=True)
    centered = pixels - mean

    n = tf.cast(tf.shape(centered)[0], tf.float32)
    cov = tf.linalg.matmul(
        centered, centered, transpose_a=True
    ) / (n - 1.0)

    eigenvalues, eigenvectors = tf.linalg.eigh(cov)

    alphas_t = tf.cast(
        tf.convert_to_tensor(alphas, dtype=tf.float64),
        tf.float32
    )

    delta = tf.linalg.matvec(
        eigenvectors, alphas_t * eigenvalues
    )

    augmented = tf.clip_by_value(img + delta, 0.0, 255.0)

    return tf.cast(augmented, tf.uint8)
