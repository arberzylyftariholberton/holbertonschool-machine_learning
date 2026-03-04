#!/usr/bin/env python3
"""PCA Color Augmentation as described in the AlexNet paper"""
import tensorflow as tf


def pca_color(image, alphas):
    """
    Performs PCA color augmentation on an image.

    Args:
        image: 3D tf.Tensor of shape (H, W, 3) - the image to augment
        alphas: tuple of length 3 - magnitude of change per channel

    Returns:
        Augmented image as a tf.Tensor clipped to [0, 255]
    """
    # Cast to float32 and reshape to (N, 3)
    img = tf.cast(image, tf.float32)
    shape = tf.shape(img)
    pixels = tf.reshape(img, [-1, 3])

    # Center the pixels
    mean = tf.reduce_mean(pixels, axis=0)
    pixels_centered = pixels - mean

    # Compute 3x3 covariance matrix: C = (X^T X) / (N - 1)
    n = tf.cast(tf.shape(pixels_centered)[0], tf.float32)
    cov = tf.linalg.matmul(
        pixels_centered, pixels_centered, transpose_a=True
    ) / (n - 1.0)

    # Eigendecomposition (eigh for symmetric matrices)
    eigenvalues, eigenvectors = tf.linalg.eigh(cov)

    # AlexNet perturbation: P @ (alphas * lambdas)
    alphas_tensor = tf.cast(tf.constant(alphas), tf.float32)
    perturbation = tf.linalg.matvec(
        eigenvectors, alphas_tensor * eigenvalues
    )

    # Add perturbation to every pixel and reshape back
    img_augmented = img + perturbation
    img_augmented = tf.clip_by_value(img_augmented, 0.0, 255.0)

    return tf.cast(img_augmented, tf.uint8)
