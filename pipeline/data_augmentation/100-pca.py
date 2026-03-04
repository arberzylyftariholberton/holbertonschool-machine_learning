#!/usr/bin/env python3
"""PCA Color Augmentation as described in the AlexNet paper"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """
    Performs PCA color augmentation on an image.

    Args:
        image: 3D tf.Tensor of shape (H, W, 3) - the image to augment
        alphas: tuple of length 3 - magnitude of change per channel

    Returns:
        Augmented image as a tf.Tensor (clipped to valid pixel range)
    """
    # Convert to float32 numpy for PCA computation
    img = tf.cast(image, tf.float32).numpy()

    # Reshape to (N, 3) where N = H * W
    h, w, c = img.shape
    pixels = img.reshape(-1, c)

    # Compute mean and center the data
    mean = pixels.mean(axis=0)
    pixels_centered = pixels - mean

    # Compute 3x3 covariance matrix of RGB channels
    cov = np.cov(pixels_centered, rowvar=False)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # AlexNet perturbation: delta = P * (alpha * lambda)
    alphas = np.array(alphas)
    perturbation = eigenvectors @ (alphas * eigenvalues)

    # Add perturbation to every pixel
    img_augmented = img + perturbation

    # Clip to valid range and return as tf.Tensor
    img_augmented = np.clip(img_augmented, 0, 255).astype(np.uint8)
    return tf.convert_to_tensor(img_augmented)
