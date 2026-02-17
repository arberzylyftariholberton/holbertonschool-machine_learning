#!/usr/bin/env python3
"""Module for grayscale convolution with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding

    Args:
        images: numpy.ndarray shape (m, h, w) - grayscale images
        kernel: numpy.ndarray shape (kh, kw) - convolution kernel
        padding: tuple (ph, pw) - padding for height and width

    Returns:
        numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    oh = h + 2 * ph - kh + 1
    ow = w + 2 * pw - kw + 1

    padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), mode='constant')

    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = np.sum(
                padded[:, i:i + kh, j:j + kw] * kernel,
                axis=(1, 2)
            )

    return output
