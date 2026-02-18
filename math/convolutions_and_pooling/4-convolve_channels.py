#!/usr/bin/env python3
"""Module for convolution with channels"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels

    Args:
        images: numpy.ndarray shape (m, h, w, c) - images with channels
        kernel: numpy.ndarray shape (kh, kw, c) - convolution kernel
        padding: 'same', 'valid', or tuple (ph, pw)
        stride: tuple (sh, sw)

    Returns:
        numpy.ndarray shape (m, oh, ow) - convolved images (no channel dim)
    """
    m, h, w, c = images.shape
    kh, kw, c = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = kh // 2
        pw = kw // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1

    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    mode='constant')

    output = np.zeros((m, oh, ow))

    for i in range(oh):
        for j in range(ow):
            output[:, i, j] = np.sum(
                padded[:, i * sh:i * sh + kh, j * sw:j * sw + kw, :] * kernel,
                axis=(1, 2, 3)
            )

    return output
