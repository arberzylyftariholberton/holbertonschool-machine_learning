#!/usr/bin/env python3
"""Module for convolution with multiple kernels"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels

    Args:
        images:  numpy.ndarray shape (m, h, w, c)
        kernels: numpy.ndarray shape (kh, kw, c, nc)
        padding: 'same', 'valid', or tuple (ph, pw)
        stride:  tuple (sh, sw)

    Returns:
        numpy.ndarray shape (m, oh, ow, nc)
    """
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
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

    output = np.zeros((m, oh, ow, nc))

    for k in range(nc):
        for i in range(oh):
            for j in range(ow):
                output[:, i, j, k] = np.sum(
                    padded[:,
                           i * sh:i * sh + kh,
                           j * sw:j * sw + kw,
                           :] * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )

    return output
