#!/usr/bin/env python3
"""
    A script that performs convolution with multiple kernels
"""

import numpy as np


def convolve(images, kernel, padding='same', stride=(1, 1)):
    """
        A function that performs a convolution on images using multiple kernel

        :param images: ndarray, shape(m, h, w, c), multiple images
        :param kernel: ndarray, shape(kh,kw,c,nc), kernel for convolution
        :param padding: tuple (ph,pw) or 'same" or "valid'
        :param stride: tuple (sh, sw)

        :return: ndarray containing convolved images
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int((((h - 1) * sh + kh - h) / 2) + 1)
        pw = int((((w - 1) * sw + kw - w) / 2) + 1)

    elif isinstance(padding, tuple):
        ph, pw = padding

    output_height = int((h - kh + 2 * ph) / sh + 1)
    output_width = int((w - kw + 2 * pw) / sw + 1)

    convolved_images = np.zeros((m, output_height, output_width, nc))

    image_pad = np.pad(images,
                       ((0, 0), (ph, ph),
                        (pw, pw), (0, 0)), mode='constant')

    for k in range(nc):
        for i in range(output_height):
            for j in range(output_width):
                image_zone = image_pad[:, i * sh:i * sh + kh,
                                       j * sw:j * sw + kw, :]

                convolved_images[:, i, j, k] = np.sum(image_zone
                                                      * kernel[:, :, :, k],
                                                      axis=(1, 2, 3))

    return convolved_images
