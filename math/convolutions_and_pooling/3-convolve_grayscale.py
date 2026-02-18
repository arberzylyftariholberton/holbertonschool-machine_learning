#!/usr/bin/env python3
"""
    A script that performs a strided convolution on grayscale images
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
        A function that performs a convolution on grayscale images

        :param images: ndarray, shape(m, h, w), multiple grayscale images
        :param kernel: ndarray, shape(kh,kw), kernel for convolution
        :param padding: tuple (ph,pw) and 'same" or "valid'
        :param stride: tuple (sh, sw)

        :return: ndarray containing convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int((((h - 1) * sh + kh - h)/2) + 1)
        pw = int((((w - 1) * sw + kw - w)/2) + 1)

    elif isinstance(padding, tuple):
        ph, pw = padding

    output_height = int((h - kh + 2 * ph) / sh + 1)
    output_width = int((w - kw + 2 * pw) / sw + 1)

    convolved_images = np.zeros((m, output_height, output_width))

    image_pad = np.pad(images,
                       ((0, 0), (ph, ph),
                        (pw, pw)), mode='constant')

    for i in range(output_height):
        for j in range(output_width):
            image_zone = image_pad[:, i * sh:i * sh + kh, j * sw:j * sw + kw]

            convolved_images[:, i, j] = np.sum(image_zone * kernel,
                                               axis=(1, 2))

    return convolved_images
