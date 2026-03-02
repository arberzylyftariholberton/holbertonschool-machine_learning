#!/usr/bin/env python3
"""Module that contains the function conv_forward
   for CNN forward propagation"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network

    Parameters:
    A_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
              containing the output of the previous layer
    W -- numpy.ndarray of shape (kh, kw, c_prev, c_new)
         containing the convolution kernels
    b -- numpy.ndarray of shape (1, 1, 1, c_new)
         containing the biases
    activation -- activation function applied to the convolution output
    padding -- string, either "same" or "valid"
    stride -- tuple of (sh, sw) containing the strides

    Returns:
    The activated output of the convolutional layer
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2))

    output_height = int((h_prev - kh + 2 * ph) / sh + 1)
    output_width = int((w_prev - kw + 2 * pw) / sw + 1)

    convolved_images = np.zeros((m, output_height, output_width, c_new))

    image_pad = np.pad(A_prev,
                       ((0, 0), (ph, ph),
                        (pw, pw), (0, 0)), mode='constant')

    for k in range(c_new):
        for h in range(output_height):
            for w in range(output_width):
                image_zone = image_pad[:, h * sh:h * sh + kh,
                                       w * sw:w * sw + kw, :]

                convolved_images[:, h, w, k] = np.sum(image_zone
                                                      * W[:, :, :, k],
                                                      axis=(1, 2, 3))

    Z = convolved_images + b

    Z = activation(Z)

    return Z
