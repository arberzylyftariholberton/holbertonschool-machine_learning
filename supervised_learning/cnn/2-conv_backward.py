#!/usr/bin/env python3
"""Module that contains the function conv_backward for CNN backpropagation"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer

    Parameters:
    dZ -- numpy.ndarray of shape (m, h_new, w_new, c_new)
    A_prev -- numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    W -- numpy.ndarray of shape (kh, kw, c_prev, c_new)
    b -- numpy.ndarray of shape (1, 1, 1, c_new)
    padding -- 'same' or 'valid'
    stride -- tuple (sh, sw)

    Returns:
    dA_prev -- gradients w.r.t. A_prev (m, h_prev, w_prev, c_prev)
    dW -- gradients w.r.t. W (kh, kw, c_prev, c_new)
    db -- gradients w.r.t. b (1, 1, 1, c_new)
    """

    _, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    elif padding == 'same':
        ph = int((((h_prev - 1) * sh + kh - h_prev) / 2 + 0.5))
        pw = int((((w_prev - 1) * sw + kw - w_prev) / 2 + 0.5))

    A_prev_pad = np.pad(A_prev,
                        [(0, 0), (ph, ph), (pw, pw), (0, 0)],
                        mode='constant')

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    dA_pad = np.zeros(shape=A_prev_pad.shape)
    dW = np.zeros(shape=W.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for f in range(c_new):
                    v_start = h * sh
                    v_end = v_start + kh
                    h_start = w * sw
                    h_end = h_start + kw

                    dA_pad[i, v_start:v_end, h_start:h_end, :]\
                        += W[:, :, :, f] * dZ[i, h, w, f]
                    dW[:, :, :, f] += (A_prev_pad[i, v_start:v_end,
                                                  h_start:h_end, :]
                                       * dZ[i, h, w, f])

    if padding == "same":
        dA = dA_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA = dA_pad

    return dA, dW, db
