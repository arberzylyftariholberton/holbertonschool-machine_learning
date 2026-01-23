#!/usr/bin/env python3
"""A script that converts a numeric label vector into a one-hot matrix """
import numpy as np


def one_hot_encode(Y, classes):
    """
    A functiin that converts a numeric label vector into a one-hot matrix
    """
    if type(Y) is not np.ndarray or len(Y.shape) != 1:
        return None
    if type(classes) is not int or classes <= 0:
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))

    for i in range(m):
        if Y[i] >= classes:
            return None
        one_hot[Y[i], i] = 1

    return one_hot
