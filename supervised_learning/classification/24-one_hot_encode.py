#!/usr/bin/env python3
"""A script that converts a numeric label vector into a one-hot matrix."""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot encoded matrix.

    Parameters:
        Y (numpy.ndarray): Array of shape (m,) containing numeric class labels.
        classes (int): Total number of classes.

    Returns:
        numpy.ndarray: One-hot encoding of Y with shape (classes, m).
        None: If input validation fails.
    """
    if type(Y) is not np.ndarray or len(Y.shape) != 1:
        return None
    if type(classes) is not int or classes <= 0:
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))

    for i in range(m):
        if Y[i] >= classes or Y[i] < 0:
            return None
        one_hot[Y[i], i] = 1

    return one_hot
