#!/usr/bin/env python3
"""A script that converts a one-hot matrix into a vector of labels."""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot encoded matrix into a vector of numeric class labels.

    Parameters:
        one_hot (numpy.ndarray): One-hot encoded array with shape (classes, m).

    Returns:
        numpy.ndarray: Array of shape (m,) containing numeric labels for each example.
        None: If input validation fails.
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None

    return np.argmax(one_hot, axis=0)
