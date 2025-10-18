#!/usr/bin/env python3
""" A script that concatenates 2 matrices along a specific axis """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    A function that returns concatenation of two matrices along a specific axis
    """

    return np.concatenate((mat1, mat2), axis=axis)
