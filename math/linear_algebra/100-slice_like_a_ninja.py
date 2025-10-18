#!/usr/bin/env python3
"""A script that slices a NumPy matrix along specific axes"""
import numpy as np


def np_slice(matrix, axes={}):
    """
    A function that slices a NumPy matrix along specific axes
    """

    slices = [slice(None)] * matrix.ndim

    for axis_num, slice_tuple in axes.items():
        slices[axis_num] = slice(*slice_tuple)

    return matrix[tuple(slices)]
