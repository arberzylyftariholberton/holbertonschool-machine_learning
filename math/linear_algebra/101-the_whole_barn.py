#!/usr/bin/env python3
"""A script that adds two matrices using Numpy"""


def add_matrices(mat1, mat2):
    """
    A function that returns the sum of two matrices using Numpy
    """

    # If they are numbers, just return their sum
    if not isinstance(mat1, list) and not isinstance(mat2, list):
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None

    result = []
    for i in range(len(mat1)):
        sum_m = add_matrices(mat1[i], mat2[i])
        if sum_m is None:
            return None
        result.append(sum_m)

    return result
