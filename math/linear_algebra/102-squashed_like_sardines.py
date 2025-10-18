#!/usr/bin/env python3
""" A script that does concatenation of 2 matrices along a specific axis """


def cat_matrices(mat1, mat2, axis=0):
    """
    A function that returns the concatenation of
    2 matrices with a specific axis
    """

    if axis == 0:
        return mat1 + mat2

    if len(mat1) != len(mat2):
        return None

    result = []

    for sub_matrix1, sub_matrix2 in zip(mat1, mat2):
        merged = cat_matrices(sub_matrix1, sub_matrix2, axis=axis-1)
        if merged is None:
            return None
        result.append(merged)

    return result
