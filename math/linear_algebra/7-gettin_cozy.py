#!/usr/bin/env python3
"""A script that does concatenation of 2 matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    A function that returns the concatenation of 2 matrices with specific axis
    """

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None

        result = []
        for row in mat1:
            result.append(row.copy())
        for row in mat2:
            result.append(row.copy())
        return result

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None

        result = []
        for i in range(len(mat1)):
            row2 = mat1[i].copy()
            row2.extend(mat2[i])
            result.append(row2)
        return result

    else:
        return None
