#!/usr/bin/env python3
""" A script that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """
    A function that adds two matrices element-wise
    """

    if len(mat1) != len(mat2):
        return None

    for i in range(len(mat1)):
        if len(mat1[i]) != len(mat2[i]):
            return None

    result = []

    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2)):
            row.append(mat1[i][j] + mat2[i][j])
        result.append(row)

    return result
