#!/usr/bin/env python3
"""A script that transposes a matrix"""


def matrix_transpose(matrix):
    """A function that transposes a matrix"""
    rows = len(matrix)
    cols = len(matrix[0])
    transposed = [[0] * rows for _ in range(cols)]

    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]

    return transposed
