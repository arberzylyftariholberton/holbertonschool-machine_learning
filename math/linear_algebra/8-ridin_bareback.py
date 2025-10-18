#!/usr/bin/env python3
"""A script that does matrix multiplication"""


def mat_mul(mat1, mat2):
    """
    A functions that returns the multiplication of 2 matrics
    """

    if len(mat1[0]) != len(mat2):
        return None

    result = []

    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            mul = 0
            for m in range(len(mat2)):
                mul += mat1[i][m] * mat2[m][j]
            row.append(mul)
        result.append(row)

    return result
