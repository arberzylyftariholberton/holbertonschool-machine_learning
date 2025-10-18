#!/usr/bin/env python3
"""A script that performs element-wise operations"""


def np_elementwise(mat1, mat2):
    """
    A function that returns element-wise operations
    """

    addition = mat1 + mat2
    substraction = mat1 - mat2
    multiplication = mat1 * mat2
    division = mat1 / mat2
    return addition, substraction, multiplication, division
