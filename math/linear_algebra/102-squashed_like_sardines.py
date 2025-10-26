#!/usr/bin/env python3
""" A script that concatenates two matrices along a specific axis """


def matrix_shape(matrix):
    """
    Returns the shape of a matrix as a list
    """

    if not isinstance(matrix, list):
        return []

    shape = [len(matrix)]

    if len(matrix) > 0 and isinstance(matrix[0], list):
        shape.extend(matrix_shape(matrix[0]))

    return shape


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    Returning concatenated matrix or None if cannot concatenate
    """

    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)

    if len(shape1) != len(shape2):
        return None

    if axis < 0 or axis >= len(shape1):
        return None

    for i in range(len(shape1)):
        if i != axis and shape1[i] != shape2[i]:
            return None

    if axis == 0:
        if not isinstance(mat1[0], list):
            return mat1 + mat2
        else:
            return [row[:] for row in mat1] + [row[:] for row in mat2]

    else:
        result = []
        for i in range(len(mat1)):
            concatenated = cat_matrices(mat1[i], mat2[i], axis - 1)
            if concatenated is None:
                return None
            result.append(concatenated)
        return result
