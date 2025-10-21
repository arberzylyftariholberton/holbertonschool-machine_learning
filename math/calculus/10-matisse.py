#!/usr/bin/env python3
""" A script that finds the derivative of a polynomial """


def poly_derivative(poly):
    """
    A function that returns the derivative of a polynomial
    By returning it in a new list
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for x in range(1, len(poly)):
        derivative.append(x * poly[x])

    return derivative
