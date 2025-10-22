#!/usr/bin/env python3
""" A script that finds the integral of a polynomial """

def poly_integral(poly, C=0):
    """
    A function that returns the integral of a polynomial
    in a list of coefficients (poly).
    """

    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not isinstance(C, (int, float)):
        return None

    for coefficient in poly:
        if not isinstance(coefficient, (int, float)):
            return None

    integral = [C]

    for i in range(len(poly)):
        x = poly[i] / (i + 1)
        if isinstance(x, float) and x.is_integer():
            x = int(x)
        integral.append(x)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
