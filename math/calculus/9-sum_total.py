#!/usr/bin/env python3
""" A script that calculates the Sum of the series"""


def summation_i_squared(n):
    """
    A function that returns the Sum of the Series
    """
    if type(n) is not int:
        return None

    result = n**3/3 + n**2/2 + n/6

    return result
