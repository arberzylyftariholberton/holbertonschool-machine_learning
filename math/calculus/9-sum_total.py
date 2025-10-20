#!/usr/bin/env python3
""" A script that calculates the Sum of the series"""


def summation_i_squared(n):
    """
    A function that returns the Sum of the Series
    If n is invalid , returns None
    """
    if not isinstance(n, int) or n < 1:
        return None

    result = n * (n + 1) * (2 * n + 1) // 6

    return result
