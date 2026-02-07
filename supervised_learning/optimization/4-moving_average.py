#!/usr/bin/env python3
"""Module to calculate weighted moving average"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set

    Args:
        data: list of data to calculate the moving average of
        beta: weight used for the moving average

    Returns:
        list containing the moving averages of data
    """

    v = 0.0
    avgs = []

    for t, x, in enumerate(data, start=1):
        v = beta * v + (1-beta) * x
        v_corrected = v / (1 - (beta ** t))
        avgs.append(v_corrected)

    return avgs
