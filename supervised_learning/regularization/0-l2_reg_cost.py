#!/usr/bin/env python3
"""Module to calculate L2 regularization cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: cost of the network without L2 regularization
        lambtha: regularization parameter
        weights: dictionary of the weights and biases (numpy.ndarrays)
                 of the neural network
        L: number of layers in the neural network
        m: number of data points used

    Returns:
        cost of the network accounting for L2 regularization
    """
    l2_sum = 0

    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        l2_sum += np.sum(np.square(W))

    l2_regularization = (lambtha / (2 * m)) * l2_sum
    total_cost = cost + l2_regularization

    return total_cost
