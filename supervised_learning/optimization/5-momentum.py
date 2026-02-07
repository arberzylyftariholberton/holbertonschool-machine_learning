#!/usr/bin/env python3
"""Module for gradient descent with momentum optimization"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent
    with momentum optimization algorithm

    Args:
        alpha: learning rate
        beta1: momentum weight
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        v: previous first moment of var

    Returns:
        updated variable and the new moment, respectively
    """

    updated_v = beta1 * v + (1 - beta1) * grad

    updated_var = var - (alpha * updated_v)

    return updated_var, updated_v
