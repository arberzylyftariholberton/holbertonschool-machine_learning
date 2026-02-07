#!/usr/bin/env python3
"""Module for RMSProp optimization"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm

    Args:
        alpha: learning rate
        beta2: RMSProp weight (decay rate)
        epsilon: small number to avoid division by zero
        var: numpy.ndarray containing the variable to be updated
        grad: numpy.ndarray containing the gradient of var
        s: previous second moment of var

    Returns:
        updated variable and the new moment, respectively
    """

    new_s = beta2 * s + (1-beta2) * (grad ** 2)

    updated_var = var - alpha * grad / (np.sqrt(new_s) + epsilon)

    return updated_var, new_s
