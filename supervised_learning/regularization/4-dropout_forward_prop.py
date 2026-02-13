#!/usr/bin/env python3
"""Module for forward propagation with Dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout

    Args:
        X: numpy.ndarray of shape (nx, m) containing the input data
           nx is the number of input features
           m is the number of data points
        weights: dictionary of the weights and biases of the neural network
        L: number of layers in the network
        keep_prob: probability that a node will be kept

    All layers except the last use tanh activation.
    The last layer uses softmax activation.

    Returns:
        dictionary containing the outputs of each layer and the dropout
        mask used on each layer
    """

    cache = {}
    cache['A0'] = X

    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]

        A_prev = cache['A' + str(layer - 1)]
        Z = np.matmul(W, A_prev) + b

        if layer == L:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            cache['A' + str(layer)] = A
        else:

            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            D = D.astype(int)

            A = A * D
            A = A / keep_prob

            cache['A' + str(layer)] = A
            cache['D' + str(layer)] = D

    return cache
