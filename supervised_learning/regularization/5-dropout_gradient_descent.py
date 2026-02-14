#!/usr/bin/env python3
"""Module for gradient descent with dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) containing
           correct labels for the data
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs and dropout masks of each layer
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network

    All layers use tanh activation except the last, which uses softmax.
    The weights of the network should be updated in place.
    """
    m = Y.shape[1]

    A = cache['A' + str(L)]
    dZ = A - Y

    A_prev = cache['A' + str(L - 1)]
    W = weights['W' + str(L)]
    dW = np.matmul(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.matmul(W.T, dZ)

    weights['W' + str(L)] -= alpha * dW
    weights['b' + str(L)] -= alpha * db

    for layer in range(L - 1, 0, -1):
        D = cache['D' + str(layer)]
        dA = dA_prev * (D / keep_prob)

        A = cache['A' + str(layer)]
        A_prev = cache['A' + str(layer - 1)]
        dZ = dA * (1 - A ** 2)
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        W = weights['W' + str(layer)]
        dA_prev = np.matmul(W.T, dZ)

        weights['W' + str(layer)] -= alpha * dW
        weights['b' + str(layer)] -= alpha * db
