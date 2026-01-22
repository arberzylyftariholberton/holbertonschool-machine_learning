#!/usr/bin/env python3
"""
    A script that performs binary classification
    in a Deep neural network
"""


import numpy as np


class DeepNeuralNetwork:
    """
    Deep neural network performing binary classification
    """

    def __init__(self, nx, layers):
        """
        A constructor that takes number of input as nx and
        layers is a list representing the number of nodes
        in each layer of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        prev = nx
        for layer in range(1, self.L + 1):
            nodes = layers[layer - 1]

            if type(nodes) is not int or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            self.weights["W{}".format(layer)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )
            self.weights["b{}".format(layer)] = np.zeros((nodes, 1))
            prev = nodes
