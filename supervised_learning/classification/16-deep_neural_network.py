#!/usr/bin/env python3
"""Defines a Deep Neural Network for binary classification."""
import numpy as np


class DeepNeuralNetwork:
    """
    Represents a deep neural network performing binary classification.
    """

    def __init__(self, nx, layers):
        """
        Initializes a DeepNeuralNetwork instance.

        Parameters:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If layers is not a list of positive integers or is empty.

        Public Attributes:
            L (int): Number of layers in the neural network.
            cache (dict): Dictionary to hold all intermediary values of the network, initialized empty.
            weights (dict): Dictionary to hold all weights and biases of the network. Weights initialized
                            using He et al. method and biases initialized to zeros.
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
