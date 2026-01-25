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

        Private Attributes:
            __L (int): Number of layers in the neural network.
            __cache (dict): Dictionary to hold all intermediary values of the network, initialized empty.
            __weights (dict): Dictionary to hold all weights and biases of the network. Weights initialized
                              using He et al. method and biases initialized to zeros.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx
        for layer in range(1, self.__L + 1):
            nodes = layers[layer - 1]
            if type(nodes) is not int or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            self.__weights[f"W{layer}"] = np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            self.__weights[f"b{layer}"] = np.zeros((nodes, 1))
            prev = nodes

    @property
    def L(self):
        """Getter for the number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Getter for the dictionary holding all intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the dictionary holding all weights and biases of the network."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the deep neural network.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            A (numpy.ndarray): The output of the neural network.
            cache (dict): Dictionary containing all intermediary activations.
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]
            A_prev = self.__cache[f"A{layer - 1}"]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache[f"A{layer}"] = A

        return self.__cache[f"A{self.__L}"], self.__cache
