#!/usr/bin/env python3
"""Defines a Neural Network for binary classification with one hidden layer."""
import numpy as np


class NeuralNetwork:
    """
    Represents a neural network with a single hidden layer for binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initializes a NeuralNetwork instance with one hidden layer.

        Parameters:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If nodes is not an integer.
            ValueError: If nodes is less than 1.

        Public Attributes:
            W1 (numpy.ndarray): Weights for the hidden layer, initialized randomly.
            b1 (numpy.ndarray): Biases for the hidden layer, initialized to zeros.
            A1 (float): Activated outputs for the hidden layer, initialized to 0.
            W2 (numpy.ndarray): Weights for the output neuron, initialized randomly.
            b2 (float): Bias for the output neuron, initialized to 0.
            A2 (float): Activated output for the output neuron (prediction), initialized to 0.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
