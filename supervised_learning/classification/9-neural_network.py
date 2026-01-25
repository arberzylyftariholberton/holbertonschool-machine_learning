#!/usr/bin/env python3
"""Defines a Neural Network for binary classification with private attributes."""
import numpy as np


class NeuralNetwork:
    """
    Represents a neural network with one hidden layer for binary classification,
    using private instance attributes for encapsulation.
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

        Private Attributes:
            __W1 (numpy.ndarray): Weights for the hidden layer, initialized randomly.
            __b1 (numpy.ndarray): Biases for the hidden layer, initialized to zeros.
            __A1 (float): Activated outputs for the hidden layer, initialized to 0.
            __W2 (numpy.ndarray): Weights for the output neuron, initialized randomly.
            __b2 (float): Bias for the output neuron, initialized to 0.
            __A2 (float): Activated output for the output neuron (prediction), initialized to 0.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter for the weight matrix of the hidden layer.
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter for the bias vector of the hidden layer.
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter for the activated output of the hidden layer.
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter for the weight vector of the output neuron.
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter for the bias of the output neuron.
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter for the activated output (prediction) of the output neuron.
        """
        return self.__A2
