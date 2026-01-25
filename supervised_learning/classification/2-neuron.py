#!/usr/bin/env python3
"""
Defines a Neuron class with forward propagation for binary classification.
"""
import numpy as np


class Neuron:
    """
    Defines a single neuron that performs binary classification
    with private instance attributes and forward propagation.
    """

    def __init__(self, nx):
        """
        Initializes a Neuron instance.

        Parameters:
            nx (int): Number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        Retrieves the weights vector of the neuron.
        """

        return self.__W

    @property
    def b(self):
        """
        Retrieves the bias value of the neuron.
        """

        return self.__b

    @property
    def A(self):
        """
        Retrieves the activated output of the neuron.
        """

        return self.__A

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron using
        a sigmoid activation function.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            numpy.ndarray: The activated output of the neuron.
        """
        Z = np.matmul(self.__W, X) + self.__b

        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A
