#!/usr/bin/env python3
"""Privatizing a Neural Network binary classification"""

import numpy as np


class NeuralNetwork:
    """
    A class that defines a Neural Network performing binary classification,
    adding private instance attributes
    """

    def __init__(self, nx, nodes):
        """
        A constructor that takes number of input as nx and
        nodes is the number of nodes found in the hidden layer
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
            Getter function of the Weight vector for the hidden layer
            """
            return self.__W1

        @property
        def b1(self):
            """
            Getter function of The bias for the hidden layer
            """
            return self.__b1

        @property
        def A1(self):
            """
            Getter function of The activated output for the hidden layer
            """
            return self.__A1

        @property
        def W2(self):
            """
            Getter function of the Weight vector for the output neuron
            """
            return self.__W2

        @property
        def b2(self):
            """
            Getter function of The bias for the output neuron
            """
            return self.__b2

        @property
        def A2(self):
            """
            Getter function of The activated output for the output neuron
            """
            return self.__A2
