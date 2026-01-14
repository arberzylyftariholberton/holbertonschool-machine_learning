#!/usr/bin/env python3
"""Neuron Forward Propagation"""
import numpy as np


class Neuron:
    """
    A class that defines a single neuron performing binary
    classification having private instances atributes and a
    forward propagation function
    """

    def __init__(self, nx):
        """
        A constructor that takes number of input as nx
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
        Getter function of the Weight
        """

        return self.__W

    @property
    def b(self):
        """
        Getter function of the bias
        """

        return self.__b

    @property
    def A(self):
        """
        Getter function of the activated output
        """

        return self.__A

    def forward_prop(self, X):
        """
        A function that calculates the forward propagation of the neuron
        """
        Z = np.matmul(self.__W, X) + self.__b

        self.__A = 1 / (np.exp(-Z))

        return self.__A
