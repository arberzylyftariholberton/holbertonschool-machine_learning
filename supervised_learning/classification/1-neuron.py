#!/usr/bin/env python3
""" Privatizing A Binary Classification Neuron"""
import numpy as np


class Neuron:
    """
    A class that defines a single neuron performing binary
    classification having private instances atributes
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
