#!/usr/bin/env python3
"""A script that trains a neuron"""
import numpy as np


class Neuron:
    """
    A class that defines a single neuron performing binary
    classification having private instances atributes,
    a forward propagation function,
    a cost calculation function using logistic regression,
    an evaluation of the neuron's predictions function
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

        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A

    def cost(self, Y, A):
        """
        A function that calculates the cost of the model
        using logistic regression
        """

        m = Y.shape[1]

        log_loss = -1/m*np.sum(Y * np.log(A) + (1 - Y)*(np.log(1.0000001 - A)))

        return log_loss

    def evaluate(self, X, Y):
        """
        A function that evaluates the neuron's predictions
        """
        A = self.forward_prop(X)
        c = self.cost(Y, A)
        result = np.where(A >= 0.5, 1, 0)

        return result, c

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        A function that calculates one pass of gradient descent on the neuron
        and updates the private attributes __W and __b.
        """

        m = Y.shape[1]

        grad_w = 1 / m * np.matmul((A - Y), X.T)
        grad_b = 1 / m * np.sum((A-Y))

        self.__W = self.__W - alpha * grad_w
        self.__b = self.__b - alpha * grad_b

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        A function that trains a neuron using Forward propagation
        and Gradient descent
        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be a positive")

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
