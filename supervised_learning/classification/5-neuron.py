#!/usr/bin/env python3
"""
Defines a Neuron class with forward propagation, cost calculation,
prediction evaluation, and gradient descent for binary classification.
"""
import numpy as np


class Neuron:
    """
    Represents a single neuron for binary classification with:
    - Private instance attributes
    - Forward propagation
    - Logistic regression cost computation
    - Evaluation of predictions
    - Gradient descent optimization
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
        Performs forward propagation using a sigmoid activation function.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            numpy.ndarray: Activated output of the neuron.
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Computes the logistic regression cost of the neuron.

        Parameters:
            Y (numpy.ndarray): Correct labels of shape (1, m).
            A (numpy.ndarray): Activated outputs of shape (1, m).

        Returns:
            float: Logistic regression cost.
        """
        m = Y.shape[1]
        log_loss = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return log_loss

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions against true labels.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).

        Returns:
            tuple:
                - numpy.ndarray: Predicted labels (0 or 1) of shape (1, m).
                - float: Logistic regression cost of the predictions.
        """
        A = self.forward_prop(X)
        c = self.cost(Y, A)
        result = np.where(A >= 0.5, 1, 0)
        return result, c

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Performs one step of gradient descent to update the neuron's parameters.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).
            A (numpy.ndarray): Activated outputs of the neuron.
            alpha (float): Learning rate (default 0.05).

        Updates:
            __W: The weights vector.
            __b: The bias.
        """
        m = Y.shape[1]
        grad_w = 1 / m * np.matmul((A - Y), X.T)
        grad_b = 1 / m * np.sum(A - Y)

        self.__W = self.__W - alpha * grad_w
        self.__b = self.__b - alpha * grad_b
