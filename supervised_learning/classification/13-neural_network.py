#!/usr/bin/env python3
"""
   A script that Calculates one pass of Gradient
   Descent on the neural network
"""

import numpy as np


class NeuralNetwork:
    """
    A class that defines a Neural Network performing binary classification,
    adding private instance attributes for weights, biases, and activated outputs.
    Includes methods for forward propagation, cost calculation, evaluation,
    and one step of gradient descent.
    """

    def __init__(self, nx, nodes):
        """
        Initializes a NeuralNetwork instance with one hidden layer.

        Parameters:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes is not an integer.
            ValueError: If nx or nodes is less than 1.

        Private Attributes:
            __W1 (numpy.ndarray): Weights for the hidden layer.
            __b1 (numpy.ndarray): Biases for the hidden layer.
            __A1 (float): Activated output of the hidden layer.
            __W2 (numpy.ndarray): Weights for the output neuron.
            __b2 (float): Bias for the output neuron.
            __A2 (float): Activated output (prediction) of the output neuron.
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
        """Getter function for the weight matrix of the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Getter function for the bias vector of the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Getter function for the activated output of the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Getter function for the weight vector of the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Getter function for the bias of the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Getter function for the activated output of the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """
        Performs forward propagation through the neural network.

        Parameters:
            X (numpy.ndarray): Input data with shape (nx, m).

        Updates:
            __A1: Activated output of the hidden layer using sigmoid.
            __A2: Activated output of the output neuron using sigmoid.

        Returns:
            tuple: (__A1, __A2)
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the neural network using logistic regression.

        Parameters:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output of the output neuron with shape (1, m).

        Returns:
            float: Logistic regression cost.
        """
        m = Y.shape[1]
        log_loss = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return log_loss

    def evaluate(self, X, Y):
        """
        Evaluates the neural network's predictions.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).

        Returns:
            tuple: (prediction, cost_value)
                - prediction (numpy.ndarray): Predicted labels (0 or 1).
                - cost_value (float): Logistic regression cost of predictions.
        """
        A1, A2 = self.forward_prop(X)
        prediction = (A2 >= 0.5).astype(int)
        cost_value = self.cost(Y, A2)
        return prediction, cost_value

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Performs one pass of gradient descent on the neural network.

        Parameters:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A1 (numpy.ndarray): Activated output from the hidden layer.
            A2 (numpy.ndarray): Activated output from the output neuron.
            alpha (float): Learning rate.

        Updates:
            __W1, __b1: Weights and biases of the hidden layer.
            __W2, __b2: Weights and bias of the output neuron.
        """
        m = Y.shape[1]

        grad_w2 = 1/m * np.matmul((A2 - Y), A1.T)
        grad_b2 = 1/m * np.sum((A2 - Y), axis=1, keepdims=True)

        grad_w1 = 1/m * np.matmul((np.matmul(self.__W2.T, (A2 - Y)) * (A1 * (1 - A1))), X.T)
        grad_b1 = 1/m * np.sum((np.matmul(self.__W2.T, (A2 - Y)) * (A1 * (1 - A1))),
                               axis=1, keepdims=True)

        self.__W1 = self.__W1 - alpha * grad_w1
        self.__W2 = self.__W2 - alpha * grad_w2
        self.__b1 = self.__b1 - alpha * grad_b1
        self.__b2 = self.__b2 - alpha * grad_b2
