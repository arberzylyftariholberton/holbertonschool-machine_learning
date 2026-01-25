#!/usr/bin/env python3
"""
   A Script That Upgrades The Training Of The Neural Network
"""

import numpy as np
import matplotlib.pyplot as plt  # Required for graphing training cost


class NeuralNetwork:
    """
    Defines a Neural Network with one hidden layer for binary classification.
    Includes private attributes for weights, biases, and activated outputs.
    Provides methods for forward propagation, cost calculation, evaluation,
    gradient descent, and enhanced training with verbose output and cost graphing.
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
        """Getter for hidden layer weights."""
        return self.__W1

    @property
    def b1(self):
        """Getter for hidden layer biases."""
        return self.__b1

    @property
    def A1(self):
        """Getter for hidden layer activated output."""
        return self.__A1

    @property
    def W2(self):
        """Getter for output neuron weights."""
        return self.__W2

    @property
    def b2(self):
        """Getter for output neuron bias."""
        return self.__b2

    @property
    def A2(self):
        """Getter for output neuron activated output."""
        return self.__A2

    def forward_prop(self, X):
        """Calculates forward propagation for the neural network."""
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the logistic regression cost of predictions."""
        m = Y.shape[1]
        log_loss = -1/m * np.sum(Y * np.log(A) + (1-Y) * np.log(1.0000001 - A))
        return log_loss

    def evaluate(self, X, Y):
        """
        Evaluates the network's predictions.

        Returns:
            tuple: (predicted labels, cost)
        """
        _, A2 = self.forward_prop(X)
        prediction = (A2 >= 0.5).astype(int)
        cost_value = self.cost(Y, A2)
        return prediction, cost_value

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Performs one pass of gradient descent on the network."""
        m = Y.shape[1]
        grad_w2 = 1/m * np.matmul((A2-Y), A1.T)
        grad_b2 = 1/m * np.sum((A2-Y), axis=1, keepdims=True)
        grad_w1 = 1/m * np.matmul((np.matmul(self.__W2.T, (A2-Y)) * (A1*(1-A1))), X.T)
        grad_b1 = 1/m * np.sum((np.matmul(self.__W2.T, (A2-Y)) * (A1*(1-A1))),
                               axis=1, keepdims=True)

        self.__W1 -= alpha * grad_w1
        self.__W2 -= alpha * grad_w2
        self.__b1 -= alpha * grad_b1
        self.__b2 -= alpha * grad_b2

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the neural network using forward propagation and gradient descent.

        Parameters:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray): Correct labels.
            iterations (int): Number of iterations.
            alpha (float): Learning rate.
            verbose (bool): Print cost every `step` iterations.
            graph (bool): Plot cost curve after training.
            step (int): Steps interval for verbose output and graphing.

        Returns:
            tuple: Evaluation of training data (prediction, cost).
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []
        iters = []

        _, A2 = self.forward_prop(X)
        c = self.cost(Y, A2)
        if verbose or graph:
            costs.append(c)
            iters.append(0)
            if verbose:
                print(f"Cost after 0 iterations: {c}")

        for i in range(1, iterations + 1):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

            if (verbose or graph) and (i % step == 0 or i == iterations):
                _, A2 = self.forward_prop(X)
                c = self.cost(Y, A2)
                costs.append(c)
                iters.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {c}")

        if graph:
            plt.plot(iters, costs, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
