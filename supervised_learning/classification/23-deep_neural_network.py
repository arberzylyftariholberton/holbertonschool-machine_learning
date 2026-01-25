#!/usr/bin/env python3
"""
    A script that Upgrades the train function of the deep neural network
"""

import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """
    Deep neural network performing binary classification
    """
    def __init__(self, nx, layers):
        """
        Initializes a DeepNeuralNetwork instance.

        Parameters:
            nx (int): Number of input features.
            layers (list): Number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx < 1.
            TypeError: If layers is not a list of positive integers.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        prev = nx
        for layer in range(1, self.__L + 1):
            nodes = layers[layer - 1]
            if type(nodes) is not int or nodes <= 0:
                raise TypeError("layers must be a list of positive integers")

            self.__weights[f"W{layer}"] = np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            self.__weights[f"b{layer}"] = np.zeros((nodes, 1))
            prev = nodes

    @property
    def L(self):
        """Number of layers in the neural network."""
        return self.__L

    @property
    def cache(self):
        """Dictionary to hold all intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Dictionary to hold all weights and biases of the network."""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates forward propagation of the neural network.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            A (numpy.ndarray): Activated output of the last layer.
            cache (dict): Dictionary containing all intermediary activations.
        """
        self.__cache["A0"] = X
        for layer in range(1, self.__L + 1):
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]
            A_prev = self.__cache[f"A{layer - 1}"]
            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f"A{layer}"] = A
        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Parameters:
            Y (numpy.ndarray): Correct labels of shape (1, m).
            A (numpy.ndarray): Activated output of shape (1, m).

        Returns:
            cost (float): Logistic regression cost.
        """
        m = Y.shape[1]
        return -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).

        Returns:
            prediction (numpy.ndarray): Predicted labels (0 or 1).
            cost_value (float): Cost of the network.
        """
        A, _ = self.forward_prop(X)
        prediction = (A >= 0.5).astype(int)
        cost_value = self.cost(Y, A)
        return prediction, cost_value

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the network.

        Parameters:
            Y (numpy.ndarray): Correct labels of shape (1, m).
            cache (dict): Dictionary containing all intermediary activations.
            alpha (float): Learning rate.
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dZ = None

        for layer in range(self.__L, 0, -1):
            A = cache[f"A{layer}"]
            A_prev = cache[f"A{layer - 1}"]
            if layer == self.__L:
                dZ = A - Y
            else:
                W_next = weights_copy[f"W{layer + 1}"]
                dZ = np.matmul(W_next.T, dZ) * (A * (1 - A))

            dW = (1/m) * np.matmul(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights[f"W{layer}"] = weights_copy[f"W{layer}"] - alpha * dW
            self.__weights[f"b{layer}"] = weights_copy[f"b{layer}"] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network.

        Parameters:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): Correct labels of shape (1, m).
            iterations (int): Number of iterations to train over.
            alpha (float): Learning rate.
            verbose (bool): If True, prints cost every step iterations.
            graph (bool): If True, plots cost vs iteration.
            step (int): Steps for printing and plotting.

        Returns:
            prediction (numpy.ndarray): Predicted labels.
            cost_value (float): Cost of the network.
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
        iteration_list = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if i == 0 or i == iterations or i % step == 0:
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")
                if graph:
                    costs.append(cost)
                    iteration_list.append(i)

            if i < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(iteration_list, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
