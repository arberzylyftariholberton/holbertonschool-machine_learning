#!/usr/bin/env python3
"""
    Deep Neural Network module for multiclass classification
"""

import matplotlib.pyplot as plt
import pickle
import os
import numpy as np


class DeepNeuralNetwork:
    """
    Deep neural network performing multiclass classification
    """
    def __init__(self, nx, layers):
        """
        Constructor for a deep neural network.

        Parameters:
            nx (int): number of input features
            layers (list): list containing the number of nodes in each layer

        Raises:
            TypeError: if nx is not an integer or layers is not a list of positive integers
            ValueError: if nx < 1 or any layer size <= 0
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

            # He initialization for weights
            self.__weights[f"W{layer}"] = np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            # Bias initialized to zeros
            self.__weights[f"b{layer}"] = np.zeros((nodes, 1))
            prev = nodes

    @property
    def L(self):
        """Number of layers in the network"""
        return self.__L

    @property
    def cache(self):
        """Dictionary holding intermediary values of the network"""
        return self.__cache

    @property
    def weights(self):
        """Dictionary holding weights and biases"""
        return self.__weights

    def forward_prop(self, X):
        """
        Performs forward propagation through the network using sigmoid for
        hidden layers and softmax for the output layer (multiclass).

        Parameters:
            X (numpy.ndarray): input data of shape (nx, m)

        Returns:
            A (numpy.ndarray): activated output of the last layer (softmax)
            cache (dict): dictionary of all layer activations
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]
            A_prev = self.__cache[f"A{layer - 1}"]

            Z = np.matmul(W, A_prev) + b

            if layer != self.__L:
                # Hidden layers use sigmoid activation
                A = 1 / (1 + np.exp(-Z))
            else:
                # Output layer uses softmax for multiclass
                Z_exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = Z_exp / np.sum(Z_exp, axis=0, keepdims=True)

            self.__cache[f"A{layer}"] = A

        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the network using multiclass cross-entropy.

        Parameters:
            Y (numpy.ndarray): one-hot labels of shape (classes, m)
            A (numpy.ndarray): softmax output of shape (classes, m)

        Returns:
            cost (float): cross-entropy cost
        """
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A)) / m
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates predictions of the network.

        Parameters:
            X (numpy.ndarray): input data
            Y (numpy.ndarray): one-hot labels

        Returns:
            predictions (numpy.ndarray): one-hot predicted labels
            cost (float): multiclass cross-entropy cost
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        # Convert softmax output to one-hot predictions
        predictions = np.zeros_like(A)
        predictions[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the network.

        Parameters:
            Y (numpy.ndarray): one-hot labels
            cache (dict): dictionary of activations
            alpha (float): learning rate

        Updates:
            self.__weights
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dZ = None

        for layer in range(self.__L, 0, -1):
            A = cache[f"A{layer}"]
            A_prev = cache[f"A{layer - 1}"]

            if layer == self.__L:
                # Output layer gradient for softmax + cross-entropy
                dZ = A - Y
            else:
                W_next = weights_copy[f"W{layer + 1}"]
                dZ = np.matmul(W_next.T, dZ) * (A * (1 - A))

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            self.__weights[f"W{layer}"] = weights_copy[f"W{layer}"] - alpha * dW
            self.__weights[f"b{layer}"] = weights_copy[f"b{layer}"] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network for multiclass classification.
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

    def save(self, filename):
        """Saves the DeepNeuralNetwork instance to a file in pickle format."""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Loads a pickled DeepNeuralNetwork object, or returns None if file doesn't exist."""
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
