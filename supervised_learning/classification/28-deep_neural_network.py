#!/usr/bin/env python3
"""
Deep Neural Network module for multiclass classification
with flexible hidden layer activations (sigmoid or tanh)
"""

import matplotlib.pyplot as plt
import pickle
import os
import numpy as np


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing multiclass classification.

    Attributes:
        __L (int): Number of layers in the network.
        __cache (dict): Dictionary to store all intermediary values of the network.
        __weights (dict): Dictionary to store all weights and biases.
        __activation (str): Activation function used in hidden layers ('sig' or 'tanh').
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Constructor for DeepNeuralNetwork.

        Args:
            nx (int): Number of input features.
            layers (list): List of nodes per layer in the network.
            activation (str): Activation function for hidden layers ('sig' or 'tanh').

        Raises:
            TypeError: If nx is not an integer or layers is not a list of positive integers.
            ValueError: If nx < 1 or activation is invalid.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation

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
        """Getter for number of layers in the network."""
        return self.__L

    @property
    def cache(self):
        """Getter for dictionary storing intermediary values of the network."""
        return self.__cache

    @property
    def weights(self):
        """Getter for dictionary storing all weights and biases of the network."""
        return self.__weights

    @property
    def activation(self):
        """Getter for the activation function used in hidden layers."""
        return self.__activation

    def forward_prop(self, X):
        """
        Calculates the forward propagation through the network.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m), where m is the number of examples.

        Returns:
            tuple: Output of the network (A_L) and the cache dictionary.

        Notes:
            - Hidden layers use the activation function specified in __activation.
            - Output layer uses softmax for multiclass predictions.
            - Activations of all layers are stored in __cache.
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]
            A_prev = self.__cache[f"A{layer - 1}"]

            Z = np.matmul(W, A_prev) + b

            if layer != self.__L:
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:
                    A = np.tanh(Z)
            else:
                Z_exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = Z_exp / np.sum(Z_exp, axis=0, keepdims=True)

            self.__cache[f"A{layer}"] = A

        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        Computes the cost of the model using multiclass cross-entropy.

        Args:
            Y (numpy.ndarray): One-hot true labels of shape (classes, m).
            A (numpy.ndarray): Output of the network (softmax probabilities).

        Returns:
            float: Cost of the network.
        """
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates predictions of the network and calculates the cost.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): One-hot true labels of shape (classes, m).

        Returns:
            tuple: Predicted labels (one-hot) and cost.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        predictions = np.zeros_like(A)
        predictions[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent to update network weights.

        Args:
            Y (numpy.ndarray): True labels of shape (classes, m).
            cache (dict): Dictionary with intermediate activations.
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
                if self.__activation == 'sig':
                    dZ = np.matmul(W_next.T, dZ) * (A * (1 - A))
                else:
                    dZ = np.matmul(W_next.T, dZ) * (1 - A ** 2)

            dW = (1/m) * np.matmul(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights[f"W{layer}"] = weights_copy[f"W{layer}"] - alpha * dW
            self.__weights[f"b{layer}"] = weights_copy[f"b{layer}"] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Trains the deep neural network using gradient descent.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): One-hot true labels of shape (classes, m).
            iterations (int): Number of iterations to train.
            alpha (float): Learning rate.
            verbose (bool): If True, prints cost every 'step' iterations.
            graph (bool): If True, plots cost vs iteration graph.
            step (int): Step interval for printing and plotting.

        Returns:
            tuple: Final predictions and cost after training.
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
        """
        Saves the instance object to a file in pickle format.

        Args:
            filename (str): File path to save the object. Adds '.pkl' if missing.
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object.

        Args:
            filename (str): File path of the pickled object.

        Returns:
            DeepNeuralNetwork or None: Loaded object, or None if file doesn't exist.
        """
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
