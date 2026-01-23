#!/usr/bin/env python3
"""
    A script that Updates the Deep Neural Network With
    methods save() and load()
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork:
    """
    Deep neural network performing binary classification
    """
    def __init__(self, nx, layers):
        """
        A constructor that takes number of input as nx and
        layers is a list representing the number of nodes
        in each layer of the network
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

            self.__weights["W{}".format(layer)] = (
                np.random.randn(nodes, prev) * np.sqrt(2 / prev)
            )
            self.__weights["b{}".format(layer)] = np.zeros((nodes, 1))
            prev = nodes

    @property
    def L(self):
        """
        Getter function for The number of layers in the neural network
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter function dictionary to hold all
        intermediary values of the network
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter function dictionary to hold all weights
        and biased of the network
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Function that Calculates the forward propagation
        of the neural network
        """
        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights[f"W{layer}"]
            b = self.__weights[f"b{layer}"]

            A_prev = self.__cache[f"A{layer-1}"]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache[f"A{layer}"] = A

        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        A function that Calculates the cost of the model
        using logistic regression
        """
        m = Y.shape[1]

        log_loss = -1/m*np.sum(Y * np.log(A) + (1-Y)*(np.log(1.0000001-A)))

        return log_loss

    def evaluate(self, X, Y):
        """
        A function that Evaluates the neural network's predictions
        """
        A, _ = self.forward_prop(X)

        prediction = (A >= 0.5).astype(int)

        cost_value = self.cost(Y, A)

        return prediction, cost_value

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        A function that calculates one pass of
        gradient descent on the neural network
        """
        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dZ = None

        for layer in range(self.__L, 0, -1):
            A = cache[f"A{layer}"]
            A_prev = cache[f"A{layer-1}"]

            if layer == self.__L:
                dZ = A - Y
            else:
                W_next = weights_copy[f"W{layer + 1}"]
                dZ = np.matmul(W_next.T, dZ) * (A * (1 - A))

            dW = (1/m) * np.matmul(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights[f"W{layer}"] = (
                weights_copy[f"W{layer}"] - alpha * dW
            )
            self.__weights[f"b{layer}"] = (
                weights_copy[f"b{layer}"] - alpha * db
            )

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        A function that upgrades the train function
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
        A function that saves the instance object to a file in pickle format
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A function that loads a pickled DeepNeuralNetwork object
        """
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
