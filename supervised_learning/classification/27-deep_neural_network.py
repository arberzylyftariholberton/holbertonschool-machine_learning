#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import os
import numpy as np


class DeepNeuralNetwork:

    def __init__(self, nx, layers):

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

        return self.__L

    @property
    def cache(self):

        return self.__cache

    @property
    def weights(self):

        return self.__weights

    def forward_prop(self, X):

        self.__cache["A0"] = X

        for layer in range(1, self.__L + 1):
            W = self.__weights["W{}".format(layer)]
            b = self.__weights["b{}".format(layer)]
            A_prev = self.__cache["A{}".format(layer - 1)]

            Z = np.matmul(W, A_prev) + b

            if layer != self.__L:
                A = 1 / (1 + np.exp(-Z))
            else:
                Z_exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = Z_exp / np.sum(Z_exp, axis=0, keepdims=True)

            self.__cache["A{}".format(layer)] = A

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):

        m = Y.shape[1]

        cost = (-1 / m) * np.sum(Y * np.log(A))

        return cost

    def evaluate(self, X, Y):

        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        predictions = np.zeros_like(A)
        predictions[np.argmax(A, axis=0), np.arange(A.shape[1])] = 1

        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):

        m = Y.shape[1]
        weights_copy = self.__weights.copy()
        dZ = None

        for layer in range(self.__L, 0, -1):
            A = cache["A{}".format(layer)]
            A_prev = cache["A{}".format(layer - 1)]

            if layer == self.__L:
                dZ = A - Y
            else:
                W_next = weights_copy["W{}".format(layer + 1)]
                dZ = np.matmul(W_next.T, dZ) * (A * (1 - A))

            dW = (1/m) * np.matmul(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)

            self.__weights["W{}".format(layer)] = (
                weights_copy["W{}".format(layer)] - alpha * dW
            )
            self.__weights["b{}".format(layer)] = (
                weights_copy["b{}".format(layer)] - alpha * db
            )

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):

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
                    print("Cost after {} iterations: {}".format(i, cost))
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

        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):

        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
