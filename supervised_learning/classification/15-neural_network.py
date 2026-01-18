#!/usr/bin/env python3
"""
   A Script That Upgrades The Training Of The Neural Network
"""

import numpy as np


class NeuralNetwork:
    """
    A class that defines a Neural Network performing binary classification,
    adding private instance attributes
    """

    def __init__(self, nx, nodes):
        """
        A constructor that takes number of input as nx and
        nodes is the number of nodes found in the hidden layer
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
        """
        Getter function of the Weight vector for the hidden layer
        """

        return self.__W1

    @property
    def b1(self):
        """
        Getter function of The bias for the hidden layer
        """

        return self.__b1

    @property
    def A1(self):
        """
        Getter function of The activated output for the hidden layer
        """

        return self.__A1

    @property
    def W2(self):
        """
        Getter function of the Weight vector for the output neuron
        """

        return self.__W2

    @property
    def b2(self):
        """
        Getter function of The bias for the output neuron
        """

        return self.__b2

    @property
    def A2(self):
        """
        Getter function of The activated output for the output neuron
        """

        return self.__A2

    def forward_prop(self, X):
        """
        A function that Calculates the forward propagation
        of the neural network
        """

        Z1 = np.matmul(self.__W1, X) + self.__b1

        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2

        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        A Function that Calculates the cost of the model
        using logistic regression
        """

        m = Y.shape[1]

        log_loss = -1/m*np.sum(Y * np.log(A) + (1-Y)*(np.log(1.0000001-A)))

        return log_loss

    def evaluate(self, X, Y):
        """
        A Function that Evaluates the neural network's predictions
        """

        A1, A2 = self.forward_prop(X)

        prediction = (A2 >= 0.5).astype(int)

        cost_value = self.cost(Y, A2)

        return prediction, cost_value

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        A Function that Calculates one pass of
        Gradient Descent on the neural network
        """

        m = Y.shape[1]

        grad_w2 = 1/m * np.matmul((A2-Y), A1.T)
        grad_b2 = 1/m * np.sum((A2-Y), axis=1, keepdims=True)

        grad_w1 = 1/m * np.matmul(
            (np.matmul(self.__W2.T, (A2-Y)) * (A1 * (1-A1))),
            X.T
        )
        grad_b1 = 1/m * np.sum(
            (np.matmul(self.__W2.T, (A2-Y)) * (A1 * (1-A1))),
            axis=1,
            keepdims=True
        )

        self.__W1 = self.__W1 - alpha * grad_w1
        self.__W2 = self.__W2 - alpha * grad_w2
        self.__b1 = self.__b1 - alpha * grad_b1
        self.__b2 = self.__b2 - alpha * grad_b2

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        A Function That Returns The Upgraded Train Of The Neural Network
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
