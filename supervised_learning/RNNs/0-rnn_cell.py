#!/usr/bin/env python3
"""Module that defines a simple RNN cell."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """Initialize the RNN cell.

        Args:
            i (int): Dimensionality of the data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the output.
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Perform forward propagation for one time step.

        Args:
            h_prev (numpy.ndarray): Previous hidden state of shape (m, h).
            x_t (numpy.ndarray): Input data of shape (m, i).

        Returns:
            tuple: h_next, y
                h_next is the next hidden state
                y is the output of the cell
        """
        hx = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(hx, self.Wh) + self.bh)

        y_linear = np.matmul(h_next, self.Wy) + self.by
        y_shifted = y_linear - np.max(y_linear, axis=1, keepdims=True)
        y_exp = np.exp(y_shifted)
        y = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return h_next, y
