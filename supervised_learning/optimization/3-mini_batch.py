#!/usr/bin/env python3
"""Module to create mini-batches for training"""
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches to be used for training a neural network
    using mini-batch gradient descent

    Args:
        X: numpy.ndarray of shape (m, nx) representing input data
           m is the number of data points
           nx is the number of features in X
        Y: numpy.ndarray of shape (m, ny) representing the labels
           m is the same number of data points as in X
           ny is the number of classes for classification tasks
        batch_size: number of data points in a batch

    Returns:
        list of mini-batches containing tuples (X_batch, Y_batch)
    """

    X_shuffled, Y_shuffled = shuffle_data(X, Y)

    m = X.shape[0]
    mini_batches = []

    num_complete_batches = m // batch_size

    for i in range(num_complete_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        X_batch = X_shuffled[start_idx:end_idx]
        Y_batch = Y_shuffled[start_idx:end_idx]

        mini_batches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        start_idx = num_complete_batches * batch_size

        X_batch = X_shuffled[start_idx:]
        Y_batch = Y_shuffled[start_idx:]

        mini_batches.append((X_batch, Y_batch))

    return mini_batches
