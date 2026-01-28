#!/usr/bin/env python3
"""Module to test a neural network"""


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network

    Args:
        network: the network model to test
        data: the input data to test the model with
        labels: the correct one-hot labels of data
        verbose: boolean that determines if
                 output should be printed during testing

    Returns:
        the loss and accuracy of the model
        with the testing data, respectively
    """

    results = network.evaluate(x=data, y=labels, verbose=verbose)
    return results
