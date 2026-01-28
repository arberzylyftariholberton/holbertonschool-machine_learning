#!/usr/bin/env python3
"""Module to make predictions using a neural network"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network

    Args:
        network: the network model to make the prediction with
        data: the input data to make the prediction with
        verbose: boolean that determines if output
        should be printed during prediction

    Returns:
        the prediction for the data
    """

    predictions = network.predict(x=data, verbose=verbose)
    return predictions
