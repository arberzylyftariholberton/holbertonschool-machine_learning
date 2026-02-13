#!/usr/bin/env python3
"""Module to calculate L2 regularization cost in Keras"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization

    Args:
        cost: tensor containing the cost of the network
              without L2 regularization
        model: Keras model that includes layers with L2 regularization

    Returns:
        tensor containing the total cost for each layer of the network,
        accounting for L2 regularization
    """

    l2_losses = model.losses

    return tf.convert_to_tensor([cost + tf.reduce_sum(l2_losses)] + l2_losses)
