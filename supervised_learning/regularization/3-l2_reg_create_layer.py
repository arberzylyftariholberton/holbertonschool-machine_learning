#!/usr/bin/env python3
"""Module to create a neural network layer
   with L2 regularization in TensorFlow"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer in TensorFlow
    that includes L2 regularization

    Args:
        prev: tensor containing the output of the previous layer
        n: number of nodes the new layer should contain
        activation: activation function that should be used on the layer
        lambtha: L2 regularization parameter

    Returns:
        output of the new layer
    """

    regularizer = tf.keras.regularizers.L2(lambtha)
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )
    output = layer(prev)

    return output
