#!/usr/bin/env python3
"""Module for creating batch normalization layer in TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function that
                    should be used on the output of the layer

    Returns:
        tensor of the activated output for the layer
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer
    )

    Z = layer(prev)

    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    Z_normalized = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, 1e-7
    )

    return activation(Z_normalized)
