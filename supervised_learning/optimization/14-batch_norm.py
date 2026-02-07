#!/usr/bin/env python3
"""Module for creating batch normalization layer in TensorFlow"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow

    Args:
        prev: activated output of the previous layer
        n: number of nodes in the layer to be created
        activation: activation function that should be
                    used on the output of the layer

    Returns:
        tensor of the activated output for the layer
    """

    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    dense_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=initializer
    )

    Z = dense_layer(prev)

    batch_norm_layer = tf.keras.layers.BatchNormalization(
        epsilon=1e-7,
        gamma_initializer='ones',
        beta_initializer='zeros'
    )

    Z_normalized = batch_norm_layer(Z)

    output = activation(Z_normalized)

    return output
