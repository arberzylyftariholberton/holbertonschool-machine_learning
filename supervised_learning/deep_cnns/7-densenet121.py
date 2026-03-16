#!/usr/bin/env python3
"""Module that defines the densenet121 function."""

from tensorflow import keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """Builds the DenseNet-121 architecture.

    The network follows the architecture described in
    "Densely Connected Convolutional Networks". It uses
    dense blocks separated by transition layers, ending
    with global average pooling and a softmax classifier.

    Returns:
        The Keras model for the DenseNet-121 network.
    """

    initializer = K.initializers.he_normal(seed=0)
    inputs = K.Input(shape=(224, 224, 3))

    X = K.layers.BatchNormalization(axis=3)(inputs)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(2 * growth_rate, (7, 7), strides=(2, 2),
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                              padding='same')(X)

    X, nb_filters = dense_block(X, 2 * growth_rate, growth_rate, 6)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 12)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 24)
    X, nb_filters = transition_layer(X, nb_filters, compression)

    X, nb_filters = dense_block(X, nb_filters, growth_rate, 16)

    X = K.layers.AveragePooling2D((7, 7), strides=(7, 7))(X)
    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=initializer)(X)

    return K.models.Model(inputs=inputs, outputs=X)
