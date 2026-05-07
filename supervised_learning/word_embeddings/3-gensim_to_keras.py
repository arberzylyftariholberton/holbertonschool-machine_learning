#!/usr/bin/env python3
"""Module that converts a gensim model to a keras embedding layer."""
import tensorflow as tf


def gensim_to_keras(model):
    """Convert a gensim Word2Vec model to a keras Embedding layer.

    Args:
        model: Trained gensim Word2Vec model.

    Returns:
        tf.keras.layers.Embedding: Trainable keras Embedding layer.
    """
    weights = model.wv.vectors
    return tf.keras.layers.Embedding(input_dim=weights.shape[0],
                                     output_dim=weights.shape[1],
                                     weights=[weights],
                                     trainable=True)
