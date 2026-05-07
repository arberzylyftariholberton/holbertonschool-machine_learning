#!/usr/bin/env python3
"""Module that creates bag of words embeddings."""
import re
import numpy as np


def tokenize(sentence):
    """Convert a sentence into normalized tokens.

    Args:
        sentence (str): Sentence to tokenize.

    Returns:
        list: List of lowercase tokens.
    """
    tokens = re.findall(r"[a-z0-9]+", sentence.lower())
    return [token for token in tokens if len(token) > 1]


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix.

    Args:
        sentences (list): List of sentences to analyze.
        vocab (list, optional): Vocabulary words to use.

    Returns:
        tuple: embeddings, features
            embeddings is a numpy.ndarray of shape (s, f)
            features is a numpy.ndarray of the features used
    """
    tokenized = [tokenize(sentence) for sentence in sentences]

    if vocab is None:
        features = sorted(set(token
                              for sentence in tokenized
                              for token in sentence))
    else:
        features = vocab

    features = np.array(features)
    embeddings = np.zeros((len(sentences), len(features)), dtype=int)

    for i, sentence in enumerate(tokenized):
        for j, word in enumerate(features):
            embeddings[i, j] = sentence.count(word)

    return embeddings, features
