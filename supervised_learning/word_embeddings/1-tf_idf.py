#!/usr/bin/env python3
"""Module that creates TF-IDF embeddings."""
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


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding matrix.

    Args:
        sentences (list): List of sentences to analyze.
        vocab (list, optional): Vocabulary words to use.

    Returns:
        tuple: embeddings, features
            embeddings is a numpy.ndarray of shape (s, f)
            features is a numpy.ndarray of the features used
    """
    tokenized = [tokenize(sentence) for sentence in sentences]
    s = len(sentences)

    if vocab is None:
        features = sorted(set(token
                              for sentence in tokenized
                              for token in sentence))
    else:
        features = vocab

    features = np.array(features)
    embeddings = np.zeros((s, len(features)))

    for j, word in enumerate(features):
        doc_freq = sum(word in sentence for sentence in tokenized)
        idf = np.log((1 + s) / (1 + doc_freq)) + 1

        for i, sentence in enumerate(tokenized):
            term_count = sentence.count(word)
            if term_count == 0:
                continue
            embeddings[i, j] = term_count * idf

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    return embeddings, features
