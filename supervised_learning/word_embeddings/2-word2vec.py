#!/usr/bin/env python3
"""Module that builds and trains a Word2Vec model."""
from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a Word2Vec model.

    Args:
        sentences (list): List of tokenized sentences.
        vector_size (int): Dimensionality of embeddings.
        min_count (int): Minimum word frequency.
        window (int): Maximum distance between current and predicted word.
        negative (int): Size of negative sampling.
        cbow (bool): True for CBOW, False for Skip-gram.
        epochs (int): Number of training epochs.
        seed (int): Random seed.
        workers (int): Number of worker threads.

    Returns:
        Word2Vec: Trained gensim Word2Vec model.
    """
    model = Word2Vec(sentences=sentences,
                     vector_size=vector_size,
                     min_count=min_count,
                     window=window,
                     negative=negative,
                     sg=0 if cbow else 1,
                     epochs=epochs,
                     seed=seed,
                     workers=workers)
    return model
