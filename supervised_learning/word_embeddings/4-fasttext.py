#!/usr/bin/env python3
"""Module that builds and trains a FastText model."""
from gensim.models import FastText


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a FastText model.

    Args:
        sentences (list): List of tokenized sentences.
        vector_size (int): Dimensionality of embeddings.
        min_count (int): Minimum word frequency.
        negative (int): Size of negative sampling.
        window (int): Maximum distance between current and predicted word.
        cbow (bool): True for CBOW, False for Skip-gram.
        epochs (int): Number of training epochs.
        seed (int): Random seed.
        workers (int): Number of worker threads.

    Returns:
        FastText: Trained gensim FastText model.
    """
    model = FastText(sentences=sentences,
                     vector_size=vector_size,
                     min_count=min_count,
                     window=window,
                     negative=negative,
                     sg=0 if cbow else 1,
                     epochs=epochs,
                     seed=seed,
                     workers=workers)
    return model
