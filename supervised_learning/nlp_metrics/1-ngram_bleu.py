#!/usr/bin/env python3
"""Module that calculates the n-gram BLEU score."""
import numpy as np


def ngram_bleu(references, sentence, n):
    """Calculate the n-gram BLEU score for a sentence.

    Args:
        references (list): List of reference translations.
        sentence (list): Model proposed sentence.
        n (int): Size of the n-gram.

    Returns:
        float: N-gram BLEU score.
    """
    sentence_length = len(sentence)
    if sentence_length < n or n < 1:
        return 0

    sentence_ngrams = {}
    for i in range(sentence_length - n + 1):
        ngram = tuple(sentence[i:i + n])
        sentence_ngrams[ngram] = sentence_ngrams.get(ngram, 0) + 1

    clipped_count = 0
    for ngram, count in sentence_ngrams.items():
        max_count = 0
        for reference in references:
            reference_ngrams = {}
            for i in range(len(reference) - n + 1):
                ref_ngram = tuple(reference[i:i + n])
                reference_ngrams[ref_ngram] = (
                    reference_ngrams.get(ref_ngram, 0) + 1
                )
            max_count = max(max_count, reference_ngrams.get(ngram, 0))
        clipped_count += min(count, max_count)

    total_ngrams = sentence_length - n + 1
    precision = clipped_count / total_ngrams

    ref_lengths = [len(reference) for reference in references]
    closest_ref = min(ref_lengths,
                      key=lambda ref_len: (abs(ref_len - sentence_length),
                                           ref_len))

    if sentence_length > closest_ref:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - (closest_ref / sentence_length))

    return brevity_penalty * precision
