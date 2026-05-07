#!/usr/bin/env python3
"""Module that calculates the cumulative n-gram BLEU score."""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """Calculate the cumulative n-gram BLEU score for a sentence.

    Args:
        references (list): List of reference translations.
        sentence (list): Model proposed sentence.
        n (int): Largest n-gram size to use.

    Returns:
        float: Cumulative n-gram BLEU score.
    """
    sentence_length = len(sentence)
    if sentence_length == 0 or n < 1:
        return 0

    precisions = []
    for gram_size in range(1, n + 1):
        if sentence_length < gram_size:
            return 0

        sentence_ngrams = {}
        for i in range(sentence_length - gram_size + 1):
            ngram = tuple(sentence[i:i + gram_size])
            sentence_ngrams[ngram] = sentence_ngrams.get(ngram, 0) + 1

        clipped_count = 0
        for ngram, count in sentence_ngrams.items():
            max_count = 0
            for reference in references:
                reference_ngrams = {}
                for i in range(len(reference) - gram_size + 1):
                    ref_ngram = tuple(reference[i:i + gram_size])
                    reference_ngrams[ref_ngram] = (
                        reference_ngrams.get(ref_ngram, 0) + 1
                    )
                max_count = max(max_count, reference_ngrams.get(ngram, 0))
            clipped_count += min(count, max_count)

        total_ngrams = sentence_length - gram_size + 1
        precisions.append(clipped_count / total_ngrams)

    if min(precisions) == 0:
        return 0

    weight = 1 / n
    geometric_mean = np.exp(
        np.sum([weight * np.log(precision) for precision in precisions])
    )

    ref_lengths = [len(reference) for reference in references]
    closest_ref = min(ref_lengths,
                      key=lambda ref_len: (abs(ref_len - sentence_length),
                                           ref_len))

    if sentence_length > closest_ref:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - (closest_ref / sentence_length))

    return brevity_penalty * geometric_mean
