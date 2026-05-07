#!/usr/bin/env python3
"""Module that calculates the unigram BLEU score."""
import numpy as np


def uni_bleu(references, sentence):
    """Calculate the unigram BLEU score for a sentence.

    Args:
        references (list): List of reference translations.
        sentence (list): Model proposed sentence.

    Returns:
        float: Unigram BLEU score.
    """
    sentence_length = len(sentence)
    if sentence_length == 0:
        return 0

    counts = {}
    for word in sentence:
        counts[word] = counts.get(word, 0) + 1

    clipped_count = 0
    for word, count in counts.items():
        max_count = 0
        for reference in references:
            reference_count = reference.count(word)
            max_count = max(max_count, reference_count)
        clipped_count += min(count, max_count)

    precision = clipped_count / sentence_length

    ref_lengths = [len(reference) for reference in references]
    closest_ref = min(ref_lengths,
                      key=lambda ref_len: (abs(ref_len - sentence_length),
                                           ref_len))

    if sentence_length > closest_ref:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - (closest_ref / sentence_length))

    return brevity_penalty * precision
