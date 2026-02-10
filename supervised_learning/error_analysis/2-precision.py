#!/usr/bin/env python3
"""Module to calculate precision"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
                   where row indices represent the correct labels and
                   column indices represent the predicted labels
                   classes is the number of classes

    Returns:
        numpy.ndarray of shape (classes,) containing the precision
        of each class
    """

    true_positives = np.diag(confusion)
    predicted_positives = np.sum(confusion, axis=0)

    precision = true_positives / predicted_positives

    return precision
