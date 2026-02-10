#!/usr/bin/env python3
"""Module to calculate sensitivity"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
                   where row indices represent the correct labels and
                   column indices represent the predicted labels
                   classes is the number of classes

    Returns:
        numpy.ndarray of shape (classes,) containing the sensitivity
        of each class
    """

    true_positives = np.diag(confusion)
    actual_positives = np.sum(confusion, axis=1)

    sensitivity = true_positives / actual_positives

    return sensitivity
