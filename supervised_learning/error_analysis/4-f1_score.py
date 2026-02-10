#!/usr/bin/env python3
"""Module to calculate F1 score"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
                   where row indices represent the correct labels and
                   column indices represent the predicted labels
                   classes is the number of classes

    Returns:
        numpy.ndarray of shape (classes,) containing the F1 score
        of each class
    """

    recall = sensitivity(confusion)
    prec = precision(confusion)

    f1 = 2 * (prec * recall) / (prec + recall)

    return f1
