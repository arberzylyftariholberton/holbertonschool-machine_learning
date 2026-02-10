#!/usr/bin/env python3
"""Module to calculate specificity"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    Args:
        confusion: confusion numpy.ndarray of shape (classes, classes)
                   where row indices represent the correct labels and
                   column indices represent the predicted labels
                   classes is the number of classes

    Returns:
        numpy.ndarray of shape (classes,) containing the specificity
        of each class
    """

    classes = confusion.shape[0]
    specificity_scores = np.zeros(classes)

    for i in range(classes):
        TP = confusion[i, i]
        FP = np.sum(confusion[:, i]) - TP
        FN = np.sum(confusion[i, :]) - TP
        TN = np.sum(confusion) - TP - FP - FN

        specificity_scores[i] = TN / (TN + FP)

    return specificity_scores
