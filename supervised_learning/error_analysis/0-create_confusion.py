#!/usr/bin/env python3
"""Module to create confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    Args:
        labels: one-hot numpy.ndarray of shape (m, classes) containing
                the correct labels for each data point
                m is the number of data points
                classes is the number of classes
        logits: one-hot numpy.ndarray of shape (m, classes) containing
                the predicted labels

    Returns:
        confusion numpy.ndarray of shape (classes, classes) with row indices
        representing the correct labels and column indices representing
        the predicted labels
    """
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))

    for true, pred in zip(true_labels, predicted_labels):
        confusion[true][pred] += 1

    return confusion
