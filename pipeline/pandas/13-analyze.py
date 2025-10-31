#!/usr/bin/env python3
""" A script that computes descriptive statistics
    for all columns except the Timestamp column"""


def analyze(df):
    """
    Function that returns the contained DataFrame
    With descriptive statistics except Timestamp
    """

    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    statistics = df.describe()

    return statistics
