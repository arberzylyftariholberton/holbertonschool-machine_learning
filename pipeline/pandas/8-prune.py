#!/usr/bin/env python3
""" A script that removes NaN values"""


def prune(df):
    """
    A function that returns the DataFrame without NaN values in Close
    """

    prune_NaN = df.dropna(subset=["Close"])

    return prune_NaN
