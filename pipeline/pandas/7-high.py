#!/usr/bin/env python3
""" A script that sorts a pandas DataFrame in descending order"""


def high(df):
    """
    A function that returns the DataFrame in descending orderd
    """

    sorted = df.sort_values("High", ascending=False)

    return sorted
