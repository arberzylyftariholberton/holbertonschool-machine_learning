#!/usr/bin/env python3
""" A script that transposes a pandas DataFrame"""


def flip_switch(df):
    """
    A function that returns the transposed DataFrame
    """

    transposed = df.sort_values("Timestamp", ascending = False)

    return transposed.T
