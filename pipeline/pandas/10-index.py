#!/usr/bin/env python3
""" A script that modifies Timestamp column as the index of the DataFrame"""


def index(df):
    """
    A function that returns the modified DataFrame index with Timestamp column
    """

    df = df.set_index("Timestamp")

    return df
