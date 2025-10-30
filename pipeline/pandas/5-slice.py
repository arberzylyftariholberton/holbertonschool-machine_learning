#!/usr/bin/env python3
""" A script that slices a pandas DataFrame"""


def slice(df):
    """
    A function that returns a sliced DataFrame
    """

    data = df[["High", "Low", "Close", "Volume_(BTC)"]]

    rows = data.iloc[::60]

    return rows
