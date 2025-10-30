#!/usr/bin/env python3
""" A script that converts pandas DataFrame input into numpy array."""

import pandas as pd


def array(df):
    """
    A function that returns the converted numpy array from pandas DataFrame
    """

    data = df[["High", "Close"]]

    rows = data.tail(10)

    arr = rows.to_numpy()

    return arr
