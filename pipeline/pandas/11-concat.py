#!/usr/bin/env python3
""" A script that concatenates a DataFrame"""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    A function that returns the concatenated DataFrame
    """

    df2 = df2[df2["Timestamp"] <= 1417411920]

    df1 = index(df1)

    df2 = index(df2)

    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"], sort=False)
