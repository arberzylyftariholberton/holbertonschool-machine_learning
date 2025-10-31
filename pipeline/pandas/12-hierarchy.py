#!/usr/bin/env python3
""" A script that concatenates the bitstamp coinbase in chronological order"""

import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """
    Function that returns the concatenated DataFrame of bitstamp and coinbase
    Timestamp should be the first level of the MultiIndex
    The data is displayed in chronological order by Timestamp
    """

    timestamp1 = 1417411980
    timestamp2 = 1417417980

    coinbaseIndex = index(df1)
    bitstampIndex = index(df2)

    coinbaseIndex = coinbaseIndex.loc[timestamp1:timestamp2]
    bitstampIndex = bitstampIndex.loc[timestamp1:timestamp2]

    concatenated = pd.concat(
        [bitstampIndex, coinbaseIndex],
        keys=["bitstamp", "coinbase"],
        sort=False
    )

    concatenated = concatenated.swaplevel(0, 1)

    concatenated = concatenated.sort_index(level=0)

    return concatenated
