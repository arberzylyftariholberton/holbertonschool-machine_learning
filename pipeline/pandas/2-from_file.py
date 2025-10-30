#!/usr/bin/env python3
"""Script that loads data from a file as a pd.DataFrame"""

import pandas as pd


def from_file(x, delimiter):
    """
    Function that return loaded data from a file as a pd.DataFrame
    """
    return pd.read_csv(x, delimiter=delimiter)