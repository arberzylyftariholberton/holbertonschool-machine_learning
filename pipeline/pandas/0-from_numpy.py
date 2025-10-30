#!/usr/bin/env python3
"""Script that creates a pandas DataFrame from numpy Array"""

import pandas as pd


def from_numpy(array):
    """
    Function for converting np.ndarray to pd.DataFrame
    """
    columnsNumber = array.shape[1] if len(array.shape) > 1 else 1

    columns = [chr(65 + i) for i in range(columnsNumber)]

    return pd.DataFrame(array, columns=columns)
