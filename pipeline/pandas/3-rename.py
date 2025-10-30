#!/usr/bin/env python3

""" A script that takes a pandas pd.DataFrame as an input
    and performs a modification in this input. """

import pandas as pd


def rename(df):
    """
    A Function returns the pandas DataFrame and modifies it
    """

    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="s")

    df = df.drop(columns=["Timestamp"])

    df = df[["Datetime", "Close"]]

    return df
