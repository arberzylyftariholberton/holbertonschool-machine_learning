#!/usr/bin/env python3
""" A script that fill missing values"""


def fill(df):
    """
    A function that returns the DataFrame with filled missing Values
    """

    df = df.drop(columns=["Weighted_Price"])

    df["Close"] = df["Close"].ffill()

    for cols in ["High", "Low", "Open"]:
        df[cols] = df[cols].fillna(df["Close"])

    for cols in ["Volume_(BTC)", "Volume_(Currency)"]:
        df[cols] = df[cols].fillna(0)

    return df
