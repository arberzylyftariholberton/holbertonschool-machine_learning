#!/usr/bin/env python3
"""
A script that visualize a DataFrame
"""

import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file


def visualize():
    """
    A function that returns a transformed DataFrame
    """

    df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

    df = df.drop(columns=['Weighted_Price'])

    df = df.rename(columns={'Timestamp': 'Date'})

    df['Date'] = pd.to_datetime(df['Date'], unit='s')

    df = df.set_index('Date')

    df['Close'] = df['Close'].ffill()

    for col in ['High', 'Low', 'Open']:
        df[col] = df[col].fillna(df['Close'])

    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        df[col] = df[col].fillna(0)

    df = df[df.index.year >= 2017]

    daily_Intervals = df.resample('D').agg({
        'High': 'max',
        'Low': 'min',
        'Open': 'mean',
        'Close': 'mean',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum'
    })

    daily_Intervals.plot(title='Virtual coins daily_Intervals ')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.tight_layout()
    plt.show()

    return daily_Intervals
