#!/usr/bin/env python3
""" A script that creates a line graph """
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    A function that returns a line graph
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, color='red')
    plt.xlim(0, 10)
    plt.show()
