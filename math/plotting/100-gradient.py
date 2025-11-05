#!/usr/bin/env python3
""" A sctipt that creates a scatter plot with gradient """
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    A function that returns a scatter plot of a elevations of a mountain
    """

    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    mountain_elevation = plt.scatter(x, y, c=z)
    plt.colorbar(mountain_elevation, label='elevation (m)')
    plt.title('Mountain Elevation')
    plt.ylabel('y coordinate (m)')
    plt.xlabel('x coordinate (m)')

    plt.show()
