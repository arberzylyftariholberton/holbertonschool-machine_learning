#!/usr/bin/env python3
""" Creating a script with a bar chart"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    A function that returns a bar chart with 3 people and 4 fruits
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    x = ['Farrah', 'Fred', 'Felicia']
    plt.bar(x, fruit[0], color='red', width=0.5, label='apples')
    plt.bar(
        x,
        fruit[1],
        bottom=fruit[0],
        color='yellow',
        width=0.5,
        label='bananas')
    plt.bar(
        x,
        fruit[2],
        bottom=fruit[0] + fruit[1],
        color='#ff8000',
        width=0.5,
        label='oranges')
    plt.bar(
        x,
        fruit[3],
        bottom=fruit[0] + fruit[1] + fruit[2],
        color='#ffe5b4',
        width=0.5,
        label='peaches')

    plt.ylim(0, 80)
    plt.title('Number of Fruit per Person')
    plt.ylabel('Quantity of Fruit')

    plt.legend()
    plt.show()
