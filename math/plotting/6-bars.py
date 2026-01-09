#!/usr/bin/env python3
"""Docstring for 6-bars.py."""
import numpy as np


import matplotlib.pyplot as plt


def bars():
    """Docstring for bars."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    categories = ["Farrah", "Fred", "Felicia"]
    level1 = fruit[0]
    level2 = fruit[1]
    level3 = fruit[2]
    level4 = fruit[3]
    plt.figure(figsize=(6.4, 4.8))
    plt.title("Number of Fruit per Person")
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.bar(categories, level1, color="red", width=0.5)
    plt.bar(categories, level2, bottom=level1,\
            color="yellow", width=0.5)
    plt.bar(categories, level3, bottom=level1+level2,\
            color="#ff8000", width=0.5)
    plt.bar(categories, level4, bottom=level1+level2+level3,\
            color="#ffe5b4", width=0.5)
    plt.legend(["apples", "bananas", "oranges", "peaches"])
