#!/usr/bin/env python3
"""Docstring for 1-scatter.py."""
import numpy as np


import matplotlib.pyplot as plt


def scatter():
    """Docstring for scatter."""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))
    plt.title("Men's Height vs Weight")
    plt.xlabel("Weight (lbs)")
    plt.ylabel("Height (in)")
    plt.scatter(x, y)
