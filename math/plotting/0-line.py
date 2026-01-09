#!/usr/bin/env python3
"""Docstring for 0-line.py."""

import numpy as np


import matplotlib.pyplot as plt


def line():
    """Docstring for line."""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y, color='red')
    plt.xlim(0, 10)
    plt.show()


line()
