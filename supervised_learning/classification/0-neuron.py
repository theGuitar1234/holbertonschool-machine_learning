#!/usr/bin/env python3
"""Docstring for 0-neuron.py."""

import numpy as np


class Neuron:
    """Docstring for Neuron."""
    def __init__(self, nx):
        """Docstring for constructor."""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(nx).reshape(1, nx)
        self.b = 0
        self.A = 0
