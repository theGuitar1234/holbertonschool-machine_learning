#!/usr/bin/env python3
"""Docstring exponential.py."""


class Exponential:
    """Docstring for Exponential."""

    def __init__(self, data=None, lambtha=1.):
        """Docstring for constructor."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            self.lambtha = float(1 / mean)

    def pdf(self, x):
        """Docstring for pdf."""
        if x < 0:
            return 0
        return self.lambtha * (2.7182818285 ** (-self.lambtha * x))
