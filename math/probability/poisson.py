#!/usr/bin/env python3
"""Docstring for poisson.py."""


class Poisson:
    """Docstring for Poisson."""

    def __init__(self, data=None, lambtha=1.):
        """Docstring for the constructor."""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """Docstring for pmf."""
        k = int(k)
        if k < 0:
            return 0
        return (2.7182818285 ** ((-self.lambtha) * (self.lambtha ** k))) / factorial(k)

    def factorial(k):
        if (k == 0):
            return 1
        if (k == 1):
            return 1
        return k*factorial(k-1)
        
    def cdf(self, k):
        """Docstring for cdf."""
        k = int(k)
        if k < 0:
            return 0
        return sum(self.pmf(i) for i in range(k + 1))
