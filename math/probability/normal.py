#!/usr/bin/env python3
"""Docstring for normal.py."""


class Normal:
    """Docstring for Nomral."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Docstring for constructor."""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            m = sum(data) / len(data)
            var = sum((x - m) ** 2 for x in data) / len(data)
            self.mean = float(m)
            self.stddev = float(var ** 0.5)

    def z_score(self, x):
        """Docstring for z_score."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Docstring for x_value."""
        return self.mean + (z * self.stddev)
