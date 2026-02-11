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

    def pdf(self, x):
        """Docstring for pdf."""
        pi = 3.1415926536
        e = 2.7182818285

        z = (x - self.mean) / self.stddev
        coef = 1 / (self.stddev * ((2 * pi) ** 0.5))
        expo = e ** (-0.5 * (z ** 2))
        return coef * expo

    def cdf(self, x):
        """Docstring for cdf."""
        pi = 3.1415926536

        t = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = (2 / (pi ** 0.5)) * (
            t
            - (t ** 3) / 3
            + (t ** 5) / 10
            - (t ** 7) / 42
            + (t ** 9) / 216
        )

        return 0.5 * (1 + erf)
