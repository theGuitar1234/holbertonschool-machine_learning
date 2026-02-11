#!/usr/bin/env python3
"""Docstring for binomial.py."""


class Binomial:
    """Docstring for Binomial."""

    def __init__(self, data=None, n=1, p=0.5):
        """Docstring for constructor."""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            var = sum((x - mean) ** 2 for x in data) / len(data)

            p_est = 1 - (var / mean)
            n_est = round(mean / p_est)
            p_final = mean / n_est

            self.n = int(n_est)
            self.p = float(p_final)

    def pmf(self, k):
        """Docstring for pmf."""
        if not isinstance(k, int):
            k = int(k)

        if k < 0 or k > self.n:
            return 0

        comb = 1
        k_eff = min(k, self.n - k)
        for i in range(1, k_eff + 1):
            comb *= (self.n - (k_eff - i))
            comb /= i

        return comb * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Docstring for cdf."""
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0
        if k > self.n:
            return 1

        return sum(self.pmf(i) for i in range(k + 1))
