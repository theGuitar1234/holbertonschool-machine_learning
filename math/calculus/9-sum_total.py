#!/usr/bin/env python3
"""Docstring."""


def summation_i_squared(n):
    """
    this is my function
    """
    if isinstance(n, int) and n > 0:
        return n*(n + 1)*(2*n + 1) / 6
    else:
        return None
