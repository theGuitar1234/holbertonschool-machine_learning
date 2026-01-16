#!/usr/bin/env python3
"""Docstring."""


sum = 0


def summation_i_squared(n):
    """Docstring."""
    if n.__class__ is not int:
        return None
    global sum
    if (n == 0):
        return sum
    sum += n**2
    return summation_i_squared(n-1)
