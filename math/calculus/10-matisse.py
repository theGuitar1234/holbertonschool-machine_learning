#!/usr/bin/env python3
"""
this is my function doc for entry
"""


def poly_derivative(poly):
    """Return the derivative of a polynomial"""

    # Validate input
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    # Compute derivative
    result = []
    for i in range(1, len(poly)):
        result.append(poly[i] * i)

    # If derivative is zero
    if len(result) == 0 or all(coef == 0 for coef in result):
        return [0]

    return result
