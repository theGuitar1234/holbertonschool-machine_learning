#!/usr/bin/env python3
"""
This is my doc
"""


def poly_integral(poly, C=0):
    """This is function doc"""
    try:
        if not all(isinstance(coef, (int, float)) for coef in poly):
            return None
    except TypeError:
        return None

        return None
    if not isinstance(C, int):
        return None
    if len(poly) == 0:
        return [C] if C != 0 else None
    integral = [C]  # Start with the integration constant

    # Integrate each coefficient
    for power, coef in enumerate(poly):
        new_coef = coef / (power + 1)
        # Keep integer if result is whole number
        if new_coef.is_integer():
            new_coef = int(new_coef)
        integral.append(new_coef)

    # Remove trailing zeros to make the list as small as possible
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
