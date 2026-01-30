#!/usr/bin/env python3
"""Docstring for numpy.4-line_up."""


def add_arrays(arr1, arr2):
    """Docstring for add_arrays."""
    if (len(arr1) != len(arr2)):
        return None
    return [arr1[i] + arr2[i]
            for i in range(len(arr1))]
