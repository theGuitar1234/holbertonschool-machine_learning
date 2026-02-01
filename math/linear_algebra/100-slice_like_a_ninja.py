#!/usr/bin/env python3
"""Docstring for 100-slice_like_a_ninja.py."""


def np_slice(matrix, axes={}):
    """Docstring for np_slice."""
    result = []
    for key in axes.keys():
        for i in range(len(matrix)):
            s = axes[key]
            result.append(matrix[i][slice(*s)].tolist())
    return result
