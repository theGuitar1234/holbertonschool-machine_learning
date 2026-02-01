#!/usr/bin/env python3
"""Docstring for 100-slice_like_a_ninja.py."""


def np_slice(matrix, axes={}):
    """Docstring for np_slice."""
    slc = [slice(None)] * matrix.ndim
    for axis, s in axes.items():
        slc[axis] = slice(*s)
    return matrix[tuple(slc)]
