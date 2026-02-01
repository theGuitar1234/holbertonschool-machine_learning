#!/usr/bin/env python3
"""Docstring for 100-slice_like_a_ninja.py."""


import numpy as np


def np_slice(matrix, axes={}):
    """Docstring for np_slice."""
    for key in axes.keys():
        for i in matrix:
            s = axes[key]
            print(i[slice(*s)])
