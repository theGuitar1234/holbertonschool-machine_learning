#!/usr/bin/env python3
"""Docstring for numpy.7-gettin_cozy."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Docstring for cat_matrices2D."""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [i[:] for i in mat1] + [i[:] for i in mat2]

    else:
        if len(mat1) != len(mat2):
            return None
        return [r1[:] + r2[:] for r1, r2 in zip(mat1, mat2)]
