#!/usr/bin/env python3
"""Docstring for 102-squashed_like_sardines.py."""


def _shape(mat):
    """Docstring for _shape."""
    sh = []
    while isinstance(mat, list):
        sh.append(len(mat))
        mat = mat[0]
    return tuple(sh)


def _copy(mat):
    """Docstring for _copy."""
    if isinstance(mat, list):
        return [_copy(x) for x in mat]
    return mat


def cat_matrices(mat1, mat2, axis=0):
    """Docstring for cat_matrices."""
    if not isinstance(axis, int) or axis < 0:
        return None

    sh1 = _shape(mat1)
    sh2 = _shape(mat2)

    if axis >= len(sh1) or axis >= len(sh2):
        return None

    for d in range(len(sh1)):
        if d == axis:
            continue
        if sh1[d] != sh2[d]:
            return None

    if axis == 0:
        return [_copy(x) for x in mat1] + [_copy(x) for x in mat2]

    result = []
    for a, b in zip(mat1, mat2):
        c = cat_matrices(a, b, axis=axis - 1)
        if c is None:
            return None
        result.append(c)
    return result
