#!/usr/bin/env python3
"""Docstring for 101-the_whole_barn.py."""


def add_matrices(mat1, mat2):
    """Docstring for add_matrices."""
    if isinstance(mat1, (int, float)) and isinstance(mat2, (int, float)):
        return mat1 + mat2

    if not (isinstance(mat1, list) and isinstance(mat2, list)):
        return None

    if len(mat1) != len(mat2):
        return None

    result = []
    for a, b in zip(mat1, mat2):
        added = add_matrices(a, b)
        if added is None:
            return None
        result.append(added)

    return result
