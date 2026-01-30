#!/usr/bin/env python3
"""Docstring for 2-size_me_please.py."""


def matrix_shape(matrix):
    """Docstring for matrix_shape."""
    result = [len(matrix)]
    matrix = matrix[0]
    while type(matrix) is not int:
        result.append(len(matrix))
        matrix = matrix[0]
    return result
