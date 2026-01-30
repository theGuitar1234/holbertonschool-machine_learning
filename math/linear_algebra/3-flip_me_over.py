#!/usr/bin/env python3
"""Docstring for numpy.3-flip_me_over."""


def matrix_transpose(matrix):
    """Docstring for matrix_transpose: param matrix: Description."""
    result = []
    temp = []
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            temp.append(matrix[j][i])
        result.append(temp)
        temp = []
    return result
