#!/usr/bin/env python3
"""Docstring for numpy.5-across_the_planes."""


def add_matrices2D(mat1, mat2):
    """Docstring for add_matrices2D."""
    if (len(mat1)*len(mat1[0]) != len(mat2)*len(mat2[0])):
        return None
    result = []
    temp = []
    for i in range(len(mat1[0])):
        for j in range(len(mat1)):
            temp.append(mat1[i][j] + mat2[i][j])
        result.append(temp)
        temp = []
    return result
