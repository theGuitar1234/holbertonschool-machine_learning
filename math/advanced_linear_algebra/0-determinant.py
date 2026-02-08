#!/usr/bin/env python3
"""Docstring for numpy.0-determinant."""


def determinant(mat):
    """Docstring for determinant."""
    if (not isinstance(mat, list)
            or len(mat) == 0
            or not all(isinstance(row, list) for row in mat)):
        raise TypeError("matrix must be a list of lists")
    if len(mat) == 1 and len(mat[0]) == 0:
        return 1
    if any(len(row) != len(mat) for row in mat):
        raise TypeError("matrix must be a square matrix")

    if (len(mat) == 1):
        #print("reached the end : ", mat)
        return mat[0][0]

    result = []

    res = 0
    
    for i in range(len(mat[0])):
        chunk = []
        for j in range(1, len(mat)):
            temp = []
            for k in range(len(mat[j])):
                if (k != i):
                    temp.append(mat[j][k])
            chunk.append(temp)
        result.append(chunk)
        res += mat[0][i] * determinant(chunk) * (-1)**i
    return res
