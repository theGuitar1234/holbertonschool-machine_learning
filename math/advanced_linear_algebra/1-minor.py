#!/usr/bin/env python3
"""Docstring for numpy.1-minor."""


def minor(mat):
    """Docstring for minor."""
    if not isinstance(mat, list) or 
            not all(isinstance(row, list) for row in mat):
        print("matrix must be a list of lists")
        return
    if len(mat) == 0 or 
            any(len(row) == 0 for row in mat) or 
            any(len(row) != len(mat) for row in mat):
        print("matrix must be a non-empty square matrix")
        return
    result = []
    for p in range(len(mat)):
        for i in range(len(mat[p])):
            chunk = []
            for j in range(len(mat)):
                if (j != p):
                    temp = []
                    for k in range(len(mat[j])):
                        if (k != i):
                            temp.append(mat[j][k])
                    chunk.append(temp)
            result.append(chunk)
    minor = [[0 for _ in range(len(mat))] for _ in range(len(mat))]
    n = len(mat)
    if n == 1:
        return [[1]]
    for idx, sub in enumerate(result):
        r = idx // n
        c = idx % n
        minor[r][c] = determinant(sub)
    return minor


def determinant(mat):
    """Docstring for determinant."""
    if (not isinstance(mat, list) or len(mat) == 0 or
            not all(isinstance(row, list) for row in mat) or
            any(len(row) == 0 for row in mat)):
        raise TypeError("matrix must be a list of lists")
    n = len(mat)
    if any(len(row) != n for row in mat):
        raise TypeError("matrix must be a square matrix")
    if (len(mat) == 1):
        # print("reached the end : ", mat)
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
