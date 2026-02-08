#!/usr/bin/env python3
"""Docstring for numpy.4-inverse.py."""


def inverse(mat):
    """Docstring for inverse."""
    if (
        not isinstance(mat, list)
        or not all(isinstance(row, list) for row in mat)
    ):
        print("matrix must be a list of lists")
        return
    if (
        len(mat) == 0
        or any(len(row) == 0 for row in mat)
        or any(len(row) != len(mat) for row in mat)
    ):
        print("matrix must be a non-empty square matrix")
        return
    return 1/determinant(mat) * adjugate(mat)


def adjugate(mat):
    """Docstring for adjugate."""
    if (
        not isinstance(mat, list)
        or not all(isinstance(row, list) for row in mat)
    ):
        print("matrix must be a list of lists")
        return
    if (
        len(mat) == 0
        or any(len(row) == 0 for row in mat)
        or any(len(row) != len(mat) for row in mat)
    ):
        print("matrix must be a non-empty square matrix")
        return
    if len(mat) == 1:
        return [[1]]

    result = []
    res = [[0 for _ in range(len(mat[0]))] for _ in range(len(mat))]
    for r in range(len(mat)):
        for i in range(len(mat)):
            chunk = []
            for j in range(len(mat)):
                if (j != r):
                    temp = []
                    for k in range(len(mat)):
                        if (k != i):
                            temp.append(mat[j][k])
                    chunk.append(temp)
            result.append(chunk)
            res[i][r] = determinant(chunk) * (-1)**(i+r)
    return res


def determinant(mat):
    """Docstring for determinant."""
    if (not isinstance(mat, list) or len(mat) == 0 or
            not all(isinstance(row, list) for row in mat)):
        print("matrix must be a list of lists")
        return
    if len(mat) == 1 and len(mat[0]) == 0:
        return 1
    n = len(mat)
    if any(len(row) != n for row in mat):
        print("matrix must be a square matrix")
        return
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
