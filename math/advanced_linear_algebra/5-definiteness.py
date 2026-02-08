#!/usr/bin/env python3
"""5-definiteness.py."""


import numpy as np


def definiteness(matrix):
    """Docstring for definiteness."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if (
        matrix.ndim != 2 
        or matrix.shape[0] == 0 
        or matrix.shape[0] != matrix.shape[1]
    ):
        return None
    tol = 1e-8
    if not np.allclose(matrix, matrix.T, atol=tol, rtol=0):
        return None
    eigvals = np.linalg.eigvalsh(matrix)
    if np.all(eigvals > tol):
        return "Positive definite"
    if np.all(eigvals >= -tol) and np.any(np.abs(eigvals) <= tol):
        return "Positive semi-definite"
    if np.all(eigvals < -tol):
        return "Negative definite"
    if np.all(eigvals <= tol) and np.any(np.abs(eigvals) <= tol):
        return "Negative semi-definite"
    if np.any(eigvals > tol) and np.any(eigvals < -tol):
        return "Indefinite"
    return None
