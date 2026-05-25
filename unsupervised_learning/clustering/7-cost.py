#!/usr/bin/env python3
"""A module that does the trick"""
import numpy as np


def cost(P, Q):
    """A function that does the trick"""
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    C = np.sum(P * np.log(P / Q))
    return C
