#!/usr/bin/env python3
'''
normalizing matrixes
'''

import numpy as np


def normalize(X, m, s):
    """
    normalizing matrix
    """
    normalized = (X - m) / s
    return normalized
