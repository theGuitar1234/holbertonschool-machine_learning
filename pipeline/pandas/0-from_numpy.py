#!/usr/bin/env python3

import pandas as pd

"""
    DocString for 0-from_numpy.py
"""


def from_numpy(seed):

    """
        DocString for from_numpy()
    """

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    return pd.DataFrame(seed, columns=list(alphabet[:seed.shape[1]]))
