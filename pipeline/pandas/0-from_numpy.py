#!/usr/bin/env python3

"""
    DocString for 0-from_numpy.py
"""


def from_numpy(seed):

    """
        DocString for from_numpy()
    """

    alphabet = __import__("string").ascii_uppercase

    return pd.DataFrame(seed, columns=list(alphabet[:seed.shape[1]]))
