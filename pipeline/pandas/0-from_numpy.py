#!/usr/bin/env python3
"""Docstring for 0-from_numpy.py."""

import pandas as pd


def from_numpy(seed):
    """Docstring for from_numpy."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    return pd.DataFrame(seed, columns=list(alphabet[:seed.shape[1]]))
