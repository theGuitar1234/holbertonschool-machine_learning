#!/usr/bin/env python3
"""Docstring for 2-from_file.py."""

import pandas as pd


def from_file(filename, delimiter):
    """Docstring for from_file."""
    return pd.read_csv(filename, delimiter=delimiter)
