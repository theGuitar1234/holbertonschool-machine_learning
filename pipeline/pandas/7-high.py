#!/usr/bin/env python3
"""Docstring for 7-high.py."""


def high(df):
    """Docstring for high."""
    df.sort_values("High", ascending=False, inplace=True)
    return df
