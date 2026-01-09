#!/usr/bin/env python3
"""Docstring for 5-slice.py."""


def slice(df):
    """Docstring for slice."""
    df = df[["High", "Low", "Close", "Volume_(BTC)"]]
    return df.iloc[::60]
