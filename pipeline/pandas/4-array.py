#!/usr/bin/env python3
"""Docstring for 4-array.py."""


def array(df):
    """Docstring for array."""
    return df[["High", "Close"]].tail(10).values
