#!/usr/bin/env python3
"""Docstring for 8-prune.py."""


def prune(df):
    """Docstring for prune."""
    df = df[df["Close"].notna()]
    return df
