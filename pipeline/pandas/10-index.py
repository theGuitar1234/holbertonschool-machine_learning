#!/usr/bin/env python3
"""Docstring for 10-index.py."""


def index(df):
    """Docstring for index."""
    df.set_index(df["Timestamp"], inplace=True)
    df.drop("Timestamp", axis=1, inplace=True)
    return df
