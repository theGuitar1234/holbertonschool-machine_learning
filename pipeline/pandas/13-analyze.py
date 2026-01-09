#!/usr/bin/env python3
"""Docstring for 13-analyze.py."""


def analyze(df):
    """Docstring for analyze."""
    df_temp = df.drop("Timestamp", axis=1)
    return df_temp.describe()
