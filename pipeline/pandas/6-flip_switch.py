#!/usr/bin/env python3
"""Docstring for 6-flip_switch.py."""


def flip_switch(df):
    """Docstring for flip_switch."""
    df.sort_values("Timestamp", ascending=False, inplace=True)
    df = df.T
    return df
