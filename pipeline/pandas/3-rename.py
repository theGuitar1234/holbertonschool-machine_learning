#!/usr/bin/env python3
"""Docstring for 3-rename.py."""

import pandas as pd


def rename(df):
    """Docstring for rename."""
    df.rename(columns={"Timestamp": "Datetime"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    df = df[["Datetime", "Close"]]
    return df
