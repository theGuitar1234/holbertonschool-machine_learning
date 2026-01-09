#!/usr/bin/env python3
"""Docstring for 12-hierarchy.py."""
import pandas as pd


index = __import__('10-index').index


def hierarchy(df1, df2):
    """Docstring for hierarchy."""
    df1 = index(df1)
    df2 = index(df2)
    df1 = df1.loc[1417411980:1417417980]
    df2 = df2.loc[1417411980:1417417980]
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    df.index = df.index.reorder_levels([1, 0])
    timestamps = df.index.get_level_values(0).unique()
    full_index = pd.MultiIndex.from_product(
            [timestamps, ["bitstamp", "coinbase"]], names=["Timestamp", None])
    df = df.reindex(full_index)
    df.sort_index(level=0, inplace=True, sort_remaining=False)
    return df
