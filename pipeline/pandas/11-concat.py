#!/usr/bin/env python3
"""Docstring for 11-concat.py."""


index = __import__(&#39;10-index&#39;).index


import pandas as pd


def concat(df1, df2):
    """Docstring for concat."""
    df1 = index(df1)
    df2 = index(df2)
    df2 = df2.loc[:1417411920]
    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
