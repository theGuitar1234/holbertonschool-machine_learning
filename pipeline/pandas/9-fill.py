#!/usr/bin/env python3
"""Docstring for 9-fill.py."""


def fill(df):
    """Docstring for fill."""
    df.drop("Weighted_Price", axis=1, inplace=True)
    df["Close"].fillna(method="ffill", inplace=True)
    df["High"].fillna(df["Close"], inplace=True)
    df["Low"].fillna(df["Close"], inplace=True)
    df["Open"].fillna(df["Close"], inplace=True)
    df["Volume_(BTC)"].fillna(0, inplace=True)
    df["Volume_(Currency)"].fillna(0, inplace=True)
    return df
